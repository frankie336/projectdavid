import asyncio
from contextlib import suppress
from typing import AsyncGenerator, Optional  # Added AsyncGenerator


class SynchronousInferenceStream:
    _GLOBAL_LOOP = asyncio.new_event_loop()
    # Setting the event loop for the current thread might be necessary
    # depending on how threads interact with this class.
    # If only used in the main thread, set_event_loop here is fine.
    # If used across threads, managing loops per thread might be needed.
    try:
        asyncio.set_event_loop(_GLOBAL_LOOP)
    except RuntimeError as e:
        # Might fail if a loop is already set for the thread
        print(
            f"[Warning] Could not set global event loop: {e}. Using existing loop if available."
        )
        _GLOBAL_LOOP = asyncio.get_event_loop()  # Try to get the existing one

    def __init__(self, inference_client):  # Renamed parameter for clarity
        self.inference_client = inference_client
        self.user_id = None
        self.thread_id = None
        self.assistant_id = None
        self.message_id = None
        self.run_id = None
        # Check if the loop we intend to use is running, start if not?
        # This depends on application structure. Usually, the loop
        # should be managed externally or started when needed.
        # if not self._GLOBAL_LOOP.is_running():
        #    # This is tricky in a synchronous context. run_forever blocks.
        #    # run_until_complete is used per task.
        #    pass

    def setup(
        self,
        user_id: str,
        thread_id: str,
        assistant_id: str,
        message_id: str,
        run_id: str,
    ):
        self.user_id = user_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.message_id = message_id
        self.run_id = run_id

    def stream_chunks(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_per_chunk: float = 10.0,
    ):
        """
        Streams inference response chunks synchronously.

        Args:
            provider: The inference provider name.
            model: The model name.
            api_key: An optional API key specific to this request.
            timeout_per_chunk: Maximum time to wait for the next chunk.
        """
        # Ensure the loop is available and not closed before starting
        if not self._GLOBAL_LOOP or self._GLOBAL_LOOP.is_closed():
            print("[Error] Event loop is not available or closed. Cannot stream.")
            # Option 1: Raise an error
            raise RuntimeError("SynchronousInferenceStream event loop is closed.")
            # Option 2: Try to create/set a new loop (might have side effects)
            # print("[Warning] Event loop was closed. Creating a new one.")
            # self._GLOBAL_LOOP = asyncio.new_event_loop()
            # asyncio.set_event_loop(self._GLOBAL_LOOP)
            # Option 3: Silently return/yield nothing (might hide issues)
            # return

        async def _stream_chunks_async() -> (
            AsyncGenerator[dict, None]
        ):  # Added type hint
            # Pass the api_key down to the underlying async method
            async for chunk in self.inference_client.stream_inference_response(
                provider=provider,
                model=model,
                thread_id=self.thread_id,
                message_id=self.message_id,
                run_id=self.run_id,
                assistant_id=self.assistant_id,
                api_key=api_key,
            ):
                yield chunk

        gen = _stream_chunks_async().__aiter__()

        while True:
            try:
                # Get the next item from the async generator using the loop
                chunk_task = gen.__anext__()
                # Run the task with a timeout
                chunk = self._GLOBAL_LOOP.run_until_complete(
                    asyncio.wait_for(chunk_task, timeout=timeout_per_chunk)
                )
                yield chunk
            except StopAsyncIteration:
                # print("[Debug] Stream finished normally.") # Optional
                break
            except asyncio.TimeoutError:
                print(
                    f"[TimeoutError] Timeout ({timeout_per_chunk}s) waiting for next chunk, stopping stream."
                )
                # Optionally cancel the underlying generator task if needed, though breaking is often sufficient
                # if 'chunk_task' in locals() and not chunk_task.done():
                #     chunk_task.cancel()
                #     with suppress(asyncio.CancelledError):
                #         self._GLOBAL_LOOP.run_until_complete(chunk_task)
                break
            except RuntimeError as e:
                # Catch cases like "Event loop is closed" if it happens mid-stream
                print(f"[Error] Runtime error during streaming (loop closed?): {e}")
                import traceback

                traceback.print_exc()
                break
            except Exception as e:
                print(f"[Error] Exception during streaming: {e}")
                import traceback

                traceback.print_exc()
                break

    @classmethod
    def shutdown_loop(cls):
        """
        Gracefully shuts down the shared asyncio event loop.
        Cancels pending tasks and closes the loop.
        """
        loop = cls._GLOBAL_LOOP
        if not loop or loop.is_closed():
            print("[Info] Global event loop is already closed or not set.")
            return

        if not loop.is_running():
            print("[Info] Global event loop is not running. Closing directly.")
            with suppress(Exception):  # Suppress errors during close
                loop.close()
            print("[Info] Global event loop closed.")
            # Ensure the class attribute reflects the closed state if needed
            # cls._GLOBAL_LOOP = None # Or mark as closed
            return

        print("[Info] Shutting down synchronous event loop gracefully...")

        try:
            # Gather all pending tasks in the loop
            # Use shield to prevent gather itself from being cancelled if shutdown is called from within the loop
            tasks = asyncio.all_tasks(loop=loop)
            pending_tasks = {task for task in tasks if not task.done()}
            print(f"[Debug] Found {len(pending_tasks)} pending tasks to cancel.")

            if not pending_tasks:
                print("[Info] No pending tasks found.")
                # If no tasks are pending, we might still need to stop the loop if running
                if loop.is_running():
                    loop.stop()
                with suppress(Exception):
                    loop.close()
                print("[Info] Global event loop stopped and closed.")
                return

            # Cancel all pending tasks
            print("[Info] Cancelling pending tasks...")
            for task in pending_tasks:
                task.cancel()

            # Create a future to gather the results of cancelled tasks
            # return_exceptions=True ensures gather doesn't fail if a task raises something during cancellation cleanup
            print("[Info] Gathering cancelled tasks...")
            cancelled_tasks_group = asyncio.gather(
                *pending_tasks, return_exceptions=True
            )

            # Run the loop until the gather future is complete
            # This allows tasks to handle CancelledError and run finally blocks
            loop.run_until_complete(cancelled_tasks_group)
            print("[Info] Finished gathering cancelled tasks.")

        except Exception as e:
            print(
                f"[Error] Exception during loop shutdown task cancellation/gathering: {e}"
            )
            import traceback

            traceback.print_exc()
        finally:
            # Stop the loop if it's somehow still running (shouldn't be after run_until_complete finishes)
            if loop.is_running():
                print("[Debug] Stopping loop after gathering tasks.")
                loop.stop()

            # Finally, close the loop
            print("[Info] Closing the event loop.")
            with suppress(Exception):  # Suppress errors during the final close
                loop.close()
            print("[Info] Global event loop closed.")
            # Mark as potentially unusable?
            # cls._GLOBAL_LOOP = None

    def close(self):
        """Closes the underlying inference client if possible."""
        print("[Info] Closing SynchronousInferenceStream wrapper.")
        with suppress(Exception):
            if hasattr(self.inference_client, "close") and callable(
                self.inference_client.close
            ):
                self.inference_client.close()
        # Loop shutdown is managed globally via the classmethod, not per instance typically.
