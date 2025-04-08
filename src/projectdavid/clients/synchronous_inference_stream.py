import asyncio
from contextlib import suppress
from typing import AsyncGenerator, Optional  # Added Optional, AsyncGenerator


class SynchronousInferenceStream:
    # Use a single event loop for all instances of this synchronous wrapper
    _GLOBAL_LOOP = asyncio.new_event_loop()
    # Set this loop as the active one for the thread that creates the first instance
    # Be mindful if using this class across multiple threads without explicit loop management per thread.
    try:
        asyncio.set_event_loop(_GLOBAL_LOOP)
    except RuntimeError as e:
        print(
            f"[Warning] Could not set global event loop: {e}. Using existing loop if available."
        )
        _GLOBAL_LOOP = asyncio.get_event_loop()  # Fallback to existing loop

    def __init__(self, inference_client):  # Renamed parameter for clarity
        self.inference_client = inference_client
        self.user_id = None
        self.thread_id = None
        self.assistant_id = None
        self.message_id = None
        self.run_id = None

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
        api_key: Optional[str] = None,  # <-- ADDED: Optional api_key parameter
        timeout_per_chunk: float = 10.0,
    ):
        """
        Streams inference response chunks synchronously.

        Args:
            provider: The inference provider name.
            model: The model name.
            api_key: An optional API key specific to this request. If provided,
                     it will be included in the payload sent to the inference service.
            timeout_per_chunk: Maximum time to wait for the next chunk.
        """
        # Ensure the loop is available and not closed before starting
        if not self._GLOBAL_LOOP or self._GLOBAL_LOOP.is_closed():
            print("[Error] Event loop is not available or closed. Cannot stream.")
            raise RuntimeError("SynchronousInferenceStream event loop is closed.")

        async def _stream_chunks_async() -> AsyncGenerator[dict, None]:
            # Pass the api_key down to the underlying async method
            async for chunk in self.inference_client.stream_inference_response(
                provider=provider,
                model=model,
                thread_id=self.thread_id,
                message_id=self.message_id,
                run_id=self.run_id,
                assistant_id=self.assistant_id,
                api_key=api_key,  # <-- ADDED: Pass the key here
            ):
                yield chunk

        # Get the async iterator from the async generator function
        gen = _stream_chunks_async().__aiter__()

        while True:
            try:
                # Run the next step of the async iterator in the event loop
                # with the specified timeout
                chunk = self._GLOBAL_LOOP.run_until_complete(
                    asyncio.wait_for(gen.__anext__(), timeout=timeout_per_chunk)
                )
                yield chunk
            except StopAsyncIteration:
                # The async generator finished, break the synchronous loop
                break
            except asyncio.TimeoutError:
                # Timeout occurred waiting for the next chunk
                print(
                    f"[TimeoutError] Timeout ({timeout_per_chunk}s) waiting for next chunk, stopping stream."
                )
                break
            except RuntimeError as e:
                # Catch cases like "Event loop is closed" if it happens mid-stream
                print(f"[Error] Runtime error during streaming (loop closed?): {e}")
                import traceback

                traceback.print_exc()
                break
            except Exception as e:
                # Catch any other unexpected errors during streaming
                print(f"[Error] Exception during streaming: {e}")
                import traceback

                traceback.print_exc()
                break

    @classmethod
    def shutdown_loop(cls):
        """
        Stops and closes the shared asyncio event loop.
        WARNING: Call this only when sure the loop is no longer needed by any
                 instance or part of the application.
        """
        loop = cls._GLOBAL_LOOP
        if loop and not loop.is_closed():
            print("[Info] Shutting down synchronous event loop.")
            # Stop the loop if it's running
            if loop.is_running():
                loop.stop()
            # Close the loop
            with suppress(Exception):  # Suppress potential errors during close
                loop.close()
            print("[Info] Synchronous event loop closed.")
            # Optionally clear the class attribute
            # cls._GLOBAL_LOOP = None

    def close(self):
        """Closes the underlying inference client if possible."""
        print("[Info] Closing SynchronousInferenceStream wrapper.")
        # Attempt to close the inference client if it has a 'close' method
        with suppress(Exception):
            if hasattr(self.inference_client, "close") and callable(
                self.inference_client.close
            ):
                self.inference_client.close()
        # Note: Instance closing does not shut down the shared _GLOBAL_LOOP.
        # Use the classmethod shutdown_loop() for that when appropriate.
