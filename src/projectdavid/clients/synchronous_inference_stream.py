import asyncio
from contextlib import suppress


class SynchronousInferenceStream:
    _GLOBAL_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_GLOBAL_LOOP)

    def __init__(self, inference):
        self.inference_client = inference
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

    def stream_chunks(self, provider: str, model: str, timeout_per_chunk: float = 10.0):
        async def _stream_chunks_async():
            async for chunk in self.inference_client.stream_inference_response(
                provider=provider,
                model=model,
                thread_id=self.thread_id,
                message_id=self.message_id,
                run_id=self.run_id,
                assistant_id=self.assistant_id,
                api_key=None,
            ):
                yield chunk

        gen = _stream_chunks_async().__aiter__()

        while True:
            try:
                chunk = self._GLOBAL_LOOP.run_until_complete(
                    asyncio.wait_for(gen.__anext__(), timeout=timeout_per_chunk)
                )
                yield chunk
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                print("[TimeoutError] Timeout occurred, stopping stream.")
                break
            except Exception as e:
                print(f"[Error] Exception during streaming: {e}")
                break

    @classmethod
    def shutdown_loop(cls):
        if cls._GLOBAL_LOOP and not cls._GLOBAL_LOOP.is_closed():
            cls._GLOBAL_LOOP.stop()
            cls._GLOBAL_LOOP.close()

    def close(self):
        with suppress(Exception):
            self.inference_client.close()
