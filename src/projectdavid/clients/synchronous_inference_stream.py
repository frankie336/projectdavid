import asyncio
from contextlib import suppress
from typing import AsyncGenerator, Optional

from projectdavid_common import UtilsInterface

logging_utility = UtilsInterface.LoggingUtility()


class SynchronousInferenceStream:
    _GLOBAL_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_GLOBAL_LOOP)

    def __init__(self, inference) -> None:
        self.inference_client = inference
        self.user_id: str = ""
        self.thread_id: str = ""
        self.assistant_id: str = ""
        self.message_id: str = ""
        self.run_id: str = ""

    def setup(
        self,
        user_id: str,
        thread_id: str,
        assistant_id: str,
        message_id: str,
        run_id: str,
    ) -> None:
        self.user_id = user_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.message_id = message_id
        self.run_id = run_id

    def stream_chunks(
        self,
        provider: str,
        model: str,
        *,  # Enforce keyword-only for the following parameters
        api_key: Optional[str] = None,
        timeout_per_chunk: float = 10.0,
    ) -> AsyncGenerator[dict, None]:
        """
        Streams inference response chunks synchronously by wrapping an async generator.

        Args:
            provider (str): The provider name.
            model (str): The model name.
            api_key (Optional[str]): API key for authentication. Defaults to None.
            timeout_per_chunk (float): Timeout per chunk in seconds.

        Yields:
            dict: A chunk of the inference response.
        """

        async def _stream_chunks_async() -> AsyncGenerator[dict, None]:
            async for chunk in self.inference_client.stream_inference_response(
                provider=provider,
                api_key=api_key,
                model=model,
                thread_id=self.thread_id,
                message_id=self.message_id,
                run_id=self.run_id,
                assistant_id=self.assistant_id,
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
                logging_utility.error(
                    "[TimeoutError] Timeout occurred, stopping stream."
                )
                break
            except Exception as e:
                logging_utility.error(f"[Error] Exception during streaming: {e}")
                break

    @classmethod
    def shutdown_loop(cls) -> None:
        if cls._GLOBAL_LOOP and not cls._GLOBAL_LOOP.is_closed():
            cls._GLOBAL_LOOP.stop()
            cls._GLOBAL_LOOP.close()

    def close(self) -> None:
        with suppress(Exception):
            self.inference_client.close()
