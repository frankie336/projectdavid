# src/projectdavid/clients/synchronous_inference_wrapper.py
import asyncio
from contextlib import suppress
from typing import Generator, Optional

from projectdavid_common import UtilsInterface

# StreamRefiner removed as categorization is now handled at the provider level
LOG = UtilsInterface.LoggingUtility()


class SynchronousInferenceStream:
    # ------------------------------------------------------------ #
    #   GLOBAL EVENT LOOP  (single hidden thread for sync wrapper)
    # ------------------------------------------------------------ #
    _GLOBAL_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_GLOBAL_LOOP)

    # ------------------------------------------------------------ #
    #   Init / setup
    # ------------------------------------------------------------ #
    def __init__(self, inference) -> None:
        self.inference_client = inference
        self.user_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.assistant_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.run_id: Optional[str] = None
        self.api_key: Optional[str] = None

    def setup(
        self,
        user_id: str,
        thread_id: str,
        assistant_id: str,
        message_id: str,
        run_id: str,
        api_key: str,
    ) -> None:
        """Populate IDs once, so callers only provide provider/model."""
        self.user_id = user_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.message_id = message_id
        self.run_id = run_id
        self.api_key = api_key

    # ------------------------------------------------------------ #
    #   Core sync-to-async streaming wrapper
    # ------------------------------------------------------------ #
    def stream_chunks(
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout_per_chunk: float = 280.0,
        suppress_fc: bool = True,  # Note: Now primarily a hint for the consumer
    ) -> Generator[dict, None, None]:
        """
        Sync generator that mirrors async `inference_client.stream_inference_response`.

        Refined Logic:
        We no longer use string-based Refiners. Chunks are now typed at the source
        (e.g., 'content', 'call_arguments', 'reasoning', 'hot_code').
        The consumer handles visibility based on these types.
        """

        resolved_api_key = api_key or self.api_key

        async def _stream_chunks_async():
            async for chk in self.inference_client.stream_inference_response(
                provider=provider,
                model=model,
                api_key=resolved_api_key,
                thread_id=self.thread_id,
                message_id=self.message_id,
                run_id=self.run_id,
                assistant_id=self.assistant_id,
            ):
                yield chk

        agen = _stream_chunks_async().__aiter__()

        LOG.debug("[SyncStream] Starting typed stream (Unified Orchestration Mode)")

        while True:
            try:
                chunk = self._GLOBAL_LOOP.run_until_complete(
                    asyncio.wait_for(agen.__anext__(), timeout=timeout_per_chunk)
                )

                # Always attach run_id for front-end helpers
                chunk["run_id"] = self.run_id

                # In the new typed architecture, we yield everything.
                # If the provider is doing its job, 'call_arguments' are already
                # separated from 'content'.

                # Logic check: If for some reason we still want the SDK to enforce
                # suppression of tool-call arguments, we can do it via the type key.
                if suppress_fc and chunk.get("type") == "call_arguments":
                    continue

                yield chunk

            except StopAsyncIteration:
                LOG.info("[SyncStream] Stream completed normally.")
                break

            except asyncio.TimeoutError:
                LOG.error("[SyncStream] Timeout waiting for next chunk.")
                break

            except Exception as exc:
                LOG.error(
                    "[SyncStream] Unexpected streaming error: %s", exc, exc_info=True
                )
                break

    # ------------------------------------------------------------ #
    #   House-keeping
    # ------------------------------------------------------------ #
    @classmethod
    def shutdown_loop(cls) -> None:
        if cls._GLOBAL_LOOP and not cls._GLOBAL_LOOP.is_closed():
            cls._GLOBAL_LOOP.stop()
            cls._GLOBAL_LOOP.close()

    def close(self) -> None:
        with suppress(Exception):
            self.inference_client.close()
