# src/projectdavid/clients/synchronous_inference_wrapper.py
import asyncio
import queue
import threading
from contextlib import suppress
from typing import Generator, Optional

from projectdavid_common import UtilsInterface

LOG = UtilsInterface.LoggingUtility()


class SynchronousInferenceStream:
    """
    Refactored Sync-to-Async wrapper.
    Uses a background Thread + Queue pipeline to eliminate per-chunk
    event loop overhead and reduce latency (TTFT).
    """

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
        self.user_id = user_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.message_id = message_id
        self.run_id = run_id
        self.api_key = api_key

    def stream_chunks(
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout_per_chunk: float = 280.0,
        suppress_fc: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Sync generator that bridges the async inference client using a
        background thread. This prevents the event loop from being
        started/stopped repeatedly, which causes stuttering.
        """

        resolved_api_key = api_key or self.api_key
        # Thread-safe queue to bridge the background producer and sync consumer
        chunk_queue = queue.Queue(maxsize=512)
        # Sentinel to signal completion
        STOP_SIGNAL = object()

        def _producer_thread():
            """
            Background task that manages its own event loop to keep the
            network connection hot and flowing.
            """
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run_async_stream():
                try:
                    async for chk in self.inference_client.stream_inference_response(
                        provider=provider,
                        model=model,
                        api_key=resolved_api_key,
                        thread_id=self.thread_id,
                        message_id=self.message_id,
                        run_id=self.run_id,
                        assistant_id=self.assistant_id,
                    ):
                        chunk_queue.put(chk)
                except Exception as e:
                    LOG.error(f"[SyncStream] Async stream producer error: {e}")
                    chunk_queue.put(e)
                finally:
                    chunk_queue.put(STOP_SIGNAL)

            try:
                loop.run_until_complete(_run_async_stream())
            finally:
                loop.close()

        # 1. Start the network connection in a background thread immediately
        producer = threading.Thread(target=_producer_thread, daemon=True)
        producer.start()

        LOG.debug("[SyncStream] Background pipeline started (Unified Mode)")

        # 2. Consume from the queue in the main thread
        while True:
            try:
                # Use the provided timeout while waiting for the next item from the queue
                item = chunk_queue.get(timeout=timeout_per_chunk)

                if item is STOP_SIGNAL:
                    LOG.info("[SyncStream] Stream completed normally.")
                    break

                if isinstance(item, Exception):
                    raise item

                # Always attach run_id for front-end helpers
                item["run_id"] = self.run_id

                # Filter tool call arguments if requested
                if suppress_fc and item.get("type") == "call_arguments":
                    continue

                yield item

            except queue.Empty:
                LOG.error("[SyncStream] Timeout waiting for next chunk in queue.")
                break

            except Exception as exc:
                LOG.error("[SyncStream] Unexpected error in sync consumption: %s", exc)
                break

    def close(self) -> None:
        """Closes the underlying inference client."""
        with suppress(Exception):
            self.inference_client.close()

    @classmethod
    def shutdown_loop(cls) -> None:
        """
        Maintained for backwards compatibility,
        but each stream now handles its own lifecycle.
        """
        pass
