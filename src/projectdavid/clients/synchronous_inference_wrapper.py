import asyncio
from contextlib import suppress
from typing import Generator, Optional

from projectdavid_common import UtilsInterface

from projectdavid.utils.function_call_suppressor import FunctionCallSuppressor
from projectdavid.utils.peek_gate import PeekGate

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
    def stream_chunks(  # noqa: PLR0915
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout_per_chunk: float = 280.0,
        suppress_fc: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Sync generator that mirrors async `inference_client.stream_inference_response`
        but (optionally) removes raw <fc> … </fc> output *and* JSON
        `{"type": "function_call" …}` objects from the stream.
        """

        resolved_api_key = api_key or self.api_key

        # ---------- async inner generator -------------------------------- #
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

        # ---------- FC-suppressor plumbing -------------------------------- #
        if suppress_fc:
            _suppressor = FunctionCallSuppressor()
            _peek_gate = PeekGate(_suppressor)

            def _filter_text(txt: str) -> str:
                return _peek_gate.feed(txt)

            LOG.debug("[SyncStream] Function-call suppression ACTIVE")
        else:

            def _filter_text(txt: str) -> str:
                return txt

            LOG.debug("[SyncStream] Function-call suppression DISABLED")

        # ---------- helper to flush residual buffered text ---------------- #
        def _drain_filters() -> Optional[dict]:
            if not suppress_fc:
                return None
            parts: list[str] = []
            while True:
                out = _filter_text("")
                if not out:
                    break
                parts.append(out)
            if not _peek_gate.suppressing and _peek_gate.buf:
                parts.append(_peek_gate.buf)
                _peek_gate.buf = ""
            if parts:
                return {
                    "type": "content",
                    "content": "".join(parts),
                    "run_id": self.run_id,
                }
            return None

        # ---------- main sync loop ---------------------------------------- #
        while True:
            try:
                chunk = self._GLOBAL_LOOP.run_until_complete(
                    asyncio.wait_for(agen.__anext__(), timeout=timeout_per_chunk)
                )

                # Always attach run_id for front-end helpers
                chunk["run_id"] = self.run_id

                # ----- bypass filters for status / code-exec related -------- #
                if chunk.get("type") == "status":
                    yield chunk
                    continue

                if chunk.get("type") in ("hot_code", "hot_code_output"):
                    yield chunk
                    continue

                if (
                    chunk.get("stream_type") == "code_execution"
                    and chunk.get("chunk", {}).get("type") == "code_interpreter_stream"
                ):
                    yield chunk
                    continue

                # ----- NEW: swallow raw JSON function_call objects ---------- #
                if suppress_fc and chunk.get("type") == "function_call":
                    LOG.debug(
                        "[SyncStream] Swallowing JSON function_call chunk: %s",
                        chunk.get("name") or "<unnamed>",
                    )
                    continue

                # ----- text-level suppression ------------------------------- #
                if isinstance(chunk.get("content"), str):
                    chunk["content"] = _filter_text(chunk["content"])
                    if chunk["content"] == "":
                        # Entire segment was inside <fc> … </fc>
                        continue

                yield chunk

            except StopAsyncIteration:
                if tail := _drain_filters():
                    yield tail
                LOG.info("[SyncStream] Stream completed normally.")
                break

            except asyncio.TimeoutError:
                if tail := _drain_filters():
                    yield tail
                LOG.error("[SyncStream] Timeout waiting for next chunk.")
                break

            except Exception as exc:  # noqa: BLE001
                if tail := _drain_filters():
                    yield tail
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
