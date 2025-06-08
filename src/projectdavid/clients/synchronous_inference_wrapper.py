import asyncio
from contextlib import suppress
from typing import Generator, Optional

from projectdavid_common import UtilsInterface

from projectdavid.utils.function_call_suppressor import FunctionCallSuppressor
from projectdavid.utils.peek_gate import PeekGate

LOG = UtilsInterface.LoggingUtility()


class SynchronousInferenceStream:
    """Wrap an async streaming generator and expose it synchronously."""

    _GLOBAL_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_GLOBAL_LOOP)

    # ------------------------------------------------------------------ #
    # construction / setup
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # main streaming entry-point
    # ------------------------------------------------------------------ #
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
        Yield provider chunks synchronously.  When *suppress_fc* is True we

        1. completely drop top-level `type="function_call"` chunks, and
        2. scrub inline `<fc> … </fc>` sections inside text using the
           PeekGate + FunctionCallSuppressor chain.
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

        # ---------- build inline filter --------------------------------
        if suppress_fc:
            suppressor = FunctionCallSuppressor()
            peek_gate = PeekGate(suppressor)

            def _filter_text(txt: str) -> str:
                return peek_gate.feed(txt)

        else:

            def _filter_text(txt: str) -> str:  # no-op
                return txt

        # ---------- main loop ------------------------------------------
        while True:
            try:
                chunk = self._GLOBAL_LOOP.run_until_complete(
                    asyncio.wait_for(agen.__anext__(), timeout=timeout_per_chunk)
                )

                # ① drop provider-labelled function_call objects
                if suppress_fc and chunk.get("type") == "function_call":
                    LOG.debug("[SUPPRESSOR] stripped top-level function_call chunk")
                    continue

                # ② never touch hot-code or code-interpreter file previews
                if chunk.get("type") == "hot_code":
                    yield chunk
                    continue

                if (
                    chunk.get("stream_type") == "code_execution"
                    and chunk.get("chunk", {}).get("type") == "code_interpreter_stream"
                ):
                    yield chunk
                    continue

                # ③ filter inline text (<fc> blocks) *only* when content is str
                if isinstance(chunk.get("content"), str):
                    chunk["content"] = _filter_text(chunk["content"])
                    if chunk["content"] == "":
                        # Either the text is still in PeekGate’s buffer
                        # or it was fully suppressed – skip for now.
                        continue

                yield chunk

            except StopAsyncIteration:
                LOG.info("Stream completed normally.")
                break
            except asyncio.TimeoutError:
                LOG.error(
                    "[Timeout] chunk wait exceeded %.1f s – aborting", timeout_per_chunk
                )
                break
            except Exception as exc:  # pylint: disable=broad-except
                LOG.error("Unexpected streaming error: %s", exc, exc_info=True)
                break

    # ------------------------------------------------------------------ #
    # housekeeping
    # ------------------------------------------------------------------ #
    @classmethod
    def shutdown_loop(cls) -> None:
        if cls._GLOBAL_LOOP and not cls._GLOBAL_LOOP.is_closed():
            cls._GLOBAL_LOOP.stop()
            cls._GLOBAL_LOOP.close()

    def close(self) -> None:
        with suppress(Exception):
            self.inference_client.close()
