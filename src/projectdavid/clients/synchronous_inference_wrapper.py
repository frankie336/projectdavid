# src/projectdavid/clients/synchronous_inference_wrapper.py
import asyncio
import json
from contextlib import suppress
from typing import Any, Generator, Optional, Union

from projectdavid_common import UtilsInterface

from .events import ContentEvent, StatusEvent, ToolCallRequestEvent

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

        # Client references for execution capability
        self.runs_client: Any = None
        self.actions_client: Any = None
        self.messages_client: Any = None

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

    def bind_clients(
        self, runs_client: Any, actions_client: Any, messages_client: Any
    ) -> None:
        """
        Injects the necessary clients to enable 'smart events' that can
        execute themselves. This should be called during Entity initialization.
        """
        self.runs_client = runs_client
        self.actions_client = actions_client
        self.messages_client = messages_client

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
        Yields raw dictionary chunks.
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
    #   High-Level Event Stream (Smart Iterator)
    # ------------------------------------------------------------ #
    def stream_events(
        self,
        provider: str,
        model: str,
        *,
        timeout_per_chunk: float = 280.0,
    ) -> Generator[Union[ContentEvent, ToolCallRequestEvent, StatusEvent], None, None]:
        """
        High-level iterator that yields Events instead of raw dicts.
        Automatically handles argument buffering, JSON parsing, and Qwen-unwrapping.
        """
        if not all([self.runs_client, self.actions_client, self.messages_client]):
            LOG.warning(
                "[SyncStream] Clients not bound. Tool execution events may fail."
            )

        # Buffers for accumulation
        tool_args_buffer = ""
        is_collecting_tool = False

        # Consume the raw chunks from the existing method
        # We set suppress_fc=False because we MUST capture the arguments here
        for chunk in self.stream_chunks(
            provider=provider,
            model=model,
            timeout_per_chunk=timeout_per_chunk,
            suppress_fc=False,
        ):
            c_type = chunk.get("type")
            run_id = chunk.get("run_id")

            # 1. Text Content
            if c_type == "content":
                yield ContentEvent(run_id=run_id, content=chunk.get("content", ""))

            # 2. Tool Argument Accumulation
            elif c_type == "call_arguments":
                is_collecting_tool = True
                tool_args_buffer += chunk.get("content", "")

            # 3. Status / Completion
            elif c_type == "status":
                status = chunk.get("status")

                # If we were collecting a tool and the stream signals completion
                if is_collecting_tool and status == "complete":
                    if tool_args_buffer:
                        try:
                            # A. Parse Raw JSON
                            captured_data = json.loads(tool_args_buffer)

                            # B. Unwrap (Robustness for Qwen/Nested args)
                            # Some models stream {"name": "func", "arguments": {...}}
                            # Others stream just {...}
                            if "arguments" in captured_data and isinstance(
                                captured_data["arguments"], dict
                            ):
                                final_args = captured_data["arguments"]
                                tool_name = captured_data.get("name", "unknown_tool")
                            else:
                                final_args = captured_data
                                tool_name = "unknown_tool"

                            # C. Yield the Executable Event
                            if self.runs_client:
                                yield ToolCallRequestEvent(
                                    run_id=run_id,
                                    tool_name=tool_name,
                                    args=final_args,
                                    thread_id=self.thread_id,
                                    assistant_id=self.assistant_id,
                                    _runs_client=self.runs_client,
                                    _actions_client=self.actions_client,
                                    _messages_client=self.messages_client,
                                )
                            else:
                                LOG.error(
                                    "[SyncStream] Cannot yield ToolCallRequestEvent: Clients not bound via bind_clients()"
                                )

                        except json.JSONDecodeError:
                            LOG.error(
                                f"[SyncStream] Failed to parse accumulated tool arguments: {tool_args_buffer}"
                            )

                    # Reset buffers
                    tool_args_buffer = ""
                    is_collecting_tool = False

                yield StatusEvent(run_id=run_id, status=status)

            # 4. Error Forwarding (Mapped to StatusEvent or separate ErrorEvent if preferred)
            elif c_type == "error":
                LOG.error(f"[SyncStream] Stream Error: {chunk}")
                # We yield a failed status event so the consumer loop knows something went wrong
                yield StatusEvent(run_id=run_id, status="failed")

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
