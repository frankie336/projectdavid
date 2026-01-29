import asyncio
import json
from contextlib import suppress
from typing import Any, Generator, Optional, Union

from projectdavid_common import UtilsInterface

# Import all event types
from projectdavid.events import (
    CodeExecutionGeneratedFileEvent,
    CodeExecutionOutputEvent,
    ComputerExecutionOutputEvent,
    ContentEvent,
    HotCodeEvent,
    ReasoningEvent,
    StatusEvent,
    ToolCallRequestEvent,
)

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
        suppress_fc: bool = True,
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
    ) -> Generator[
        Union[
            ContentEvent,
            ToolCallRequestEvent,
            StatusEvent,
            ReasoningEvent,
            HotCodeEvent,
            CodeExecutionOutputEvent,
            CodeExecutionGeneratedFileEvent,
            ComputerExecutionOutputEvent,
        ],
        None,
        None,
    ]:
        """
        High-level iterator that yields Events instead of raw dicts.
        Handles buffering, parsing, unwrapping (Code/Computer), and execution prep.
        """
        if not all([self.runs_client, self.actions_client, self.messages_client]):
            LOG.warning(
                "[SyncStream] Clients not bound. Tool execution events may fail."
            )

        # [REMOVED] Legacy local accumulation buffers
        # tool_args_buffer = ""
        # is_collecting_tool = False

        for chunk in self.stream_chunks(
            provider=provider,
            model=model,
            timeout_per_chunk=timeout_per_chunk,
            # [FIX] Suppress raw args so we don't trigger legacy local reconstruction
            suppress_fc=True,
        ):
            # ----------------------------------------------------
            # UNWRAPPING LOGIC for Code & Computer Mixins
            # ----------------------------------------------------
            stream_type = chunk.get("stream_type")

            if stream_type in ["code_execution", "computer_execution"]:
                payload = chunk.get("chunk", {})
                if "run_id" not in payload:
                    payload["run_id"] = chunk.get("run_id")
                chunk = payload
            # ----------------------------------------------------

            c_type = chunk.get("type")
            run_id = chunk.get("run_id")

            # --- 1. Tool Call Manifest (THE AUTHORITATIVE EVENT) ---
            if c_type == "tool_call_manifest":
                tool_name = chunk.get("tool", "unknown_tool")
                final_args = chunk.get("args", {})
                action_id = chunk.get("action_id")

                if self.runs_client:
                    yield ToolCallRequestEvent(
                        run_id=run_id,
                        tool_name=tool_name,
                        args=final_args,
                        action_id=action_id,
                        thread_id=self.thread_id,
                        assistant_id=self.assistant_id,
                        _runs_client=self.runs_client,
                        _actions_client=self.actions_client,
                        _messages_client=self.messages_client,
                    )
                continue

            # --- 2. Standard Content ---
            elif c_type == "content":
                yield ContentEvent(run_id=run_id, content=chunk.get("content", ""))

            # --- 3. Reasoning (DeepSeek) ---
            elif c_type == "reasoning":
                yield ReasoningEvent(run_id=run_id, content=chunk.get("content", ""))

            # --- 4. Code Execution: Hot Code (Typing) ---
            elif c_type == "hot_code":
                yield HotCodeEvent(run_id=run_id, content=chunk.get("content", ""))

            # --- 5. Code Execution: Output (Stdout/Stderr) ---
            elif c_type == "hot_code_output":
                yield CodeExecutionOutputEvent(
                    run_id=run_id, content=chunk.get("content", "")
                )

            # --- 6. Computer/Shell Execution Output ---
            elif c_type == "computer_output":
                yield ComputerExecutionOutputEvent(
                    run_id=run_id, content=chunk.get("content", "")
                )

            # --- 7. Code Execution: Generated Files ---
            elif c_type == "code_interpreter_stream":
                file_data = chunk.get("content", {})
                yield CodeExecutionGeneratedFileEvent(
                    run_id=run_id,
                    filename=file_data.get("filename", "unknown"),
                    file_id=file_data.get("file_id"),
                    base64_data=file_data.get("base64", ""),
                    mime_type=file_data.get("mime_type", "application/octet-stream"),
                )

            # --- 8. Legacy Tool Accumulation (DISABLED) ---
            # elif c_type == "call_arguments":
            #     is_collecting_tool = True
            #     tool_args_buffer += chunk.get("content", "")

            # --- 9. Status / Completion ---
            elif c_type == "status":
                status = chunk.get("status")

                # [FIX] Disabled local synthesis of tool events on 'complete'
                # if is_collecting_tool and status == "complete":
                #     ... (Legacy logic removed to prevent duplicates) ...

                yield StatusEvent(run_id=run_id, status=status)

            # --- 10. Error ---
            elif c_type == "error":
                LOG.error(f"[SyncStream] Stream Error: {chunk}")
                yield StatusEvent(run_id=run_id, status="failed")

    # ------------------------------------------------------------ #
    #   Typed JSON Stream (Front-end Handover)
    # ------------------------------------------------------------ #
    def stream_typed_json(
        self,
        provider: str,
        model: str,
        *,
        timeout_per_chunk: float = 280.0,
    ) -> Generator[str, None, None]:
        """
        Consumes high-level Events and yields serialized JSON strings.
        Ensures every chunk has a 'type' discriminator for the UI.
        """
        for event in self.stream_events(
            provider=provider, model=model, timeout_per_chunk=timeout_per_chunk
        ):
            # event.to_dict() must be implemented in projectdavid.events
            yield json.dumps(event.to_dict())

    @classmethod
    def shutdown_loop(cls) -> None:
        if cls._GLOBAL_LOOP and not cls._GLOBAL_LOOP.is_closed():
            cls._GLOBAL_LOOP.stop()
            cls._GLOBAL_LOOP.close()

    def close(self) -> None:
        with suppress(Exception):
            self.inference_client.close()
