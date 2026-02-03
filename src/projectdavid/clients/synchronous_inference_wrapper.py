import asyncio
import json
from contextlib import suppress
from typing import Any, Generator, Optional, Union

# [FIX] Import nest_asyncio to allow nested event loops in Uvicorn/Flask
import nest_asyncio
from projectdavid_common import UtilsInterface

# Import all event types, including the new DecisionEvent
from projectdavid.events import (
    CodeExecutionGeneratedFileEvent,
    CodeExecutionOutputEvent,
    ComputerExecutionOutputEvent,
    ContentEvent,
    DecisionEvent,
    HotCodeEvent,
    ReasoningEvent,
    StatusEvent,
    StreamEvent,
    ToolCallRequestEvent,
)

LOG = UtilsInterface.LoggingUtility()


class SynchronousInferenceStream:
    # ------------------------------------------------------------ #
    #   GLOBAL EVENT LOOP  (Fallback for standalone scripts)
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

        # Determine the correct loop to use (standalone vs active web server)
        try:
            active_loop = asyncio.get_running_loop()
            nest_asyncio.apply(active_loop)
        except RuntimeError:
            active_loop = self._GLOBAL_LOOP

        while True:
            try:
                chunk = active_loop.run_until_complete(
                    asyncio.wait_for(agen.__anext__(), timeout=timeout_per_chunk)
                )

                # Always attach run_id for front-end helpers
                chunk["run_id"] = self.run_id

                if suppress_fc and chunk.get("type") == "call_arguments":
                    continue

                yield chunk

            except StopAsyncIteration:
                LOG.info("[SyncStream] Chunk stream completed.")
                break
            except asyncio.TimeoutError:
                LOG.error("[SyncStream] Timeout waiting for next chunk.")
                break
            except Exception as exc:
                LOG.error(f"[SyncStream] Streaming error: {exc}", exc_info=True)
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
        max_turns: int = 10,
    ) -> Generator[Union[StreamEvent, Any], None, None]:
        """
        High-level iterator that yields Events.
        Handles Turn 2+ automatically: if a tool is executed, it loops back
        to the model to get the next response.
        """
        if not all([self.runs_client, self.actions_client, self.messages_client]):
            LOG.warning("[SyncStream] Clients not bound. Tool execution will fail.")

        turn_count = 0
        while turn_count < max_turns:
            turn_count += 1
            last_tool_call: Optional[ToolCallRequestEvent] = None

            # Execute current turn
            for chunk in self.stream_chunks(
                provider=provider,
                model=model,
                timeout_per_chunk=timeout_per_chunk,
                suppress_fc=True,
            ):
                event = self._map_chunk_to_event(chunk)
                if not event:
                    continue

                yield event

                # Track if the model requested a tool
                if isinstance(event, ToolCallRequestEvent):
                    last_tool_call = event

            # End of stream check:
            # If a tool call was yielded and the user executed it, start next turn.
            if last_tool_call and last_tool_call.executed:
                LOG.info(
                    f"[SyncStream] Tool executed. Automatically starting turn {turn_count + 1}"
                )
                continue

            # If no tool was called, or it wasn't executed, the conversation turn is over.
            break

    def _map_chunk_to_event(self, chunk: dict) -> Optional[StreamEvent]:
        """Maps raw API chunks to Typed Event instances."""
        # --- 1. Unwrapping Mixins (Code/Computer) ---
        stream_type = chunk.get("stream_type")
        if stream_type in ["code_execution", "computer_execution"]:
            payload = chunk.get("chunk", {})
            if "run_id" not in payload:
                payload["run_id"] = chunk.get("run_id")
            chunk = payload

        c_type = chunk.get("type")
        run_id = chunk.get("run_id")

        # --- 2. Tool Call Manifest ---
        if c_type == "tool_call_manifest":
            return ToolCallRequestEvent(
                run_id=run_id,
                tool_name=chunk.get("tool", "unknown_tool"),
                args=chunk.get("args", {}),
                action_id=chunk.get("action_id"),
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                _runs_client=self.runs_client,
                _actions_client=self.actions_client,
                _messages_client=self.messages_client,
            )

        # --- 3. Content Deltas ---
        elif c_type == "content":
            return ContentEvent(run_id=run_id, content=chunk.get("content", ""))

        # --- 4. Reasoning (DeepSeek R1) ---
        elif c_type == "reasoning":
            return ReasoningEvent(run_id=run_id, content=chunk.get("content", ""))

        # --- 5. Structural Decisions ---
        elif c_type == "decision":
            return DecisionEvent(run_id=run_id, content=chunk.get("content", ""))

        # --- 6. Code Interpreter Events ---
        elif c_type == "hot_code":
            return HotCodeEvent(run_id=run_id, content=chunk.get("content", ""))
        elif c_type == "hot_code_output":
            return CodeExecutionOutputEvent(
                run_id=run_id, content=chunk.get("content", "")
            )
        elif c_type == "computer_output":
            return ComputerExecutionOutputEvent(
                run_id=run_id, content=chunk.get("content", "")
            )
        elif c_type == "code_interpreter_stream":
            file_data = chunk.get("content", {})
            return CodeExecutionGeneratedFileEvent(
                run_id=run_id,
                filename=file_data.get("filename", "unknown"),
                file_id=file_data.get("file_id"),
                base64_data=file_data.get("base64", ""),
                mime_type=file_data.get("mime_type", "application/octet-stream"),
            )

        # --- 7. Status & Errors ---
        elif c_type == "status":
            return StatusEvent(run_id=run_id, status=chunk.get("status"))
        elif c_type == "error":
            LOG.error(f"[SyncStream] Error chunk received: {chunk}")
            return StatusEvent(run_id=run_id, status="failed")

        return None

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
        """
        for event in self.stream_events(
            provider=provider, model=model, timeout_per_chunk=timeout_per_chunk
        ):
            yield json.dumps(event.to_dict())

    @classmethod
    def shutdown_loop(cls) -> None:
        if cls._GLOBAL_LOOP and not cls._GLOBAL_LOOP.is_closed():
            cls._GLOBAL_LOOP.stop()
            cls._GLOBAL_LOOP.close()

    def close(self) -> None:
        with suppress(Exception):
            self.inference_client.close()
