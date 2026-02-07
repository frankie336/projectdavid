import asyncio
import json
import re
from contextlib import suppress
from typing import Any, Dict, Generator, List, Optional, Union

import nest_asyncio
from projectdavid_common import ToolValidator, UtilsInterface

# Import the PlanEvent along with other types
from projectdavid.events import PlanEvent  # [NEW]
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
    _GLOBAL_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_GLOBAL_LOOP)

    def __init__(self, inference) -> None:
        self.inference_client = inference
        self.user_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.assistant_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.run_id: Optional[str] = None
        self.api_key: Optional[str] = None

        self.runs_client: Any = None
        self.actions_client: Any = None
        self.messages_client: Any = None
        self.assistants_client: Any = None

        self.validator = ToolValidator()

    def setup(
        self,
        thread_id: str,
        assistant_id: str,
        message_id: str,
        run_id: str,
        api_key: str,
    ) -> None:
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.message_id = message_id
        self.run_id = run_id
        self.api_key = api_key

    def bind_clients(
        self,
        runs_client: Any,
        actions_client: Any,
        messages_client: Any,
        assistants_client: Any,
    ) -> None:
        self.runs_client = runs_client
        self.actions_client = actions_client
        self.messages_client = messages_client
        self.assistants_client = assistants_client

    def stream_chunks(
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout_per_chunk: float = 280.0,
        suppress_fc: bool = True,
    ) -> Generator[dict, None, None]:
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
                chunk["run_id"] = self.run_id
                if suppress_fc and chunk.get("type") == "call_arguments":
                    continue
                yield chunk
            except StopAsyncIteration:
                break
            except Exception as exc:
                LOG.error(f"[SyncStream] Streaming error: {exc}", exc_info=True)
                break

    def stream_events(
        self,
        provider: str,
        model: str,
        *,
        timeout_per_chunk: float = 280.0,
        max_turns: int = 10,
    ) -> Generator[Union[StreamEvent, Any], None, None]:

        if self.assistant_id and self.assistants_client:
            try:
                ast = self.assistants_client.retrieve_assistant(self.assistant_id)
                tools = (
                    getattr(ast, "tools", [])
                    if not isinstance(ast, dict)
                    else ast.get("tools", [])
                )
                self.validator.build_registry_from_assistant(tools)
            except Exception as e:
                LOG.warning(
                    f"[SyncStream] Failed to retrieve Assistant tool schemas: {e}"
                )

        turn_count = 0
        while turn_count < max_turns:
            turn_count += 1
            last_tool_call: Optional[ToolCallRequestEvent] = None
            validation_failed_this_turn = False

            for chunk in self.stream_chunks(
                provider=provider,
                model=model,
                timeout_per_chunk=timeout_per_chunk,
                suppress_fc=True,
            ):
                event = self._map_chunk_to_event(chunk)
                if not event:
                    continue

                if isinstance(event, ToolCallRequestEvent):
                    error_msg = self.validator.validate_args(
                        event.tool_name, event.args
                    )
                    if error_msg:
                        LOG.warning(f"[SDK] Intercepted invalid tool call: {error_msg}")
                        self.messages_client.submit_tool_output(
                            thread_id=self.thread_id,
                            content=error_msg,
                            role="tool",
                            assistant_id=self.assistant_id,
                            tool_id=event.action_id,
                        )
                        event.executed = True
                        last_tool_call = event
                        validation_failed_this_turn = True
                        break

                yield event
                if isinstance(event, ToolCallRequestEvent):
                    last_tool_call = event

            if validation_failed_this_turn or (
                last_tool_call and last_tool_call.executed
            ):
                LOG.info(
                    f"[SyncStream] Self-Correction triggered. Turn {turn_count + 1}"
                )
                continue

            break

    def _map_chunk_to_event(self, chunk: dict) -> Optional[StreamEvent]:
        """Maps raw API chunks to Typed Event instances."""
        stream_type = chunk.get("type") if "type" in chunk else chunk.get("stream_type")
        if stream_type in ["code_execution", "computer_execution"]:
            payload = chunk.get("chunk", {})
            if "run_id" not in payload:
                payload["run_id"] = chunk.get("run_id")
            chunk = payload

        c_type = chunk.get("type")
        run_id = chunk.get("run_id")

        if c_type == "tool_call_manifest":
            raw_args = chunk.get("args", {})
            if isinstance(raw_args, str):
                try:
                    match = re.search(r"(\{.*\}|\[.*\])", raw_args, re.DOTALL)
                    if match:
                        raw_args = json.loads(match.group(1))
                except Exception:
                    LOG.warning(f"[SDK] Stage 1 Healing failed for args: {raw_args}")

            return ToolCallRequestEvent(
                run_id=run_id,
                tool_name=chunk.get("tool", "unknown_tool"),
                args=raw_args,
                action_id=chunk.get("action_id"),
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                _runs_client=self.runs_client,
                _actions_client=self.actions_client,
                _messages_client=self.messages_client,
            )

        elif c_type == "content":
            return ContentEvent(run_id=run_id, content=chunk.get("content", ""))

        elif c_type == "reasoning":
            return ReasoningEvent(run_id=run_id, content=chunk.get("content", ""))

        elif c_type == "decision":
            return DecisionEvent(run_id=run_id, content=chunk.get("content", ""))

        elif c_type == "plan":  # [NEW] Map Plan chunks to PlanEvent
            return PlanEvent(run_id=run_id, content=chunk.get("content", ""))

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

        elif c_type == "status":
            return StatusEvent(run_id=run_id, status=chunk.get("status"))

        elif c_type == "error":
            return StatusEvent(run_id=run_id, status="failed")

        return None

    def stream_typed_json(
        self, provider: str, model: str, *, timeout_per_chunk: float = 280.0
    ):
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
