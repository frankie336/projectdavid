# src/projectdavid/clients/synchronous_inference_wrapper.py
import asyncio
import json
import re
from contextlib import suppress
from typing import Any, Dict, Generator, Optional, Union

import nest_asyncio
from projectdavid_common import ToolValidator, UtilsInterface

from projectdavid.events import (
    CodeExecutionGeneratedFileEvent,
    CodeExecutionOutputEvent,
    CodeStatusEvent,
    ComputerExecutionOutputEvent,
    ContentEvent,
    DecisionEvent,
    EngineerStatusEvent,
    HotCodeEvent,
    PlanEvent,
    ReasoningEvent,
    ResearchStatusEvent,
    ScratchpadEvent,
    StreamEvent,
    ToolCallRequestEvent,
    ToolInterceptEvent,
    WebStatusEvent,
)

LOG = UtilsInterface.LoggingUtility()


class SynchronousInferenceStream:
    """
    A per-request inference stream.

    IMPORTANT — Thread Safety:
    --------------------------
    This class must be instantiated fresh for every request. It must NOT be
    stored as a shared singleton on the client (e.g. as a cached property).
    Sharing an instance across concurrent requests causes .setup() to race-
    overwrite thread_id, run_id, and other fields mid-stream.

    Correct usage in Flask route:
        sync_stream = SynchronousInferenceStream.for_request(
            inference=client.inference,
            runs_client=client.runs,
            actions_client=client.actions,
            messages_client=client.messages,
            assistants_client=client.assistants,
        )
        sync_stream.setup(thread_id=..., assistant_id=..., message_id=...,
                          run_id=..., api_key=..., meta_data=...)
        for event in sync_stream.stream_events(model=...):
            ...
    """

    def __init__(self, inference) -> None:
        self.inference_client = inference
        self.user_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.assistant_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.run_id: Optional[str] = None
        self.api_key: Optional[str] = None
        self.meta_data: Optional[Dict[str, Any]] = None  # NEW

        self.runs_client: Any = None
        self.actions_client: Any = None
        self.messages_client: Any = None
        self.assistants_client: Any = None

        self.validator = ToolValidator()

    # ------------------------------------------------------------------
    # Factory: preferred way to create a ready-to-use, isolated instance
    # ------------------------------------------------------------------
    @classmethod
    def for_request(
        cls,
        inference: Any,
        runs_client: Any,
        actions_client: Any,
        messages_client: Any,
        assistants_client: Any,
    ) -> "SynchronousInferenceStream":
        """
        Create a fresh, fully-bound instance for a single request.
        Avoids shared-state races between concurrent requests.
        """
        instance = cls(inference)
        instance.bind_clients(
            runs_client=runs_client,
            actions_client=actions_client,
            messages_client=messages_client,
            assistants_client=assistants_client,
        )
        return instance

    def setup(
        self,
        thread_id: str,
        assistant_id: str,
        message_id: str,
        run_id: str,
        api_key: Optional[str] = None,  # NEW: now optional to match schema
        meta_data: Optional[Dict[str, Any]] = None,  # NEW
    ) -> None:
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.message_id = message_id
        self.run_id = run_id
        self.api_key = api_key
        self.meta_data = meta_data  # NEW

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
        model: str,
        *,
        api_key: Optional[str] = None,
        meta_data: Optional[Dict[str, Any]] = None,  # NEW
        timeout_per_chunk: float = 600.0,
        suppress_fc: bool = True,
    ) -> Generator[dict, None, None]:
        resolved_api_key = api_key or self.api_key
        resolved_meta_data = meta_data or self.meta_data  # NEW

        # Capture instance fields into locals NOW so a concurrent .setup()
        # on a (mis-)shared instance cannot mutate them mid-stream.
        thread_id = self.thread_id
        message_id = self.message_id
        run_id = self.run_id
        assistant_id = self.assistant_id

        async def _stream_chunks_async():
            async for chk in self.inference_client.stream_inference_response(
                model=model,
                api_key=resolved_api_key,
                thread_id=thread_id,
                message_id=message_id,
                run_id=run_id,
                assistant_id=assistant_id,
                meta_data=resolved_meta_data,  # NEW
                timeout=timeout_per_chunk,
            ):
                yield chk

        agen = _stream_chunks_async().__aiter__()
        # ---------------------------------------------------------
        # LOOP DETECTION LOGIC
        # ---------------------------------------------------------
        active_loop = None
        is_new_loop = False

        try:
            active_loop = asyncio.get_running_loop()
            nest_asyncio.apply(active_loop)
        except RuntimeError:
            active_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(active_loop)
            is_new_loop = True

        try:
            while True:
                try:
                    chunk = active_loop.run_until_complete(
                        asyncio.wait_for(agen.__anext__(), timeout=timeout_per_chunk)
                    )

                    chunk["run_id"] = run_id
                    if suppress_fc and chunk.get("type") == "call_arguments":
                        continue
                    yield chunk

                except StopAsyncIteration:
                    break
                except Exception as exc:
                    LOG.error(f"[SyncStream] Streaming error: {exc}", exc_info=True)
                    break
        finally:
            if is_new_loop and active_loop:
                try:
                    pending = asyncio.all_tasks(active_loop)
                    for task in pending:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            active_loop.run_until_complete(task)
                    active_loop.close()
                except Exception as e:
                    LOG.error(f"[SyncStream] Error cleanup loop: {e}")

    def stream_events(
        self,
        model: str,
        *,
        meta_data: Optional[Dict[str, Any]] = None,  # NEW
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
                model=model,
                meta_data=meta_data or self.meta_data,  # NEW
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
        """Maps raw API chunks to typed Event instances."""

        # --- 1. ENFORCE CONTRACT: UNWRAP MIXIN JSON ---
        if chunk.get("type") == "content" and isinstance(chunk.get("content"), str):
            text = chunk["content"].strip()
            if text.startswith("{") and text.endswith("}"):
                with suppress(Exception):
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        if (
                            parsed.get("type")
                            in {
                                "web_status",
                                "research_status",
                                "scratchpad_status",
                                "code_status",
                                "engineer_status",
                                "tool_intercept",
                                "status",
                                "error",
                            }
                            or "stream_type" in parsed
                        ):
                            chunk = parsed

        # --- 2. EXTRACT NESTED PAYLOADS ---
        stream_type = chunk.get("type", chunk.get("stream_type"))
        if stream_type in {"code_execution", "computer_execution", "delegation"}:
            payload = chunk.get("chunk", {})
            if isinstance(payload, dict):
                payload.setdefault("run_id", chunk.get("run_id"))
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

        elif c_type == "plan":
            return PlanEvent(run_id=run_id, content=chunk.get("content", ""))

        elif c_type == "hot_code":
            return HotCodeEvent(run_id=run_id, content=chunk.get("content", ""))

        elif c_type == "hot_code_output":
            return CodeExecutionOutputEvent(
                run_id=run_id, content=chunk.get("content", "")
            )

        elif c_type == "scratchpad_status":
            return ScratchpadEvent(
                run_id=run_id,
                operation=chunk.get("operation", "unknown"),
                state=chunk.get("state", "in_progress"),
                activity=chunk.get("activity"),
                tool=chunk.get("tool"),
                entry=chunk.get("entry"),
                content=chunk.get("content"),
                assistant_id=chunk.get("assistant_id"),
            )

        elif c_type == "computer_output":
            return ComputerExecutionOutputEvent(
                run_id=run_id, content=chunk.get("content", "")
            )

        elif c_type == "code_status":
            return CodeStatusEvent(
                run_id=run_id,
                activity=chunk.get("activity", ""),
                state=chunk.get("state", "in_progress"),
                tool=chunk.get("tool"),
            )

        elif c_type == "code_interpreter_file":
            return CodeExecutionGeneratedFileEvent(
                run_id=run_id,
                filename=chunk.get("filename", "unknown"),
                file_id=chunk.get("file_id"),
                base64_data=chunk.get("base64"),
                mime_type=chunk.get("mime_type", "application/octet-stream"),
                url=chunk.get("url"),
            )

        elif c_type == "research_status":
            return ResearchStatusEvent(
                run_id=run_id,
                activity=chunk.get("activity", ""),
                state=chunk.get("state", "in_progress"),
                tool=chunk.get("tool"),
            )

        elif c_type == "web_status":
            return WebStatusEvent(
                run_id=run_id,
                status=chunk.get("status", "running"),
                tool=chunk.get("tool"),
                message=chunk.get("message"),
            )

        elif c_type == "engineer_status":
            return EngineerStatusEvent(
                run_id=run_id,
                activity=chunk.get("activity") or chunk.get("message", ""),
                state=chunk.get("state") or chunk.get("status", "in_progress"),
                tool=chunk.get("tool"),
            )

        elif c_type == "tool_intercept":
            return ToolInterceptEvent(
                run_id=run_id,
                tool_name=chunk.get("tool_name", ""),
                args=chunk.get("args", {}),
                action_id=chunk.get("action_id"),
                origin=chunk.get("origin"),
                thread_id=chunk.get("thread_id"),
                tool_call_id=chunk.get("tool_call_id"),
                origin_run_id=chunk.get("origin_run_id"),
                origin_assistant_id=chunk.get("origin_assistant_id"),
            )

        elif c_type == "error":
            return WebStatusEvent(
                run_id=run_id,
                status="failed",
                tool=chunk.get("tool"),
                message=chunk.get("error") or chunk.get("message"),
            )

        return None
