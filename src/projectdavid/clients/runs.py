# src/projectdavid/clients/runs.py
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
import requests
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.validation import StatusEnum, TruncationStrategy
from pydantic import ValidationError
from sseclient import SSEClient

from projectdavid.clients.base_client import BaseAPIClient

ent_validator = ValidationInterface()
logging_utility = UtilsInterface.LoggingUtility()
LOG = logging_utility


class RunsClient(BaseAPIClient):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        write_timeout: float = 30.0,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        logging_utility.info("RunsClient ready at: %s", self.base_url)

    def create_run(
        self,
        assistant_id: str,
        thread_id: str,
        instructions: str = "",
        meta_data: Optional[Dict[str, Any]] = None,
        *,
        model: Optional[str] = None,
        response_format: str = "text",
        tool_choice: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        truncation_strategy: Optional[TruncationStrategy] = None,
    ) -> Any:  # Returns ent_validator.Run
        meta_data = meta_data or {}
        tool_choice = tool_choice or "none"
        model = model or "gpt-4"

        now = int(time.time())

        run_payload = ent_validator.RunCreate(
            id=UtilsInterface.IdentifierService.generate_run_id(),
            user_id=None,
            assistant_id=assistant_id,
            thread_id=thread_id,
            instructions=instructions,
            meta_data=meta_data,
            cancelled_at=None,
            completed_at=None,
            created_at=now,
            expires_at=now + 3600,
            failed_at=None,
            incomplete_details=None,
            last_error=None,
            max_completion_tokens=1000,
            max_prompt_tokens=500,
            model=model,
            object="run",
            parallel_tool_calls=False,
            required_action=None,
            response_format=response_format,
            started_at=None,
            status=ent_validator.RunStatus.pending,
            tool_choice=tool_choice,
            tools=[],
            usage=None,
            temperature=temperature,
            top_p=top_p,
            tool_resources={},
            truncation_strategy=truncation_strategy,
        )

        logging_utility.info(
            "Creating run for assistant_id=%s, thread_id=%s", assistant_id, thread_id
        )

        try:
            payload_dict = run_payload.model_dump(exclude_none=True)
            resp = self.client.post("/v1/runs", json=payload_dict)
            resp.raise_for_status()
            run_out = ent_validator.Run(**resp.json())
            logging_utility.info("Run created successfully: %s", run_out.id)
            return run_out
        except ValidationError as e:
            logging_utility.error("Validation error: %s", e.json())
            raise ValueError(f"Validation error: {e}") from e
        except httpx.HTTPStatusError:
            logging_utility.error("HTTP error during run creation")
            raise
        except Exception as e:
            logging_utility.error("Unexpected error during run creation: %s", str(e))
            raise

    def retrieve_run(self, run_id: str) -> Any:  # Returns RunReadDetailed
        logging_utility.info("Retrieving run with id: %s", run_id)
        try:
            response = self.client.get(f"/v1/runs/{run_id}")
            response.raise_for_status()
            run_data = response.json()
            validated_run = ent_validator.RunReadDetailed(**run_data)
            return validated_run
        except (ValidationError, httpx.HTTPStatusError, Exception):
            raise

    def update_run_status(self, run_id: str, new_status: str) -> Any:
        logging_utility.info(
            "Updating run status for run_id: %s to %s", run_id, new_status
        )
        update_data = {"status": new_status}
        try:
            validated_data = ent_validator.RunStatusUpdate(**update_data)
            response = self.client.put(
                f"/v1/runs/{run_id}/status", json=validated_data.dict()
            )
            response.raise_for_status()
            return ent_validator.Run(**response.json())
        except Exception:
            raise

    def delete_run(self, run_id: str) -> Dict[str, Any]:
        logging_utility.info("Deleting run with id: %s", run_id)
        try:
            response = self.client.delete(f"/v1/runs/{run_id}")
            response.raise_for_status()
            return response.json()
        except Exception:
            raise

    def generate(
        self, run_id: str, model: str, prompt: str, stream: bool = False
    ) -> Dict[str, Any]:
        try:
            run = self.retrieve_run(run_id)
            response = self.client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": stream,
                    "context": run.meta_data.get("context", []),
                    "temperature": run.temperature,
                    "top_p": run.top_p,
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            raise

    def chat(
        self,
        run_id: str,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
    ) -> Dict[str, Any]:
        try:
            run = self.retrieve_run(run_id)
            response = self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "context": run.meta_data.get("context", []),
                    "temperature": run.temperature,
                    "top_p": run.top_p,
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            raise

    def cancel_run(self, run_id: str) -> Any:
        logging_utility.info("Cancelling run with id: %s", run_id)
        try:
            response = self.client.post(f"/v1/runs/{run_id}/cancel")
            response.raise_for_status()
            return ent_validator.Run(**response.json())
        except Exception:
            raise

    def poll_and_execute_action(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tool_executor: Callable[[str, Dict[str, Any]], str],
        actions_client: Any,
        messages_client: Any,
        timeout: float = 60.0,
        interval: float = 1.0,
    ) -> bool:
        if timeout <= 0 or interval <= 0:
            raise ValueError("Timeout and interval must be positive numbers.")
        if not callable(tool_executor):
            raise TypeError("tool_executor must be a callable function.")

        start_time = time.time()
        action_handled_successfully = False
        logging_utility.info(f"[SDK Helper] Monitoring run {run_id} for actions...")

        terminal_run_states = {
            StatusEnum.completed.value,
            StatusEnum.failed.value,
            StatusEnum.cancelled.value,
            StatusEnum.expired.value,
        }
        target_run_state = StatusEnum.pending_action.value

        while (time.time() - start_time) < timeout:
            action_to_handle: Optional[Dict[str, Any]] = None

            try:
                current_run = self.retrieve_run(run_id)
                status_str = (
                    current_run.status.value
                    if hasattr(current_run.status, "value")
                    else str(current_run.status)
                )

                if status_str in terminal_run_states:
                    return False

                if status_str == target_run_state:
                    pending_actions = actions_client.get_pending_actions(run_id=run_id)
                    if pending_actions:
                        action_to_handle = pending_actions[0]

            except Exception as e:
                logging_utility.error(f"[SDK Helper] Error polling run status: {e}")
                return False

            if action_to_handle:
                action_id = action_to_handle.get("action_id")
                tool_name = action_to_handle.get("tool_name")
                tool_call_id = action_to_handle.get("tool_call_id")
                arguments = action_to_handle.get("function_arguments")

                # Validate required fields before calling tool_executor
                if not isinstance(tool_name, str):
                    logging_utility.error(
                        "[SDK Helper] action missing valid tool_name: %r", tool_name
                    )
                    break
                if not isinstance(arguments, dict):
                    arguments = {}

                try:
                    actions_client.update_action(
                        action_id, status=StatusEnum.processing.value
                    )
                    result_content = tool_executor(tool_name, arguments)
                    if not isinstance(result_content, str):
                        result_content = json.dumps(result_content)

                    messages_client.submit_tool_output(
                        thread_id=thread_id,
                        tool_id=action_id,
                        tool_call_id=tool_call_id,
                        content=result_content,
                        role="tool",
                        assistant_id=assistant_id,
                    )
                    action_handled_successfully = True
                    break
                except Exception as tool_exc:
                    error_payload = json.dumps({"error": str(tool_exc)})
                    messages_client.submit_tool_output(
                        thread_id=thread_id,
                        tool_id=action_id,
                        tool_call_id=tool_call_id,
                        content=error_payload,
                        role="tool",
                        assistant_id=assistant_id,
                    )
                    break
            time.sleep(interval)

        return action_handled_successfully

    def execute_pending_action(
        self,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tool_executor: Callable[[str, Dict[str, Any]], str],
        actions_client: Any,
        messages_client: Any,
        streamed_args: Dict[str, Any],
        action_id: str,
        tool_name: str,
        tool_call_id: Optional[str] = None,
    ) -> bool:
        try:
            actions_client.update_action(action_id, status="processing")
            result_content = tool_executor(tool_name, streamed_args)
            if not isinstance(result_content, str):
                result_content = json.dumps(result_content)

            messages_client.submit_tool_output(
                thread_id=thread_id,
                tool_id=action_id,
                tool_call_id=tool_call_id,
                content=result_content,
                role="tool",
                assistant_id=assistant_id,
            )
            actions_client.update_action(action_id, status=StatusEnum.completed.value)
            return True
        except Exception as e:
            instructional_hint = f"Error for '{tool_name}': {str(e)}"
            messages_client.submit_tool_output(
                thread_id=thread_id,
                tool_id=action_id,
                tool_call_id=tool_call_id,
                content=json.dumps({"error": instructional_hint}),
                role="tool",
                assistant_id=assistant_id,
            )
            actions_client.update_action(action_id, status=StatusEnum.failed.value)
            return True

    def execute_delegated_action(
        self,
        tool_name: str,
        args: Dict[str, Any],
        action_id: str,
        thread_id: str,
        assistant_id: str,
        tool_executor: Callable[[str, Dict[str, Any]], str],
        actions_client: Any,
        messages_client: Any,
        tool_call_id: Optional[str] = None,
    ) -> bool:
        try:
            actions_client.update_action(action_id, status="processing")
            result_content = tool_executor(tool_name, args)
            if not isinstance(result_content, str):
                result_content = json.dumps(result_content)

            messages_client.submit_tool_output(
                thread_id=thread_id,
                tool_id=action_id,
                tool_call_id=tool_call_id,
                content=result_content,
                role="tool",
                assistant_id=assistant_id,
            )
            actions_client.update_action(action_id, status=StatusEnum.completed.value)
            return True
        except Exception as e:
            messages_client.submit_tool_output(
                thread_id=thread_id,
                tool_id=action_id,
                tool_call_id=tool_call_id,
                content=json.dumps({"error": str(e)}),
                role="tool",
                assistant_id=assistant_id,
            )
            actions_client.update_action(action_id, status=StatusEnum.failed.value)
            return True

    def watch_run_events(
        self,
        run_id: str,
        tool_executor: Callable[[str, dict], str],
        actions_client: Any,
        messages_client: Any,
        assistant_id: str,
        thread_id: str,
    ) -> None:
        url = f"{self.base_url}/v1/runs/{run_id}/events"
        headers = self.client.headers

        def _listen_and_handle():
            # BANDIT FIX: Timeout added here
            resp = requests.get(url, headers=headers, stream=True, timeout=(10, 60))
            resp.raise_for_status()
            client = SSEClient(resp.iter_lines())
            for event in client.events():
                if event.event == "action_required":
                    action = json.loads(event.data)
                    result = tool_executor(
                        action.get("tool_name"), action.get("function_arguments", {})
                    )
                    if not isinstance(result, str):
                        result = json.dumps(result)
                    messages_client.submit_tool_output(
                        thread_id=thread_id,
                        tool_id=action.get("action_id"),
                        content=result,
                        role="tool",
                        assistant_id=assistant_id,
                    )
                    break

        t = threading.Thread(target=_listen_and_handle, daemon=True)
        t.start()
        t.join()

    def list_runs(self, thread_id: str, limit: int = 20, order: str = "asc") -> Any:
        params: Dict[str, Union[str, int]] = {
            "limit": limit,
            "order": order if order in ("asc", "desc") else "asc",
        }
        resp = self.client.get(f"/v1/threads/{thread_id}/runs", params=params)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and "data" in payload:
            return ent_validator.RunListResponse(**payload)
        runs = [ent_validator.Run(**item) for item in payload]
        return ent_validator.RunListResponse(object="list", data=runs, has_more=False)

    def list_all_runs(self, limit: int = 20, order: str = "asc") -> Any:
        params: Dict[str, Union[str, int]] = {
            "limit": limit,
            "order": order if order in ("asc", "desc") else "asc",
        }
        resp = self.client.get("/v1/runs", params=params)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and "data" in payload:
            return ent_validator.RunListResponse(**payload)
        runs = [ent_validator.Run(**item) for item in payload]
        return ent_validator.RunListResponse(object="list", data=runs, has_more=False)

    def update_run_fields(self, run_id: str, **kwargs) -> Any:
        logging_utility.info("Patching run %s fields: %s", run_id, list(kwargs.keys()))
        try:
            resp = self.client.patch(f"/v1/runs/{run_id}/fields", json=kwargs)
            resp.raise_for_status()
            return ent_validator.Run(**resp.json())
        except Exception:
            raise
