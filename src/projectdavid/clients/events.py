# src/projectdavid/clients/events.py
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union


@dataclass
class StreamEvent:
    """Base class for all stream events."""

    run_id: str


@dataclass
class ContentEvent(StreamEvent):
    """Represents a text delta from the assistant."""

    content: str

    def __str__(self):
        return self.content


class ToolCallRequestEvent(StreamEvent):
    """
    Represents a fully accumulated tool call request.
    Contains the parsed arguments and a helper method to execute it.
    """

    def __init__(
        self,
        run_id: str,
        tool_name: str,
        args: Dict[str, Any],
        # Context required for execution
        thread_id: str,
        assistant_id: str,
        _runs_client: Any,
        _actions_client: Any,
        _messages_client: Any,
    ):
        super().__init__(run_id)
        self.tool_name = tool_name
        self.args = args

        # internal references for execution
        self._thread_id = thread_id
        self._assistant_id = assistant_id
        self._runs_client = _runs_client
        self._actions_client = _actions_client
        self._messages_client = _messages_client

    def execute(self, tool_executor: Callable[[str, Dict[str, Any]], str]) -> bool:
        """
        Executes the provided local function using the accumulated arguments
        and submits the result to the API.
        """
        return self._runs_client.execute_pending_action(
            run_id=self.run_id,
            thread_id=self._thread_id,
            assistant_id=self._assistant_id,
            tool_executor=tool_executor,
            actions_client=self._actions_client,
            messages_client=self._messages_client,
            streamed_args=self.args,  # Pass the pre-parsed args!
        )

    def __repr__(self):
        return f"<ToolCallRequestEvent tool='{self.tool_name}' args={self.args}>"


@dataclass
class StatusEvent(StreamEvent):
    """Represents a status change (e.g., complete, failed)."""

    status: str
