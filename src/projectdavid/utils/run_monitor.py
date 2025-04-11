import threading
import time
from typing import Any, Callable, Dict, List, Optional


class HttpRunMonitor:
    """
    Monitors the lifecycle of a run using the HTTP-based SDK clients (RunsClient + ActionsClient).
    Triggers callbacks on lifecycle transitions.
    """

    TERMINAL_STATUSES = ({"completed", "failed", "cancelled", "expired"},)
    ACTION_REQUIRED_STATUS = ("pending_action",)
    PLATFORM_TOOLS = [
        "code_interpreter",
        "web_search",
        "vector_store_search",
        "computer",
    ]

    def __init__(
        self,
        run_id: str,
        runs_client: Any,
        actions_client: Any,
        on_status_change: Callable[
            [str, str, Optional[str], Optional[Dict[str, Any]]], None
        ],
        on_complete: Callable[[str, str, Optional[Dict[str, Any]]], None],
        on_error: Callable[[str, str], None],
        on_action_required: Optional[
            Callable[[str, Dict[str, Any], List[Dict[str, Any]]], None]
        ] = None,
        check_interval: int = 5,
        initial_delay: int = 1,
    ):

        self.run_id = run_id
        self.runs_client = runs_client
        self.actions_client = actions_client
        self.on_status_change = on_status_change
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_action_required = on_action_required
        self.check_interval = check_interval
        self.initial_delay = initial_delay

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_status: Optional[str] = None

    def start(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()

    def is_active(self) -> bool:
        """Return True if the monitor thread is active and running."""
        return (
            self._monitor_thread is not None
            and self._monitor_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def _monitor_loop(self):
        time.sleep(self.initial_delay)

        while not self._stop_event.is_set():
            try:
                run = self.runs_client.retrieve_run(self.run_id)
                current_status = run.status

                if self._last_status != current_status:
                    self.on_status_change(
                        self.run_id, current_status, self._last_status, run.dict()
                    )
                    self._last_status = current_status

                if (
                    current_status == self.ACTION_REQUIRED_STATUS
                    and self.on_action_required
                ):
                    try:
                        pending_actions = self.actions_client.get_pending_actions(
                            self.run_id
                        )
                        self.on_action_required(
                            self.run_id, run.dict(), pending_actions
                        )
                    except Exception as action_err:
                        self.on_error(
                            self.run_id, f"Error fetching actions: {action_err}"
                        )

                if current_status in self.TERMINAL_STATUSES:
                    self.on_complete(self.run_id, current_status, run.dict())
                    break

            except Exception as e:
                self.on_error(self.run_id, f"Run monitoring error: {e}")
                break

            time.sleep(self.check_interval)
