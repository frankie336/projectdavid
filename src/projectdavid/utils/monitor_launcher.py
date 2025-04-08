import threading

from projectdavid_common import UtilsInterface

logging_utility = UtilsInterface.LoggingUtility()


class MonitorLauncher:
    def __init__(
        self,
        client,
        actions_client,
        run_id,
        *,
        on_status_change=None,
        on_complete=None,
        on_error=None,
        on_action_required=None,
        events=None,
    ):
        self.client = client
        self.actions_client = actions_client
        self.run_id = run_id
        self.events = events

        # --- Default handlers ---
        def default_status_change(run_id, new_status, old_status, run_data):
            logging_utility.info(
                f"[MONITOR STATUS] {run_id}: {old_status} → {new_status}"
            )

        def default_completion(run_id, final_status, run_data):
            logging_utility.info(
                f"[MONITOR COMPLETE] {run_id} ended with status: {final_status}"
            )

        def default_error(run_id, error_msg):
            logging_utility.error(f"[MONITOR ERROR] {run_id}: {error_msg}")

        def default_action_required(run_id, run_data, pending_actions):
            try:
                logging_utility.info(
                    f"[ACTION_REQUIRED] run {run_id} has {len(pending_actions)} pending action(s)"
                )
                for action in pending_actions:
                    action_id = action.get("id")
                    tool_name = action.get("tool_name")
                    args = action.get("function_args", {})
                    logging_utility.info(
                        f"[ACTION] ID: {action_id}, Tool: {tool_name}, Args: {args}"
                    )
            except Exception as e:
                logging_utility.error(
                    f"[MonitorLauncher] Error processing actions: {e}"
                )

        # --- Use defaults unless overridden ---
        self.monitor = self.events.HttpRunMonitor(
            run_id=run_id,
            runs_client=client.runs,
            actions_client=actions_client,
            on_status_change=on_status_change or default_status_change,
            on_complete=on_complete or default_completion,
            on_error=on_error or default_error,
            on_action_required=on_action_required or default_action_required,
        )

    def _monitor_loop(self):
        try:
            logging_utility.info(
                f"[MonitorLauncher] Starting monitor for run {self.run_id}"
            )
            self.monitor.start()
        except Exception as e:
            logging_utility.error(f"[MonitorLauncher] Monitoring thread failed: {e}")

    def start(self):
        t = threading.Thread(target=self._monitor_loop)
        t.daemon = True
        t.start()
        logging_utility.info(f"[MonitorLauncher] Launched thread for run {self.run_id}")
