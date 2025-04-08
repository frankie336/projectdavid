from projectdavid.utils.monitor_launcher import MonitorLauncher
from projectdavid.utils.run_monitor import HttpRunMonitor


class EventsInterface:
    """
    Exposes core event monitoring utilities for tracking assistant run lifecycle events.

    This interface includes:

    - `HttpRunMonitor`: Low-level polling monitor for observing status changes and triggering callbacks.
    - `MonitorLauncher`: Threaded utility that simplifies asynchronous monitoring with default logging callbacks.

    These can be used to handle events such as `status_change`,
    `action_required`, `complete`, and `error` during the execution of a run.
    """

    HttpRunMonitor = HttpRunMonitor
    MonitorLauncher = MonitorLauncher
