# projectdavid/__init__.py
from projectdavid._version import MIN_COMPATIBLE_API_VERSION, SDK_VERSION

from .entity import Entity

# Expose events so users can import them cleanly
from .events import (
    CodeExecutionGeneratedFileEvent,
    CodeExecutionOutputEvent,
    ComputerExecutionOutputEvent,
    ContentEvent,
    HotCodeEvent,
    ReasoningEvent,
    StatusEvent,
    StreamEvent,
    ToolCallRequestEvent,
)

__all__ = [
    "Entity",
    "SDK_VERSION",
    "MIN_COMPATIBLE_API_VERSION",
    "StreamEvent",
    "ContentEvent",
    "ReasoningEvent",
    "HotCodeEvent",
    "CodeExecutionOutputEvent",
    "CodeExecutionGeneratedFileEvent",
    "ComputerExecutionOutputEvent",
    "ToolCallRequestEvent",
    "StatusEvent",
]
