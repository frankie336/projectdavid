# decorators.py

import functools
import warnings
from typing import Optional


class ExperimentalWarning(UserWarning):
    """
    API is experimental, may be incomplete (and possibly not fully functional),
    and may change or be removed without notice.
    """


class InternalWarning(UserWarning):
    """
    API is internal/admin-only. Intended for framework authors and
    superusers—will likely change or go away without notice.
    """


def experimental(func):
    """
    Decorator to mark methods as experimental.
    Emits an ExperimentalWarning on first call.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is experimental, may be incomplete, and may change in a future release",
            ExperimentalWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper.__experimental__ = True
    return wrapper


def internal(func):
    """
    Decorator to mark a method as internal/admin-only.
    Raises PermissionError if self.is_admin is False or missing.
    Emits an InternalWarning on use.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 1️⃣ Enforce admin
        if not getattr(self, "is_admin", False):
            raise PermissionError(
                f"{func.__name__} is for admin use only; your client.is_admin is False."
            )
        # 2️⃣ Warn about internal usage
        warnings.warn(
            f"{func.__name__} is internal/admin-only and may change without notice",
            InternalWarning,
            stacklevel=2,
        )
        return func(self, *args, **kwargs)

    wrapper.__internal__ = True
    return wrapper
