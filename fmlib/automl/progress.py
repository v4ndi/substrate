"""Run-scoped progress logging shared by local and remote AutoML actions."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_fallback_logger = logging.getLogger(__name__)
_active_logger: ContextVar[logging.Logger | None] = ContextVar(
    "fmlib_automl_progress_logger", default=None
)


def log_progress(message: str, *args: object) -> None:
    """Write an INFO progress event to the current run log.

    Args:
        message: Logging format string.
        *args: Values interpolated into ``message`` by the logging backend.

    Returns:
        ``None``.
    """
    (_active_logger.get() or _fallback_logger).info(message, *args)


@contextmanager
def use_progress_logger(logger: logging.Logger) -> Iterator[None]:
    """Route progress events to ``logger`` for the duration of one action.

    Args:
        logger: Run-scoped logger receiving progress events.

    Yields:
        Control while the supplied logger is bound to the current context.
    """
    token = _active_logger.set(logger)
    try:
        yield
    finally:
        _active_logger.reset(token)
