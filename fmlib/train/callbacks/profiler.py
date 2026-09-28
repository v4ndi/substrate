"""``torch.profiler`` driving, on rank 0 only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fmlib.train.callbacks.base import TrainerCallback
from fmlib.train.state import CallbackContext


class ProfilerCallback(TrainerCallback):
    """Start, step and stop a profiler around the run.

    Args:
        profiler: A configured ``torch.profiler.profile`` object.
    """

    def __init__(self, profiler: Any, trace_paths: list[Path] | None = None):
        self.profiler = profiler
        self.trace_paths = trace_paths if trace_paths is not None else []
        self._running = False

    def on_train_begin(self, ctx: CallbackContext) -> None:
        if ctx.env.is_main and self.profiler is not None:
            self.profiler.start()
            self._running = True

    def on_step_end(self, ctx: CallbackContext) -> None:
        if self._running:
            self.profiler.step()

    def on_train_end(self, ctx: CallbackContext) -> None:
        self._stop(ctx)

    def _stop(self, ctx: CallbackContext) -> None:
        if self._running:
            self.profiler.stop()
            self._running = False
        if self.trace_paths:
            ctx.extra["profiler_traces"] = tuple(self.trace_paths)
