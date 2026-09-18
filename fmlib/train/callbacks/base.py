"""The callback seam: what the loop announces, and how the announcements fan out."""

from __future__ import annotations

from collections.abc import Iterable

from fmlib.train.state import CallbackContext

EVENTS = (
    "on_train_begin",
    "on_epoch_begin",
    "on_batch_begin",
    "on_forward_end",
    "on_backward_end",
    "on_optimizer_step",
    "on_step_end",
    "on_evaluate",
    "on_save",
    "on_epoch_end",
    "on_log",
    "on_train_end",
)


class TrainerCallback:
    """Observe and steer the loop; never run the math.

    Every hook receives the same :class:`~fmlib.train.state.CallbackContext`
    and returns nothing. To influence the loop, mutate ``ctx.control``.

    Hooks fire on **every** rank. A callback that should only act on rank 0
    guards on ``ctx.env.is_main`` itself — that way callbacks which genuinely
    need a collective (throughput, early stopping) can still run everywhere.
    """

    def on_train_begin(self, ctx: CallbackContext) -> None: ...

    def on_epoch_begin(self, ctx: CallbackContext) -> None: ...

    def on_batch_begin(self, ctx: CallbackContext) -> None:
        """After ``move_to_device``, before the forward pass."""

    def on_forward_end(self, ctx: CallbackContext) -> None:
        """``ctx.output`` and ``ctx.loss`` are available."""

    def on_backward_end(self, ctx: CallbackContext) -> None:
        """Gradients are populated; the optimizer has not stepped."""

    def on_optimizer_step(self, ctx: CallbackContext) -> None:
        """Accumulation boundary only. ``ctx.grad_norm`` is set when clipping."""

    def on_step_end(self, ctx: CallbackContext) -> None:
        """One full (accumulated) optimizer step has completed."""

    def on_evaluate(self, ctx: CallbackContext) -> None:
        """``ctx.metrics`` holds the validation scores (rank 0) or is empty."""

    def on_save(self, ctx: CallbackContext) -> None: ...

    def on_epoch_end(self, ctx: CallbackContext) -> None: ...

    def on_log(self, ctx: CallbackContext) -> None:
        """``ctx.logs`` holds the values to record, ``ctx.state.global_step`` the step."""

    def on_train_end(self, ctx: CallbackContext) -> None: ...


class CallbackHandler:
    """Dispatch one event to an ordered list of callbacks."""

    def __init__(self, callbacks: Iterable[TrainerCallback] | None = None):
        self.callbacks: list[TrainerCallback] = list(callbacks or [])

    def add(self, callback: TrainerCallback | None) -> None:
        if callback is not None:
            self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)

    def fire(self, event: str, ctx: CallbackContext) -> None:
        if event not in EVENTS:
            raise ValueError(f"Unknown callback event: {event!r}")
        for callback in self.callbacks:
            getattr(callback, event)(ctx)
