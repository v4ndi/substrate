"""Early stopping as a callback around the existing :class:`EarlyStopping`."""

from __future__ import annotations

from fmlib.train.callbacks.base import TrainerCallback
from fmlib.train.early_stopping import EarlyStopping
from fmlib.train.state import CallbackContext


class EarlyStoppingCallback(TrainerCallback):
    """Stop when the monitored metric plateaus, and veto non-improving saves.

    The decision is made on rank 0 — it is the only rank holding the scores —
    and the loop all-reduces ``control.should_training_stop`` afterwards so the
    other ranks leave the loop at the same iteration.
    """

    def __init__(self, early_stopping: EarlyStopping):
        self.early_stopping = early_stopping

    def on_evaluate(self, ctx: CallbackContext) -> None:
        if not ctx.env.is_main or not ctx.metrics:
            return
        self.early_stopping(ctx.metrics)
        ctx.state.best_metric = self.early_stopping.best_score
        # counter == 0 means this evaluation set a new best.
        ctx.control.should_save = self.early_stopping.counter == 0
        ctx.control.should_training_stop = (
            ctx.control.should_training_stop or self.early_stopping.early_stop
        )
