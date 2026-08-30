"""Checkpoint writing and rotation."""

from __future__ import annotations

from avatar.train.callbacks.base import TrainerCallback
from avatar.train.checkpoint import (
    checkpoint_path,
    model_path,
    rotate_checkpoints,
    save_checkpoint,
    save_model,
)
from avatar.train.state import CallbackContext


class CheckpointCallback(TrainerCallback):
    """Write the full training state whenever the loop asks for a save.

    The loop sets ``control.should_save`` before firing ``on_evaluate`` and
    lets callbacks veto it — :class:`~avatar.train.callbacks.early_stopping.EarlyStoppingCallback`
    clears the flag when the monitored metric did not improve, which is how the
    old "only checkpoint the best" behaviour is preserved. Ordering therefore
    matters: early stopping must run before this callback.

    Args:
        directory: Root directory; each save lands in ``<directory>/<step>/``.
        max_checkpoints: Keep only the newest N step directories. None keeps all.
        save_weights: Also write a bare ``model.bin`` next to the full state.
    """

    def __init__(
        self,
        directory: str,
        max_checkpoints: int | None = None,
        save_weights: bool = True,
    ):
        self.directory = directory
        self.max_checkpoints = max_checkpoints
        self.save_weights = save_weights
        self.last_path: str | None = None

    def on_save(self, ctx: CallbackContext) -> None:
        if not ctx.env.is_main:
            return
        step = int(ctx.state.global_step)
        model = ctx.extra.get("eval_model") or ctx.model
        self.last_path = save_checkpoint(
            checkpoint_path(self.directory, step),
            model=model,
            optimizer=ctx.optimizer,
            scheduler=ctx.scheduler,
            scaler=ctx.extra.get("scaler"),
            state=ctx.state,
        )
        if self.save_weights:
            save_model(model, model_path(self.directory, step))
        rotate_checkpoints(self.directory, self.max_checkpoints)
