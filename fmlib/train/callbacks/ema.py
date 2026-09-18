"""Exponential moving average / SWA weight tracking."""

from __future__ import annotations

import torch

from fmlib.train.callbacks.base import TrainerCallback
from fmlib.train.dist import unwrap_model
from fmlib.train.state import CallbackContext


class EMACallback(TrainerCallback):
    """Maintain an :class:`~torch.optim.swa_utils.AveragedModel` alongside training.

    Once the averaged copy exists it becomes the model used for evaluation and
    for checkpointing, published through ``ctx.extra["eval_model"]``.

    Args:
        swa_model: The averaged model, already built over the training module.
        update_every: Optimizer steps between averaged-weight updates.
        min_epoch: First epoch at which averaging starts.

    Note:
        ``update_every`` is a literal step count. The old inline implementation
        divided it by the process count to compensate for accelerate's step
        semantics; with an explicit loop the budget means what it says.
    """

    def __init__(
        self,
        swa_model: torch.optim.swa_utils.AveragedModel,
        update_every: int = 128,
        min_epoch: int = 0,
    ):
        if update_every < 1:
            raise ValueError("update_every must be at least 1")
        self.swa_model = swa_model
        self.update_every = update_every
        self.min_epoch = min_epoch
        self._started = False

    def _publish(self, ctx: CallbackContext) -> None:
        if self._started:
            ctx.extra["eval_model"] = self.swa_model

    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        self._publish(ctx)

    def on_step_end(self, ctx: CallbackContext) -> None:
        if ctx.state.epoch < self.min_epoch:
            return
        if ctx.state.global_step % self.update_every != 0:
            return
        model = unwrap_model(ctx.model)
        was_training = model.training
        with torch.inference_mode():
            model.eval()
            self.swa_model.update_parameters(model)
        if was_training:
            model.train()
        self._started = True
        self._publish(ctx)

    def on_evaluate(self, ctx: CallbackContext) -> None:
        self._publish(ctx)
