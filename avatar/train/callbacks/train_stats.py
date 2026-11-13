"""Per-epoch means of the numbers the loop produces: loss, learning rate, grad norm."""

from __future__ import annotations

import torch

from avatar.train.callbacks.base import TrainerCallback
from avatar.train.state import CallbackContext


class TrainStatsCallback(TrainerCallback):
    """Accumulate loss / LR / grad-norm locally, reduce and log once per epoch.

    Accumulating on the local rank and reducing once at the epoch boundary keeps
    the step loop free of collectives; the old inline version gathered a list of
    per-step tensors and reduced them all at the end, which is the same
    arithmetic with more memory.
    """

    def __init__(self, log_grad_norm: bool = True):
        self.log_grad_norm = log_grad_norm
        self._loss_sum = 0.0
        self._loss_count = 0
        self._lr_sum = 0.0
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0

    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        self._loss_sum = 0.0
        self._loss_count = 0
        self._lr_sum = 0.0
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0

    def on_forward_end(self, ctx: CallbackContext) -> None:
        if ctx.loss is None:
            return
        loss = ctx.loss
        self._loss_sum += float(loss.detach() if torch.is_tensor(loss) else loss)
        self._loss_count += 1

    def on_optimizer_step(self, ctx: CallbackContext) -> None:
        self._lr_sum += float(ctx.state.learning_rate)
        if ctx.grad_norm is not None:
            value = ctx.grad_norm
            self._grad_norm_sum += float(
                value.detach() if torch.is_tensor(value) else value
            )
            self._grad_norm_count += 1

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        env = ctx.env
        loss_sum = env.all_reduce_sum(self._loss_sum)
        loss_count = env.all_reduce_sum(float(self._loss_count))
        logs: dict[str, float] = {}
        if loss_count:
            logs["train_loss"] = loss_sum / loss_count
        if ctx.state.epoch_step:
            logs["mean_epoch_lr"] = self._lr_sum / ctx.state.epoch_step
        if self.log_grad_norm:
            norm_sum = env.all_reduce_sum(self._grad_norm_sum)
            norm_count = env.all_reduce_sum(float(self._grad_norm_count))
            if norm_count:
                logs["grad_norm"] = norm_sum / norm_count
        if logs and ctx.log is not None:
            ctx.log(logs, step=ctx.state.epoch)
