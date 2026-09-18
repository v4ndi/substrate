"""tqdm progress bar, driven by callback events rather than by wrapping the loader."""

from __future__ import annotations

import torch
from tqdm import tqdm

from fmlib.train.callbacks.base import TrainerCallback
from fmlib.train.state import CallbackContext


class ProgressBarCallback(TrainerCallback):
    """Draw one bar per epoch on the local main process.

    The bar is advanced from ``on_batch_begin`` instead of wrapping the
    dataloader, so the loop keeps iterating a plain iterator and the bar stays
    an ordinary, removable callback.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._bar: tqdm | None = None

    def _total(self, ctx: CallbackContext) -> int | None:
        loader = ctx.extra.get("train_dataloader")
        try:
            return len(loader)
        except TypeError:
            return None

    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        if not (self.enabled and ctx.env.is_local_main):
            return
        self._close()
        epoch, total_epochs = ctx.state.epoch + 1, ctx.state.num_epochs
        self._bar = tqdm(
            total=self._total(ctx),
            desc=f"Training; epoch={epoch}/{total_epochs}",
        )

    def on_batch_begin(self, ctx: CallbackContext) -> None:
        if self._bar is not None:
            self._bar.update(1)

    def on_forward_end(self, ctx: CallbackContext) -> None:
        if self._bar is None or ctx.loss is None:
            return
        loss = ctx.loss
        loss = float(loss.detach() if torch.is_tensor(loss) else loss)
        self._bar.set_postfix(loss=f"{loss:.4f}", lr=f"{ctx.state.learning_rate:.3g}")

    def _close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        self._close()

    def on_train_end(self, ctx: CallbackContext) -> None:
        self._close()
