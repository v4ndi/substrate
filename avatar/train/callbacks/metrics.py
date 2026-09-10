"""Training-set metric accumulation."""

from __future__ import annotations

from avatar.metrics import BaseMetric
from avatar.train.callbacks.base import TrainerCallback
from avatar.train.evaluate import dataloader_is_sharded
from avatar.train.state import CallbackContext
from avatar.train.utils import (
    metric_field_selection,
    narrow_for_metrics,
    prefix_metrics,
    wrap_metrics,
)


class TrainMetricsCallback(TrainerCallback):
    """Feed ``(batch, output)`` pairs to the training metrics and log per epoch.

    Gathering happens only when the dataset shards; with a replicated dataset
    every rank holds the same rows and gathering would count each sample
    ``world_size`` times.
    """

    def __init__(
        self,
        metrics: BaseMetric | list[BaseMetric] | None,
        prefix: str = "train_",
    ):
        self.metrics = wrap_metrics(metrics)
        self.prefix = prefix
        self._gather = False
        # Resolved once: the metric list does not change during a run, and this
        # runs on every forward.
        self._selection = metric_field_selection(self.metrics)

    def on_train_begin(self, ctx: CallbackContext) -> None:
        loader = ctx.extra.get("train_dataloader")
        self._gather = ctx.env.distributed and (
            loader is None or dataloader_is_sharded(loader)
        )

    def on_forward_end(self, ctx: CallbackContext) -> None:
        if self.metrics is None:
            return
        payload = narrow_for_metrics(ctx.batch, ctx.output, self._selection)
        pairs = ctx.env.gather_objects(payload) if self._gather else [payload]
        if not ctx.env.is_main:
            return
        for metric in self.metrics:
            for inputs, outputs in pairs:
                metric.update(inputs=inputs, outputs=outputs)

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        if self.metrics is None or not ctx.env.is_main:
            return
        scores: dict[str, float] = {}
        for metric in self.metrics:
            scores.update(metric.compute())
            metric.reset()
        if scores and ctx.log is not None:
            ctx.log(prefix_metrics(scores, self.prefix), step=ctx.state.epoch)
