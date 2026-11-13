"""Scaling observability: samples/s, step time and cross-rank load imbalance."""

from __future__ import annotations

import time

from avatar.train.callbacks.base import TrainerCallback
from avatar.train.state import CallbackContext


class ThroughputCallback(TrainerCallback):
    """Report how fast the run goes and how evenly the work is spread.

    All of the reductions happen once per epoch, so the numbers cost nothing per
    step. ``step_time_imbalance_pct`` is the signal to watch when adding ranks:
    it is the gap between the slowest and the fastest rank's compute time, as a
    percentage of the slowest, and it is what turns into idle GPU at the
    gradient all-reduce.
    """

    def __init__(self, log_every_epoch: bool = True):
        self.log_every_epoch = log_every_epoch
        self._epoch_start = 0.0
        self._compute_sec = 0.0
        self._batch_started = 0.0

    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        self._epoch_start = time.perf_counter()
        self._compute_sec = 0.0

    def on_batch_begin(self, ctx: CallbackContext) -> None:
        self._batch_started = time.perf_counter()

    def on_step_end(self, ctx: CallbackContext) -> None:
        self._compute_sec += time.perf_counter() - self._batch_started

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        if not (self.log_every_epoch and ctx.log is not None):
            return
        env = ctx.env
        wall_sec = time.perf_counter() - self._epoch_start
        wall_max = env.all_reduce_max(wall_sec)
        global_samples = env.all_reduce_sum(float(ctx.state.epoch_samples))
        compute_max = env.all_reduce_max(self._compute_sec)
        compute_min = env.all_reduce_min(self._compute_sec)
        steps = max(ctx.state.epoch_step, 1)

        ctx.log(
            {
                "throughput/samples_per_sec": (
                    global_samples / wall_max if wall_max else 0.0
                ),
                "throughput/samples_per_sec_per_rank": (
                    global_samples / wall_max / env.world_size if wall_max else 0.0
                ),
                "throughput/step_time_sec": wall_max / steps,
                "throughput/step_time_imbalance_pct": (
                    100.0 * (compute_max - compute_min) / compute_max
                    if compute_max
                    else 0.0
                ),
                "throughput/non_compute_sec": max(wall_max - compute_max, 0.0),
            },
            step=ctx.state.epoch,
        )
