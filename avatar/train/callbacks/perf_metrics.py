"""System, epoch and shard performance metrics.

Ports the performance block that used to be inlined in ``avatar/train.py``.
All reductions use the scalar collectives in
:mod:`avatar.utils.performance_metrics`, which are already plain
``torch.distributed``.
"""

from __future__ import annotations

import time

from avatar.train.callbacks.base import TrainerCallback
from avatar.train.state import CallbackContext
from avatar.utils.performance_metrics import (
    SystemMetricsCollector,
    reduce_epoch_performance,
    reduce_max_duration,
    reduce_system_snapshot,
)


class PerfMetricsCallback(TrainerCallback):
    """Sample host/GPU counters and reduce them once per epoch.

    Args:
        sampling_interval_sec: Minimum gap between host/GPU samples.
        startup_begin_time: ``time.perf_counter()`` taken at process start, so
            the cost of everything before the first epoch can be reported.
        shard_metrics: Whether to also collect the dataset's per-epoch shard
            counters (row/IO read amplification, filter time).
    """

    def __init__(
        self,
        sampling_interval_sec: float = 2.0,
        startup_begin_time: float | None = None,
        shard_metrics: bool = True,
    ):
        self.sampling_interval_sec = sampling_interval_sec
        self.startup_begin_time = startup_begin_time
        self.shard_metrics = shard_metrics
        self._collector: SystemMetricsCollector | None = None
        self._dataset = None
        self._shard_enabled = False
        self._epoch_start = 0.0
        self._previous_epoch_end: float | None = None
        self._first_batch_sec: float | None = None

    # -- setup ---------------------------------------------------------------

    def on_train_begin(self, ctx: CallbackContext) -> None:
        env = ctx.env
        loader = ctx.extra.get("train_dataloader")
        self._dataset = getattr(loader, "dataset", None)
        self._shard_enabled = bool(
            self.shard_metrics and getattr(self._dataset, "shard", False)
        )
        self._collector = SystemMetricsCollector(
            is_local_main_process=env.is_local_main,
            device=env.device,
            sampling_interval_sec=self.sampling_interval_sec,
        )
        if self._shard_enabled and hasattr(self._dataset, "configure_epoch_metrics"):
            self._dataset.configure_epoch_metrics(getattr(loader, "num_workers", 0))
        self._log_scan_metrics(ctx)
        self._log_startup(ctx)

    def _log_scan_metrics(self, ctx: CallbackContext) -> None:
        """One-off metrics describing the shard scan that already happened."""
        if not self._shard_enabled:
            return
        env = ctx.env
        dataset = self._dataset
        scan_sec = reduce_max_duration(
            float(getattr(dataset, "scan_duration_sec", 0.0)), env.device
        )
        total_raw_rows = int(getattr(dataset, "total_length", 0))
        file_counts = getattr(dataset, "file_counts", None)
        total_valid_rows = int(file_counts.sum()) if file_counts is not None else 0
        dropped_tail_rows = (
            total_valid_rows % env.world_size
            if getattr(dataset, "drop_tail", True)
            else 0
        )
        if not env.is_main or ctx.log is None:
            return
        metrics = {
            "shard/scan_sec": scan_sec,
            "shard/scan_rows_per_sec": total_raw_rows / scan_sec if scan_sec else 0.0,
            "shard/total_raw_rows": total_raw_rows,
            "shard/total_valid_rows": total_valid_rows,
            "shard/dropped_tail_rows": dropped_tail_rows,
            "shard/parquet_file_count": len(getattr(dataset, "files", [])),
        }
        if getattr(dataset, "uses_filter_cache", False):
            metrics["shard/filter_cache_hit"] = float(
                getattr(dataset, "filter_cache_hit", False)
            )
            metrics["shard/scan_rank_count"] = float(
                getattr(dataset, "scan_rank_count", 0)
            )
        ctx.log(metrics, step=0)

    def _log_startup(self, ctx: CallbackContext) -> None:
        if self.startup_begin_time is None:
            return
        env = ctx.env
        env.barrier()
        startup_max = reduce_max_duration(
            time.perf_counter() - self.startup_begin_time, env.device
        )
        if env.is_main and ctx.log is not None:
            ctx.log(
                {"startup/before_first_epoch_sec": startup_max},
                step=ctx.state.epoch,
            )

    # -- per-epoch -----------------------------------------------------------

    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        env = ctx.env
        env.barrier()
        if self._previous_epoch_end is not None and ctx.log is not None:
            between = reduce_max_duration(
                time.perf_counter() - self._previous_epoch_end, env.device
            )
            if env.is_main:
                ctx.log(
                    {"train_epoch/between_epochs_sec": between},
                    step=ctx.state.epoch - 1,
                )
        self._epoch_start = time.perf_counter()
        self._first_batch_sec = None
        self._collector.reset_epoch()
        if self._shard_enabled and hasattr(self._dataset, "reset_epoch_metrics"):
            self._dataset.reset_epoch_metrics()

    def on_batch_begin(self, ctx: CallbackContext) -> None:
        if self._first_batch_sec is None:
            self._first_batch_sec = time.perf_counter() - self._epoch_start
        self._collector.maybe_sample()

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        env = ctx.env
        epoch_wall_sec = time.perf_counter() - self._epoch_start
        self._previous_epoch_end = time.perf_counter()
        shard_stats = (
            self._dataset.get_epoch_metrics()
            if self._shard_enabled and hasattr(self._dataset, "get_epoch_metrics")
            else None
        )
        epoch_performance = reduce_epoch_performance(
            device=env.device,
            epoch_wall_sec=epoch_wall_sec,
            local_samples=ctx.state.epoch_samples,
            local_batches=ctx.state.epoch_batches,
            shard_stats=shard_stats,
            total_raw_rows=int(getattr(self._dataset, "total_length", 0)),
            total_parquet_bytes=int(getattr(self._dataset, "total_parquet_bytes", 0)),
        )
        system_performance = reduce_system_snapshot(
            self._collector.snapshot(), env.device
        )
        if ctx.extra.get("first_epoch", False):
            epoch_performance["startup/first_batch_sec"] = reduce_max_duration(
                self._first_batch_sec or 0.0, env.device
            )
        if env.is_main and ctx.log is not None:
            ctx.log(
                {**epoch_performance, **system_performance},
                step=ctx.state.epoch,
            )
