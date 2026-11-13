"""Low-overhead system and distributed training performance metrics."""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

try:
    import psutil
except ImportError:  # pragma: no cover - depends on deployment extras
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover - depends on deployment extras
    pynvml = None


def infer_batch_size(batch: Any) -> int | None:
    """Find a batch dimension without depending on a specific collate class."""
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.ndim > 0 else None
    if isinstance(batch, Mapping):
        for value in batch.values():
            size = infer_batch_size(value)
            if size is not None:
                return size
        return None
    if isinstance(batch, tuple | list):
        for value in batch:
            size = infer_batch_size(value)
            if size is not None:
                return size
        return len(batch) if batch else None

    for attribute in ("cat_features", "num_features", "hidden_states"):
        if hasattr(batch, attribute):
            size = infer_batch_size(getattr(batch, attribute))
            if size is not None:
                return size
    return None


def _reduce(values: list[float], device: torch.device, operation) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=operation)
    return tensor.cpu().tolist()


def _sum(values: list[float], device: torch.device) -> list[float]:
    return _reduce(values, device, dist.ReduceOp.SUM)


def _max(values: list[float], device: torch.device) -> list[float]:
    return _reduce(values, device, dist.ReduceOp.MAX)


def _min(values: list[float], device: torch.device) -> list[float]:
    return _reduce(values, device, dist.ReduceOp.MIN)


def _resolve_nvml_handle(device_index: int):
    """Resolve a CUDA device after any ``CUDA_VISIBLE_DEVICES`` remapping."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        selectors = [item.strip() for item in visible_devices.split(",")]
        if device_index >= len(selectors):
            raise RuntimeError(
                f"CUDA device {device_index} is not present in CUDA_VISIBLE_DEVICES"
            )
        selector = selectors[device_index]
        if selector.isdigit():
            return pynvml.nvmlDeviceGetHandleByIndex(int(selector))
        return pynvml.nvmlDeviceGetHandleByUUID(selector)
    return pynvml.nvmlDeviceGetHandleByIndex(device_index)


@dataclass
class LocalSystemSnapshot:
    """One rank's host and GPU counters, before cross-rank reduction."""

    host_samples: int = 0
    cpu_mean: float = 0.0
    memory_peak: float = 0.0
    disk_read_mib_per_sec: float = 0.0
    network_receive_mib_per_sec: float = 0.0
    network_transmit_mib_per_sec: float = 0.0
    gpu_samples: int = 0
    gpu_utilization_mean: float = 0.0
    gpu_memory_peak: float = 0.0


@dataclass
class SystemMetricsCollector:
    """Collect local metrics without communication from the training loop."""

    is_local_main_process: bool
    device: torch.device
    sampling_interval_sec: float = 2.0
    _cpu_samples: list[float] = field(default_factory=list, init=False)
    _memory_samples: list[float] = field(default_factory=list, init=False)
    _gpu_utilization_samples: list[float] = field(default_factory=list, init=False)
    _gpu_memory_samples: list[float] = field(default_factory=list, init=False)
    _last_sample_time: float = field(default=0.0, init=False)
    _epoch_start_time: float = field(default=0.0, init=False)
    _disk_read_start: int = field(default=0, init=False)
    _network_receive_start: int = field(default=0, init=False)
    _network_transmit_start: int = field(default=0, init=False)
    _gpu_handle: Any = field(default=None, init=False)

    def __post_init__(self):
        if self.sampling_interval_sec <= 0:
            raise ValueError("sampling_interval_sec must be greater than zero")

        if self.is_local_main_process and psutil is None:
            warnings.warn(
                "Performance metrics are enabled, but psutil is unavailable; "
                "host system_epoch metrics will be omitted.",
                stacklevel=2,
            )
        elif self.is_local_main_process:
            psutil.cpu_percent(interval=None)

        if self.device.type == "cuda" and pynvml is not None:
            try:
                pynvml.nvmlInit()
                device_index = self.device.index if self.device.index is not None else 0
                self._gpu_handle = _resolve_nvml_handle(device_index)
            except Exception as error:  # pragma: no cover - hardware dependent
                warnings.warn(
                    f"Unable to initialize NVIDIA metrics: {error}",
                    stacklevel=2,
                )
        elif self.device.type == "cuda" and pynvml is None:
            warnings.warn(
                "Performance metrics are enabled, but nvidia-ml-py is unavailable; "
                "GPU system_epoch metrics will be omitted.",
                stacklevel=2,
            )

    def reset_epoch(self) -> None:
        self._cpu_samples.clear()
        self._memory_samples.clear()
        self._gpu_utilization_samples.clear()
        self._gpu_memory_samples.clear()
        self._epoch_start_time = time.perf_counter()
        self._last_sample_time = 0.0

        if self.is_local_main_process and psutil is not None:
            disk = psutil.disk_io_counters()
            network = psutil.net_io_counters()
            self._disk_read_start = disk.read_bytes if disk is not None else 0
            self._network_receive_start = (
                network.bytes_recv if network is not None else 0
            )
            self._network_transmit_start = (
                network.bytes_sent if network is not None else 0
            )

    def maybe_sample(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_sample_time < self.sampling_interval_sec:
            return
        self._last_sample_time = now

        if self.is_local_main_process and psutil is not None:
            self._cpu_samples.append(float(psutil.cpu_percent(interval=None)))
            self._memory_samples.append(float(psutil.virtual_memory().percent))

        if self._gpu_handle is not None:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self._gpu_utilization_samples.append(float(utilization.gpu))
                self._gpu_memory_samples.append(100.0 * memory.used / memory.total)
            except Exception:  # pragma: no cover - hardware dependent
                self._gpu_handle = None

    def snapshot(self) -> LocalSystemSnapshot:
        self.maybe_sample(force=True)
        elapsed = max(time.perf_counter() - self._epoch_start_time, 1e-12)
        snapshot = LocalSystemSnapshot(
            host_samples=len(self._cpu_samples),
            cpu_mean=(
                sum(self._cpu_samples) / len(self._cpu_samples)
                if self._cpu_samples
                else 0.0
            ),
            memory_peak=max(self._memory_samples, default=0.0),
            gpu_samples=len(self._gpu_utilization_samples),
            gpu_utilization_mean=(
                sum(self._gpu_utilization_samples) / len(self._gpu_utilization_samples)
                if self._gpu_utilization_samples
                else 0.0
            ),
            gpu_memory_peak=max(self._gpu_memory_samples, default=0.0),
        )

        if self.is_local_main_process and psutil is not None:
            disk = psutil.disk_io_counters()
            network = psutil.net_io_counters()
            if disk is not None:
                snapshot.disk_read_mib_per_sec = max(
                    disk.read_bytes - self._disk_read_start, 0
                ) / (1024**2 * elapsed)
            if network is not None:
                snapshot.network_receive_mib_per_sec = max(
                    network.bytes_recv - self._network_receive_start, 0
                ) / (1024**2 * elapsed)
                snapshot.network_transmit_mib_per_sec = max(
                    network.bytes_sent - self._network_transmit_start, 0
                ) / (1024**2 * elapsed)
        return snapshot


def reduce_system_snapshot(
    snapshot: LocalSystemSnapshot, device: torch.device
) -> dict[str, float]:
    """Reduce node/GPU summaries without gathering sample histories."""
    sums = _sum(
        [
            float(snapshot.host_samples > 0),
            snapshot.cpu_mean if snapshot.host_samples else 0.0,
            snapshot.disk_read_mib_per_sec,
            snapshot.network_receive_mib_per_sec,
            snapshot.network_transmit_mib_per_sec,
            float(snapshot.gpu_samples > 0),
            snapshot.gpu_utilization_mean if snapshot.gpu_samples else 0.0,
        ],
        device,
    )
    maxima = _max(
        [snapshot.cpu_mean, snapshot.memory_peak, snapshot.gpu_memory_peak],
        device,
    )
    gpu_min_value = snapshot.gpu_utilization_mean if snapshot.gpu_samples else math.inf
    gpu_min = _min([gpu_min_value], device)[0]

    host_count, cpu_sum, disk_sum, receive_sum, transmit_sum, gpu_count, gpu_sum = sums
    return {
        "system_epoch/host_sampler_count": host_count,
        "system_epoch/gpu_sampler_count": gpu_count,
        "system_epoch/cpu_utilization_pct_mean": (
            cpu_sum / host_count if host_count else 0.0
        ),
        "system_epoch/cpu_utilization_pct_max": maxima[0],
        "system_epoch/memory_utilization_pct_max": maxima[1],
        "system_epoch/gpu_utilization_pct_mean": (
            gpu_sum / gpu_count if gpu_count else 0.0
        ),
        "system_epoch/gpu_utilization_pct_min": (
            gpu_min if math.isfinite(gpu_min) else 0.0
        ),
        "system_epoch/gpu_memory_utilization_pct_max": maxima[2],
        "system_epoch/disk_read_mib_per_sec": disk_sum,
        "system_epoch/network_receive_mib_per_sec": receive_sum,
        "system_epoch/network_transmit_mib_per_sec": transmit_sum,
    }


def reduce_epoch_performance(
    *,
    device: torch.device,
    epoch_wall_sec: float,
    local_samples: int,
    local_batches: int,
    shard_stats: dict[str, float] | None = None,
    total_raw_rows: int = 0,
    total_parquet_bytes: int = 0,
) -> dict[str, float]:
    """Reduce training and optional shard metrics using scalar collectives only."""
    sample_sum = _sum([float(local_samples)], device)[0]
    wall_max, samples_max, batches_max = _max(
        [epoch_wall_sec, float(local_samples), float(local_batches)], device
    )
    samples_min, batches_min = _min(
        [float(local_samples), float(local_batches)], device
    )
    metrics = {
        "train_epoch/wall_sec": wall_max,
        "train_epoch/global_samples": sample_sum,
        "train_epoch/global_samples_per_sec": (
            sample_sum / wall_max if wall_max else 0.0
        ),
    }

    if shard_stats is None:
        return metrics

    raw_rows = float(shard_stats.get("source_rows_scanned", 0.0))
    parquet_bytes = float(shard_stats.get("parquet_bytes_opened", 0.0))
    filter_sec = float(shard_stats.get("filter_sec", 0.0))
    raw_rows_sum, parquet_bytes_sum = _sum([raw_rows, parquet_bytes], device)
    filter_fraction_max = _max(
        [filter_sec / epoch_wall_sec if epoch_wall_sec else 0.0], device
    )[0]
    metrics.update({
        "shard_epoch/row_read_amplification": (
            raw_rows_sum / total_raw_rows if total_raw_rows else 0.0
        ),
        "shard_epoch/io_read_amplification": (
            parquet_bytes_sum / total_parquet_bytes if total_parquet_bytes else 0.0
        ),
        "shard_epoch/filter_time_pct": 100.0 * filter_fraction_max,
        "shard_epoch/record_count_delta": samples_max - samples_min,
        "shard_epoch/batch_count_delta": batches_max - batches_min,
    })
    return metrics


def reduce_max_duration(duration: float, device: torch.device) -> float:
    """Return the slowest rank duration."""
    return _max([duration], device)[0]


def configure_mlflow_system_metrics(
    enabled: bool, sampling_interval_sec: float
) -> None:
    """Enable MLflow's node-0 time series before its run is created."""
    if not enabled:
        return
    try:
        import mlflow

        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(max(int(sampling_interval_sec), 1))
    except (AttributeError, ImportError, ValueError) as error:
        warnings.warn(
            f"Unable to enable native MLflow system metrics: {error}",
            stacklevel=2,
        )
