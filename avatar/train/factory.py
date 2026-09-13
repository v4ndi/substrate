"""Assemble the callback list for a run from its config.

A config may list callbacks explicitly::

    callbacks:
      - _target_: avatar.train.MLflowCallback
      - _target_: avatar.train.ProgressBarCallback

When the key is absent the default list below is used, which reproduces what
the old inline loop did.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from avatar.train.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    EMACallback,
    MLflowCallback,
    PerfMetricsCallback,
    ProfilerCallback,
    ProgressBarCallback,
    ThroughputCallback,
    TrainerCallback,
    TrainMetricsCallback,
    TrainStatsCallback,
)
from avatar.train.utils import get_default_log_params
from avatar.utils.init_modules import init_profiler, init_swa_model

DEFAULT_PERFORMANCE_CONFIG = {
    "enabled": False,
    "sampling_interval_sec": 2.0,
    "native_system_metrics": True,
}


def resolve_performance_config(config: DictConfig) -> dict[str, Any]:
    """Merge the ``logging.performance_metrics`` block over its defaults."""
    resolved = dict(DEFAULT_PERFORMANCE_CONFIG)
    logging_config = config.get("logging") if "logging" in config else None
    if logging_config and "performance_metrics" in logging_config:
        configured = OmegaConf.to_container(logging_config["performance_metrics"])
        if configured:
            resolved.update(configured)
    return resolved


def dataset_log_params(
    train_dataloader, performance_config: dict[str, Any]
) -> dict[str, Any]:
    """Shard and environment facts worth recording alongside the run."""
    if not performance_config.get("enabled"):
        return {}
    dataset = train_dataloader.dataset
    available_cpu_cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    return {
        "performance_metrics_enabled": True,
        "performance_metrics_sampling_interval_sec": performance_config[
            "sampling_interval_sec"
        ],
        "shard": bool(getattr(dataset, "shard", False)),
        "drop_tail": getattr(dataset, "drop_tail", None),
        "rotate_tail": getattr(dataset, "rotate_tail", None),
        "filter_cache": getattr(dataset, "filter_cache", None),
        "filter_cache_dir": getattr(dataset, "filter_cache_dir", None),
        "scan_workers_per_rank": getattr(dataset, "scan_workers", None),
        "scan_ranks": getattr(dataset, "scan_ranks", None),
        "effective_scan_ranks": getattr(dataset, "scan_rank_count", None),
        "batch_size_per_rank": train_dataloader.batch_size,
        "num_workers_per_rank": train_dataloader.num_workers,
        "parquet_file_count": len(getattr(dataset, "files", [])),
        "dataset_size_bytes": getattr(dataset, "total_parquet_bytes", None),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "available_cpu_cores": available_cpu_cores,
    }


def build_default_callbacks(
    config: DictConfig,
    *,
    model,
    train_dataloader,
    checkpoint_dir: str,
    early_stopping=None,
    train_metrics=None,
    startup_begin_time: float | None = None,
) -> list[TrainerCallback]:
    """The stock callback list: what the old inline loop did, unbundled."""
    performance_config = resolve_performance_config(config)
    logging_enabled = bool(config.get("logging", {}).get("enable", True))
    profiler_enabled = bool(config.get("logging", {}).get("enable_profiler", False))
    performance_enabled = bool(logging_enabled and performance_config.get("enabled"))

    params = get_default_log_params(config)
    params.update(dataset_log_params(train_dataloader, performance_config))
    if performance_enabled and performance_config.get("native_system_metrics", True):
        params["system_metrics_scope"] = "node_0"

    mlflow_arguments = (
        OmegaConf.to_container(config["mlflow"]) if "mlflow" in config else {}
    )
    callbacks: list[TrainerCallback] = [
        MLflowCallback(
            experiment_name=mlflow_arguments.get("experiment_name"),
            run_name=mlflow_arguments.get("run_name"),
            tracking_uri=mlflow_arguments.get("tracking_uri"),
            params=params,
            enabled=logging_enabled and bool(mlflow_arguments),
        ),
        ProgressBarCallback(enabled=logging_enabled),
        TrainStatsCallback(),
    ]

    swa_model, min_num_steps, min_epoch = init_swa_model(config, model)
    if swa_model is not None:
        callbacks.append(
            EMACallback(swa_model, update_every=min_num_steps, min_epoch=min_epoch)
        )
    if train_metrics is not None:
        callbacks.append(TrainMetricsCallback(train_metrics))
    if performance_enabled:
        callbacks.append(
            PerfMetricsCallback(
                sampling_interval_sec=float(
                    performance_config.get("sampling_interval_sec", 2.0)
                ),
                startup_begin_time=startup_begin_time or time.perf_counter(),
            )
        )
        callbacks.append(ThroughputCallback())
    if profiler_enabled:
        callbacks.append(ProfilerCallback(init_profiler(mlflow_arguments)))

    # Early stopping must precede the checkpointer: it is what vetoes a save.
    if early_stopping is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping))
    callbacks.append(
        CheckpointCallback(
            checkpoint_dir,
            max_checkpoints=config.get("train", {}).get("max_saved_checkpoints"),
        )
    )
    return callbacks


def build_callbacks(config: DictConfig, **kwargs) -> list[TrainerCallback]:
    """Instantiate ``callbacks:`` from config, or fall back to the default list."""
    if config.get("callbacks"):
        return [instantiate(entry) for entry in config["callbacks"]]
    return build_default_callbacks(config, **kwargs)
