"""Small helpers shared by the loop, the evaluator and the entrypoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from avatar.metrics import BaseMetric

IGNORE_CONFIG_KEYS = ["tokens_meta", "_target_"]
PRESERVE_CONFIG_KEYS = ["filters"]

# Config sections worth recording as MLflow params. Missing sections are skipped
# rather than logged as None, so a config that never had `swa_model` stays clean.
LOGGED_CONFIG_SECTIONS = (
    "model",
    "distributed",
    "ddp",
    "amp",
    "compile",
    "accelerator",
    "optimizer",
    "scheduler",
    "swa_model",
    "train",
    "train_dataloader",
    "valid_dataloader",
)


def move_to_device(
    data: torch.Tensor | dict[str, Any] | list[Any] | tuple[Any],
    device: str | torch.device,
) -> torch.Tensor | dict[str, Any] | list[Any] | tuple[Any]:
    """Recursively move data and all nested contents to the specified device.

    This function handles multiple data types including:
    - PyTorch tensors
    - Dictionaries containing tensors
    - Lists and tuples of tensors
    - Any object exposing a ``.to`` method (the batch containers do)

    Args:
        data: Input data to move.
        device: Target device, as a string or ``torch.device``.

    Returns:
        The same data structure with all tensors moved to the target device.
    """
    if isinstance(data, torch.Tensor) or hasattr(data, "to"):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list | tuple):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def wrap_metrics(metrics: BaseMetric | list[BaseMetric] | None) -> list | None:
    """Normalise a metric, a list of metrics or None into a list or None."""
    if OmegaConf.is_list(metrics):
        metrics = OmegaConf.to_container(metrics)
    elif not OmegaConf.is_list(metrics) and metrics is not None:
        metrics = [metrics]
    return metrics


def prefix_metrics(
    metrics: dict[str, Any], prefix: str | None = None
) -> dict[str, float]:
    """Prefix metric names and coerce tensor values to floats for logging."""
    prepared = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if prefix else key
        prepared[name] = value.item() if isinstance(value, torch.Tensor) else value
    return prepared


def flatten_dict(
    params_dict: dict[str, Any] | DictConfig,
    parent_key: str = "",
    sep: str = "_",
    ignore_keys: set[str] | list | None = None,
    preserve_keys: set[str] | list | None = None,
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary or DictConfig into a single level.

    Args:
        params_dict: Input dictionary or DictConfig to flatten. Can be nested.
        parent_key: Base key string used for recursion (empty for top-level).
        sep: Separator to use between concatenated keys.
        ignore_keys: Keys to exclude from the output.
        preserve_keys: Container keys to retain as one value instead of flattening.

    Returns:
        A flattened dictionary whose keys are paths joined by ``sep``.
    """
    if ignore_keys is None:
        ignore_keys = set()
    else:
        ignore_keys = set(ignore_keys)
    if preserve_keys is None:
        preserve_keys = set()
    else:
        preserve_keys = set(preserve_keys)
    items = {}
    for k, v in params_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if k in ignore_keys:
            continue

        if k in preserve_keys:
            items[new_key] = v
        elif isinstance(v, DictConfig):
            items.update(
                flatten_dict(
                    v,
                    new_key,
                    sep=sep,
                    ignore_keys=ignore_keys,
                    preserve_keys=preserve_keys,
                )
            )
        else:
            items[new_key] = v

    return items


def normalize_log_param(value: Any) -> Any:
    """Convert resolved config values to logging-safe primitive containers.

    Hydra resolvers may return NumPy values such as ``np.datetime64``. Those
    values are valid filter bounds, but OmegaConf and MLflow cannot serialize
    them as run parameters without normalization.
    """
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, np.datetime64 | np.timedelta64):
        return str(value)
    if isinstance(value, np.ndarray):
        return [normalize_log_param(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return normalize_log_param(value.item())
    if isinstance(value, dict):
        return {str(key): normalize_log_param(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [normalize_log_param(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((normalize_log_param(item) for item in value), key=repr)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def get_default_log_params(config: DictConfig) -> dict[str, Any]:
    """Flatten the run-shaping config sections into MLflow params."""
    sections = {name: config[name] for name in LOGGED_CONFIG_SECTIONS if name in config}
    return normalize_log_param(
        flatten_dict(
            sections,
            ignore_keys=IGNORE_CONFIG_KEYS,
            preserve_keys=PRESERVE_CONFIG_KEYS,
        )
    )


def apply_basic_loader_checks(*dataloaders) -> None:
    """``lazy_process`` datasets need ``drop_last`` — a ragged tail breaks them."""
    for dataloader in dataloaders:
        if dataloader is None:
            continue
        if getattr(dataloader.dataset, "lazy_process", False):
            assert dataloader.drop_last, (
                "If you want to use lazy_process=True, set dataloader.drop_last=True"
            )
