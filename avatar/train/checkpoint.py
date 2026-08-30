"""Explicit checkpoint save/load — one file, one ``torch.save``.

Replaces ``accelerator.save_state``/``load_state`` and the parallel
``model.bin`` + ``<step>/checkpoints/pytorch_model.bin`` layout that the eval
path used to reach into by hand.
"""

from __future__ import annotations

import os
import random
import shutil
from glob import glob
from typing import Any

import numpy as np
import torch

from avatar.train.dist import unwrap_model

CHECKPOINT_NAME = "checkpoint.pt"
MODEL_NAME = "model.bin"


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in state["cuda"]])


def checkpoint_path(directory: str, step: int) -> str:
    return os.path.join(directory, str(step), CHECKPOINT_NAME)


def model_path(directory: str, step: int) -> str:
    return os.path.join(directory, str(step), MODEL_NAME)


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    state: Any = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write the full training state to ``path``. Rank 0 only — the caller decides."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "model": unwrap_model(model).state_dict(),
        "rng": _rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    if state is not None:
        payload["state"] = state.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module | None = None,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    state: Any = None,
    map_location: Any = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore whatever the caller passed in; return the raw payload as well."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in payload:
        unwrap_model(model).load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if state is not None and "state" in payload:
        state.load_state_dict(payload["state"])
    if restore_rng:
        _restore_rng_state(payload.get("rng"))
    return payload


def save_model(model: torch.nn.Module, path: str) -> str:
    """Write only the weights — what inference and the test phase load."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)
    return path


def rotate_checkpoints(directory: str, max_checkpoints: int | None) -> None:
    """Keep the ``max_checkpoints`` most recently written step directories."""
    if max_checkpoints is None:
        return
    checkpoints = sorted(glob(f"{directory}/*"), key=os.path.getmtime)
    for stale in checkpoints[:-max_checkpoints]:
        shutil.rmtree(stale, ignore_errors=True)


def latest_checkpoint_step(directory: str) -> int | None:
    """Highest numbered step directory under ``directory``, or None if empty."""
    if not os.path.isdir(directory):
        return None
    steps = [int(name) for name in os.listdir(directory) if name.isdigit()]
    return max(steps) if steps else None
