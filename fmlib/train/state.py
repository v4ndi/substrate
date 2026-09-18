"""Mutable state the loop publishes to callbacks, and the flags they steer it with."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from fmlib.train.config import RunConfig
from fmlib.train.dist import DistEnv


@dataclass
class TrainerState:
    """Where the run currently is. Read by callbacks, written by the loop."""

    epoch: int = 0
    num_epochs: int = 0
    global_step: int = 0
    micro_step: int = 0
    epoch_step: int = 0
    epoch_batches: int = 0
    max_steps: int | None = None
    world_size: int = 1
    samples_seen: int = 0
    epoch_samples: int = 0
    learning_rate: float = 0.0
    log_history: list[dict[str, Any]] = field(default_factory=list)
    best_metric: float | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "best_metric": self.best_metric,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = state.get("epoch", self.epoch)
        self.global_step = state.get("global_step", self.global_step)
        self.samples_seen = state.get("samples_seen", self.samples_seen)
        self.best_metric = state.get("best_metric", self.best_metric)


@dataclass
class TrainerControl:
    """Callback-writable flags. The loop reads them at defined points."""

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_evaluate: bool = False
    should_save: bool = False
    should_log: bool = False

    def reset_step_flags(self) -> None:
        self.should_evaluate = False
        self.should_save = False
        self.should_log = False


@dataclass
class CallbackContext:
    """Everything a callback may look at, assembled once per event.

    ``model`` is always the *unwrapped* module — callbacks that save weights or
    read parameters should never have to know whether DDP is in play. The loop
    keeps the wrapped one to itself.
    """

    env: DistEnv
    state: TrainerState
    control: TrainerControl
    run_config: RunConfig
    config: Any = None
    model: torch.nn.Module | None = None
    optimizer: Any = None
    scheduler: Any = None
    batch: Any = None
    output: Any = None
    loss: torch.Tensor | None = None
    grad_norm: torch.Tensor | float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    log_step: int = 0
    checkpoint_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Set by the Trainer: ``ctx.log({"name": value}, step=...)`` publishes
    # metrics to whichever logging callbacks are installed.
    log: Callable[..., None] | None = None
