"""Process-group setup and the handful of collectives the trainer needs.

This replaces ``accelerate.Accelerator``'s distributed surface. Everything the
training loop does across ranks goes through :class:`DistEnv`: ``all_reduce``
for scalars, ``all_gather_object`` for metric payloads, and ``barrier``.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

DEFAULT_TIMEOUT_SEC = 36_000_000


def default_backend(device: torch.device) -> str:
    """NCCL when we actually have CUDA devices, gloo otherwise.

    gloo is what makes multi-rank tests runnable on a single-GPU (or CPU-only)
    host: two NCCL ranks cannot share one device, two gloo ranks can.
    """
    return "nccl" if device.type == "cuda" else "gloo"


@dataclass(frozen=True)
class DistEnv:
    """The distributed facts of the current process.

    Built once at startup from the environment ``torchrun`` provides. Holding
    it as a frozen dataclass keeps the "which rank am I" question answerable
    without reaching into global state at every call site.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        return self.local_rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @classmethod
    def from_env(
        cls,
        backend: str | None = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> DistEnv:
        """Read ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` and join the process group.

        A single process (no ``torchrun``) gets ``world_size == 1`` and no
        process group at all, so the same code path runs under a plain
        ``python -m avatar.train``.
        """
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend=backend or default_backend(device),
                timeout=timedelta(seconds=timeout_sec),
            )
        if dist.is_initialized():
            # torchrun is the source of truth for the geometry, but a group
            # initialised elsewhere (tests, notebooks) wins over stale env vars.
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        return cls(
            rank=rank, local_rank=local_rank, world_size=world_size, device=device
        )

    def barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def all_reduce(self, value: float, op: Any = None) -> float:
        """Reduce one scalar across ranks and return it on every rank."""
        op = dist.ReduceOp.SUM if op is None else op
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=self.device)
        if self.distributed and dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def all_reduce_sum(self, value: float) -> float:
        return self.all_reduce(value, dist.ReduceOp.SUM)

    def all_reduce_max(self, value: float) -> float:
        return self.all_reduce(value, dist.ReduceOp.MAX)

    def all_reduce_min(self, value: float) -> float:
        return self.all_reduce(value, dist.ReduceOp.MIN)

    def all_reduce_mean(self, value: float) -> float:
        return self.all_reduce_sum(value) / self.world_size

    def all_reduce_any(self, flag: bool) -> bool:
        """True on every rank as soon as it is true on any one of them."""
        return self.all_reduce_max(float(bool(flag))) > 0.0

    def broadcast_flag(self, flag: bool) -> bool:
        """Give every rank rank 0's answer.

        Used for decisions only rank 0 can make — it is the only rank holding
        the evaluation scores — but that every rank must act on together, such
        as "stop training" or "write a checkpoint".
        """
        return self.all_reduce_sum(float(bool(flag)) if self.is_main else 0.0) > 0.0

    def gather_objects(self, obj: Any) -> list[Any]:
        """Gather one picklable object per rank onto every rank, in rank order."""
        if not (self.distributed and dist.is_initialized()):
            return [obj]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered

    def destroy(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print on the main process only — the old ``accelerator.print``."""
        if self.is_main:
            print(*args, **kwargs)


def seed_everything(seed: int, rank: int = 0, device_specific: bool = False) -> None:
    """Seed python, numpy and torch.

    ``device_specific`` offsets the seed by rank. It stays False by default:
    every rank drawing the same dropout mask is the behaviour the old
    ``set_seed(..., device_specific=False)`` call had.
    """
    effective = seed + rank if device_specific else seed
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP and ``AveragedModel`` wrappers off to reach the real module."""
    while True:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        elif isinstance(model, torch.optim.swa_utils.AveragedModel):
            model = model.module
        else:
            return model
