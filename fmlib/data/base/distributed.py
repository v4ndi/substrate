"""Rank / worker resolution shared by every sharded dataset.

Datasets are built before the trainer wraps them, sometimes before
``torch.distributed`` is initialised, and are then re-entered inside DataLoader
worker processes. These helpers give one answer to "who am I" in all three
situations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

__all__ = ["WorkerInfo", "resolve_dist_info", "resolve_worker_info"]


def resolve_dist_info() -> tuple[int, int]:
    """Return ``(world_size, rank)`` from torch or launcher variables.

    Prefers an initialised process group; falls back to the ``WORLD_SIZE`` /
    ``RANK`` variables ``torchrun`` exports, so a dataset built before
    ``init_process_group`` still shards correctly.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))


@dataclass(frozen=True)
class WorkerInfo:
    """Where the current code is running within the rank x worker grid."""

    num_workers: int
    worker_id: int
    world_size: int
    rank: int

    @property
    def total_shards(self) -> int:
        return self.num_workers * self.world_size


def resolve_worker_info(
    world_size: int | None = None, rank: int | None = None
) -> WorkerInfo:
    """Resolve DataLoader worker and distributed rank information.

    Args:
        world_size: Overrides the resolved world size. Used to honour the value
            captured at dataset construction time when the process group came up
            later and reports a smaller world.
        rank: Overrides the resolved rank.
    """
    torch_worker_info = torch.utils.data.get_worker_info()
    if torch_worker_info is None:
        num_workers, worker_id = 1, 0
    else:
        num_workers, worker_id = torch_worker_info.num_workers, torch_worker_info.id

    current_world_size, current_rank = resolve_dist_info()
    if world_size is None or rank is None:
        world_size, rank = current_world_size, current_rank
    elif (world_size, rank) == (1, 0) or current_world_size > 1:
        # A process group that came up after construction wins over the stale
        # single-process defaults captured in __init__.
        world_size, rank = current_world_size, current_rank

    return WorkerInfo(
        num_workers=num_workers,
        worker_id=worker_id,
        world_size=world_size,
        rank=rank,
    )
