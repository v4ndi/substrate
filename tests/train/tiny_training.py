"""A minimal model and sharded dataset for exercising the training loop.

Small enough to train in under a second, but shaped like the real thing: the
dataset shards itself by rank (``shard = True``), exposes ``set_epoch`` and
``__len__``, and the model returns an output object with a ``.loss`` attribute.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class TinyOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class TinyModel(nn.Module):
    """Logistic regression over ``num_features`` inputs."""

    def __init__(self, num_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, features, target):
        logits = self.linear(features).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target.float())
        return TinyOutput(loss=loss, logits=logits)


class TinyShardedDataset(IterableDataset):
    """A learnable synthetic task, split into equal contiguous slices per rank.

    Equal slices are the point: a rank that yields a different number of
    batches than its neighbours deadlocks DDP at the first gradient
    all-reduce, so the dataset drops the global remainder just like the real
    sharded datasets do.
    """

    shard = True

    def __init__(
        self,
        num_records: int = 512,
        num_features: int = 4,
        seed: int = 0,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.num_records = num_records
        self.num_features = num_features
        self.seed = seed
        self.world_size = max(1, world_size)
        self.rank = rank
        self.epoch = 0

        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(
            num_records, num_features, generator=generator, dtype=torch.float32
        )
        weights = torch.randn(num_features, generator=generator)
        self.targets = (self.features @ weights > 0).to(torch.float32)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @property
    def per_rank(self) -> int:
        return self.num_records // self.world_size

    def __len__(self) -> int:
        return self.per_rank

    def _worker_slice(self) -> tuple[int, int]:
        start = self.rank * self.per_rank
        stop = start + self.per_rank
        info = get_worker_info()
        if info is None or info.num_workers == 1:
            return start, stop
        base, remainder = divmod(self.per_rank, info.num_workers)
        offset = info.id * base + min(info.id, remainder)
        count = base + int(info.id < remainder)
        return start + offset, start + offset + count

    def __iter__(self):
        start, stop = self._worker_slice()
        for index in range(start, stop):
            yield {
                "features": self.features[index],
                "target": self.targets[index],
                "record_id": index,
            }


def collate(records: list[dict]) -> dict:
    return {
        "features": torch.stack([record["features"] for record in records]),
        "target": torch.stack([record["target"] for record in records]),
    }


def parameter_fingerprint(model: nn.Module) -> list[float]:
    """A short, comparable summary of the weights, for cross-rank equality."""
    return [round(float(p.detach().double().sum()), 6) for p in model.parameters()]
