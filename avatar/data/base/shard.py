"""Record-level sharding of a parquet corpus across ranks and workers.

The unit of ownership is a *valid record* — a row that survives whatever filter
the dataset applies — not a file. Files vary in size and in how many rows pass
the filter, so splitting by file leaves ranks with unequal cardinality, and an
``IterableDataset`` whose ranks disagree on the number of batches deadlocks DDP
at the first gradient all-reduce.

The planner therefore works on a global prefix sum of per-file valid-record
counts. Each ``(rank, worker)`` owns one contiguous slice of ``[0, total)``,
which is mapped back onto ``(file, lo, hi)`` filtered-row segments. With
``drop_tail`` every rank owns exactly ``total // world_size`` records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ShardPlanner", "ShardSegment"]


@dataclass(frozen=True)
class ShardSegment:
    """A half-open ``[start, stop)`` range of valid records inside one file.

    Offsets count *valid* records — the n-th record of the file that passes the
    dataset's filter — not physical parquet rows.
    """

    file_index: int
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


class ShardPlanner:
    """Decides which records each ``(rank, worker)`` reads.

    Args:
        file_counts: Number of valid records per file, in canonical file order.
        world_size: Number of distributed ranks.
        rank: This process's rank.
        drop_tail: Drop the global remainder so every rank owns exactly the same
            number of records. Required for training; disable it for evaluation
            when no sample may be lost, at the cost of an uneven last rank.
        rotate_tail: Rotate the global stream by the remainder each epoch, so
            the records dropped by ``drop_tail`` differ from epoch to epoch.
    """

    def __init__(
        self,
        file_counts: np.ndarray,
        world_size: int,
        rank: int,
        drop_tail: bool = True,
        rotate_tail: bool = False,
    ):
        if world_size < 1:
            raise ValueError("world_size must be greater than zero")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is out of range for world_size {world_size}")

        self.file_counts = np.asarray(file_counts, dtype=np.int64)
        if self.file_counts.ndim != 1:
            raise ValueError("file_counts must be a one-dimensional array")
        if (self.file_counts < 0).any():
            raise ValueError("file_counts must not contain negative values")

        self.world_size = world_size
        self.rank = rank
        self.drop_tail = drop_tail
        self.rotate_tail = rotate_tail
        self._cum = np.concatenate([[0], np.cumsum(self.file_counts)])

    # ------------------------------------------------------------------ #
    # Global geometry                                                     #
    # ------------------------------------------------------------------ #
    @property
    def total(self) -> int:
        """Valid records across the whole corpus."""
        return int(self._cum[-1])

    @property
    def per_rank(self) -> int:
        """Records every rank owns, before the ``drop_tail`` remainder."""
        return self.total // self.world_size

    @property
    def tail(self) -> int:
        """Records left over after an equal split."""
        return self.total % self.world_size

    def rank_record_count(self) -> int:
        """Records this rank yields per epoch — the value ``__len__`` returns."""
        keeps_tail = not self.drop_tail and self.rank == self.world_size - 1
        return self.per_rank + (self.tail if keeps_tail else 0)

    def worker_record_count(self, worker_id: int, num_workers: int) -> int:
        """Records one DataLoader worker of this rank yields per epoch."""
        base, remainder = divmod(self.rank_record_count(), num_workers)
        return base + int(worker_id < remainder)

    # ------------------------------------------------------------------ #
    # Planning                                                            #
    # ------------------------------------------------------------------ #
    def worker_segments(
        self, worker_id: int = 0, num_workers: int = 1, epoch: int = 0
    ) -> list[ShardSegment]:
        """Return the ``(file_index, lo, hi)`` segments this worker owns."""
        if num_workers < 1:
            raise ValueError("num_workers must be greater than zero")
        if not 0 <= worker_id < num_workers:
            raise ValueError(
                f"worker_id {worker_id} is out of range for num_workers {num_workers}"
            )

        total = self.total
        rank_count = self.rank_record_count()
        base, remainder = divmod(rank_count, num_workers)
        worker_relative_start = worker_id * base + min(worker_id, remainder)
        worker_count = base + int(worker_id < remainder)

        if total == 0 or worker_count == 0:
            return []

        worker_start = (
            self._epoch_offset(epoch)
            + self.rank * self.per_rank
            + worker_relative_start
        ) % total

        segments: list[ShardSegment] = []
        for start, stop in self._split_cyclic_range(worker_start, worker_count, total):
            segments.extend(self._range_to_segments(start, stop))
        return segments

    def _epoch_offset(self, epoch: int) -> int:
        """A deterministic rotation of the global valid-record stream.

        Rotating by the remainder means a different ``tail`` records fall off
        the end each epoch, so ``drop_tail`` does not permanently hide the same
        rows from training.
        """
        if not self.rotate_tail or self.tail == 0 or self.total == 0:
            return 0
        return (int(epoch) * self.tail) % self.total

    @staticmethod
    def _split_cyclic_range(
        start: int, length: int, total: int
    ) -> list[tuple[int, int]]:
        """Split a cyclic logical range into at most two ordinary ranges."""
        if length <= 0:
            return []
        if total <= 0:
            raise ValueError("total must be positive for a non-empty cyclic range")
        if length > total:
            raise ValueError("cyclic range length cannot exceed total")

        start %= total
        stop = start + length
        if stop <= total:
            return [(start, stop)]
        return [(start, total), (0, stop - total)]

    def _range_to_segments(self, start: int, stop: int) -> list[ShardSegment]:
        """Map one non-cyclic global logical range to per-file record slices."""
        cum = self._cum
        first = max(int(np.searchsorted(cum, start, side="right")) - 1, 0)
        last = min(int(np.searchsorted(cum, stop, side="left")), len(self.file_counts))
        segments = []
        for index in range(first, last):
            lo = max(start, int(cum[index])) - int(cum[index])
            hi = min(stop, int(cum[index + 1])) - int(cum[index])
            if hi > lo:
                segments.append(ShardSegment(file_index=index, start=lo, stop=hi))
        return segments

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """One-line summary used in the dataset's construction-time warning."""
        counts = self.file_counts
        if len(counts) == 0:
            return "[shard] no files"
        return (
            f"[shard] files: {len(counts)}, records per file: "
            f"min {int(counts.min())}, mean {counts.mean():.1f}, "
            f"std {counts.std():.1f}, max {int(counts.max())}\n"
            f"[shard] records: total {self.total}, per rank {self.per_rank}, "
            f"tail {self.tail} "
            f"({'dropped' if self.drop_tail else 'to the last rank'}), "
            f"epoch rotation {'enabled' if self.rotate_tail else 'disabled'}"
        )
