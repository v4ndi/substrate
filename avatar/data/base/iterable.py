"""Base iterable parquet datasets.

:class:`BaseParquetDataset` handles file discovery and the plain per-worker file
split. :class:`BaseShardedParquetDataset` adds record-level sharding across
ranks and workers on top of it, and is what every modality dataset derives from.

The sharded base owns the whole pipeline — scan, plan, read, slice, process —
and asks subclasses for exactly two things:

``count_valid_records(file)``
    How many records of ``file`` pass this dataset's filter.
``iter_file_records(file)``
    Those records, in physical order.

The two must agree exactly. They are the same predicate evaluated two ways: once
vectorised over a column subset during the scan, once row-wise during iteration.
A mismatch means ranks disagree about who owns what, so iteration checks the
counts and fails loudly rather than letting DDP hang.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

import numpy as np
import pyarrow.fs as pafs
import torch
from tqdm import tqdm

from avatar.data.base.distributed import resolve_dist_info, resolve_worker_info
from avatar.data.base.fs import discover_parquet_files, file_size, resolve_filesystems
from avatar.data.base.parquet import parquet_num_rows, read_parquet_file
from avatar.data.base.shard import ShardPlanner
from avatar.data.sampler import BaseSampler

__all__ = ["BaseParquetDataset", "BaseShardedParquetDataset"]


class BaseParquetDataset(torch.utils.data.IterableDataset):
    """Iterate parquet files, optionally through a sampler.

    Args:
        path: Directory (or list of directories) holding parquet files. A path
            may be a local path or a URI on a filesystem pyarrow understands,
            e.g. ``hdfs://namenode:8020/user/team/dataset``.
        filesystem: Explicit :class:`pyarrow.fs.FileSystem`, or a mapping of
            options such as ``{"type": "hdfs", "host": ..., "port": ...}``.
            ``None`` infers it from ``path``.
        read_columns: Columns to read. ``None`` reads all of them.
        shuffle_files: Shuffle the order files are read in.
        shuffle_pq: Shuffle rows within each file.
        sampler: Object with ``__iter__`` and ``set_dataset_iterator``, wrapping
            the record stream.

    Raises:
        AssertionError: If a path does not exist or holds no parquet files.
    """

    def __init__(
        self,
        path: str | list,
        read_columns: list[str] | None = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        sampler: BaseSampler = None,
        filesystem: pafs.FileSystem | dict | None = None,
    ):
        super().__init__()
        self.filesystem, self._base_paths = resolve_filesystems(path, filesystem)
        self.files = discover_parquet_files(self.filesystem, self._base_paths)
        assert len(self.files) != 0, f"Find empty folder: {path}"
        self._shuffle_files = shuffle_files
        if shuffle_files:
            self.shuffle_files()
        self.read_columns = (
            list(read_columns) if read_columns is not None else read_columns
        )
        self.shuffle_pq = shuffle_pq
        self.sampler = sampler
        self._total_length: int | None = None
        self._total_parquet_bytes: int | None = None
        self._rng: np.random.Generator | None = None

    @property
    def rng(self) -> np.random.Generator:
        """Per-``(rank, worker, epoch)`` generator for reproducible sampling."""
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    @property
    def total_length(self) -> int:
        """Physical rows across every file (lazily counted)."""
        if self._total_length is None:
            self._total_length = sum(
                parquet_num_rows(file, self.filesystem) for file in self.files
            )
        return self._total_length

    @property
    def total_parquet_bytes(self) -> int:
        """On-disk size of the whole corpus (lazily measured)."""
        if self._total_parquet_bytes is None:
            self._total_parquet_bytes = sum(
                file_size(self.filesystem, file) for file in self.files
            )
        return self._total_parquet_bytes

    def shuffle_files(self) -> None:
        """Shuffles the list of parquet files."""
        np.random.shuffle(self.files)

    def files_per_worker(self) -> tuple[int, int]:
        """Distribute whole files among DataLoader workers.

        Returns:
            tuple[int, int]: The start and end indices of files for the current worker.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > len(self.files):
            warnings.warn(
                f"Number of workers ({worker_info.num_workers}) is greater than "
                f"number of files ({len(self.files)}). Some workers will not "
                "receive any work.",
                RuntimeWarning,
                stacklevel=2,
            )

        if worker_info is None:
            return 0, len(self.files)

        per_worker = math.ceil(len(self.files) / float(worker_info.num_workers))
        iter_start = worker_info.id * per_worker
        iter_end = min(iter_start + per_worker, len(self.files))
        return iter_start, iter_end

    def __len__(self) -> int:
        """Returns the total number of rows across all parquet files."""
        return self.total_length

    def process(self, record: dict[str, Any]) -> dict[str, Any]:
        """Convert one raw record into its trainable form.

        Called after ownership has been decided, so for a sharded dataset this
        must never drop a record — filtering belongs in ``iter_file_records``.
        """
        return record

    def _iter_impl(self) -> Iterator[dict[str, Any]]:
        iter_start, iter_end = self.files_per_worker()
        for index in range(iter_start, iter_end):
            records = read_parquet_file(
                file=self.files[index],
                columns=self.read_columns,
                shuffle=self.shuffle_pq,
                filesystem=self.filesystem,
            )
            for record in records:
                processed_record = self.process(record)
                if processed_record is not None:
                    yield processed_record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.sampler is None:
            return self._iter_impl()
        self.sampler.set_dataset_iterator(self._iter_impl())
        return self.sampler.__iter__()


class BaseShardedParquetDataset(BaseParquetDataset):
    """Parquet dataset that splits its records across ranks and workers.

    Args:
        shard: Distribute the global record stream. Disable only for debugging
            or tiny corpora — it skips the startup scan and falls back to the
            whole-file worker split, which gives ranks unequal cardinality.
        drop_tail: Drop the global remainder so every rank yields exactly the
            same number of records. Required for DDP training.
        rotate_tail: Rotate the global stream by the remainder each epoch, so
            ``drop_tail`` does not always hide the same records.
        seed: Base seed for per-``(rank, worker, epoch)`` shuffling.
        scan_workers: Threads used to count valid records per file.
        scan_ranks: Global ranks that participate in building a filter cache.
        filter_cache: Persist accepted physical row indexes between runs.
        filter_cache_dir: Shared POSIX directory for the cache. Required when
            reading from a non-local filesystem.
        filter_cache_timeout_sec: Maximum shared-filesystem coordination wait.
    """

    _SOURCE_ROWS = 0
    _PARQUET_BYTES = 1
    _FILTER_NANOSECONDS = 2

    def __init__(
        self,
        path: str | list,
        read_columns: list[str] | None = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        sampler: BaseSampler = None,
        filesystem: pafs.FileSystem | dict | None = None,
        shard: bool = True,
        drop_tail: bool = True,
        rotate_tail: bool = False,
        seed: int = 0,
        scan_workers: int = 4,
        scan_ranks: int | None = None,
        filter_cache: bool = True,
        filter_cache_dir: str | None = None,
        filter_cache_timeout_sec: float = 3600.0,
    ):
        if scan_workers <= 0:
            raise ValueError("scan_workers must be greater than zero")
        if scan_ranks is not None and scan_ranks <= 0:
            raise ValueError("scan_ranks must be greater than zero or None")
        if float(filter_cache_timeout_sec) <= 0:
            raise ValueError("filter_cache_timeout_sec must be greater than zero")

        self.shard = shard
        self.drop_tail = drop_tail
        self.rotate_tail = rotate_tail
        self.seed = seed
        self.scan_workers = scan_workers
        self.scan_ranks = scan_ranks
        self.filter_cache = filter_cache
        self.filter_cache_dir = filter_cache_dir
        self.filter_cache_timeout_sec = float(filter_cache_timeout_sec)

        # Sharded ownership is derived from a canonical file order; file
        # shuffling is applied to the already-owned segments during iteration.
        super().__init__(
            path=path,
            read_columns=read_columns,
            shuffle_files=shuffle_files if not shard else False,
            shuffle_pq=shuffle_pq,
            sampler=sampler,
            filesystem=filesystem,
        )
        self._shuffle_files = shuffle_files

        self._file_counts: np.ndarray | None = None
        self._planner: ShardPlanner | None = None
        self._scan_duration_sec = 0.0
        self._epoch_state = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._epoch_metric_counters: torch.Tensor | None = None
        self._filter_nanoseconds = 0
        self._source_rows = 0
        self._world_size, self._rank = resolve_dist_info()

        # A launcher variable with world_size 1 means the process group has not
        # come up yet — every rank would then plan for the whole corpus.
        launched_distributed = any(
            key in os.environ
            for key in ("LOCAL_RANK", "MASTER_ADDR", "TORCHELASTIC_RUN_ID")
        )
        if self.shard and self._world_size == 1 and launched_distributed:
            warnings.warn(
                "[shard] world_size resolved to 1 at dataset construction time, "
                "but this looks like a distributed launch. Initialize the process "
                "group or export WORLD_SIZE/RANK before building the dataset; "
                "otherwise every rank will read the full dataset.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # Subclass contract                                                   #
    # ------------------------------------------------------------------ #
    def count_valid_records(self, file: str) -> int:
        """Records of ``file`` that pass this dataset's filter."""
        raise NotImplementedError(
            "count_valid_records must be implemented by "
            "BaseShardedParquetDataset subclasses"
        )

    def iter_file_records(self, file: str) -> Iterator[dict[str, Any]]:
        """Yield the records of ``file`` that pass the filter, in physical order.

        Implementations should account for the physical rows they consume in
        ``self._source_rows`` and the time their predicate costs in
        ``self._filter_nanoseconds``; both are reset before each segment and
        surface through :meth:`get_epoch_metrics`.
        """
        raise NotImplementedError(
            "iter_file_records must be implemented by "
            "BaseShardedParquetDataset subclasses"
        )

    @contextmanager
    def _time_filter(self) -> Iterator[None]:
        """Attribute the enclosed work to the filter-time metric."""
        if self._epoch_metric_counters is None:
            yield
            return
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self._filter_nanoseconds += time.perf_counter_ns() - started

    # ------------------------------------------------------------------ #
    # Scan + plan                                                         #
    # ------------------------------------------------------------------ #
    def _scan_files(self) -> None:
        """Count valid records per file and build the shard planner.

        Subclasses call this at the end of ``__init__``, once everything the
        predicate depends on is set up.
        """
        if not self.shard:
            return

        scan_start = time.perf_counter()
        counts = self._collect_file_counts()
        self._file_counts = counts
        self._scan_duration_sec = time.perf_counter() - scan_start

        self._planner = ShardPlanner(
            file_counts=counts,
            world_size=self._world_size,
            rank=self._rank,
            drop_tail=self.drop_tail,
            rotate_tail=self.rotate_tail,
        )
        if self._rank == 0:
            warnings.warn(
                "\n" + self._planner.describe() + self._scan_suffix(), stacklevel=2
            )

    def _collect_file_counts(self) -> np.ndarray:
        """Threaded scan over every file. Override to use a cache instead."""
        counts = np.empty(len(self.files), dtype=np.int64)
        with ThreadPoolExecutor(max_workers=self.scan_workers) as executor:
            future_to_index = {
                executor.submit(self.count_valid_records, file): index
                for index, file in enumerate(self.files)
            }
            completed = tqdm(
                as_completed(future_to_index),
                total=len(self.files),
                desc="Scanning parquet files, preparing sharded dataset",
                disable=self._rank != 0,
            )
            for future in completed:
                counts[future_to_index[future]] = future.result()
        return counts

    def _scan_suffix(self) -> str:
        """Extra detail appended to the construction-time scan warning."""
        return ""

    # ------------------------------------------------------------------ #
    # Scan reporting                                                      #
    # ------------------------------------------------------------------ #
    @property
    def scan_duration_sec(self) -> float:
        """Wall time the construction-time scan took on this rank."""
        return self._scan_duration_sec

    @property
    def file_counts(self) -> np.ndarray | None:
        """Valid records per file, or ``None`` when not sharding."""
        return self._file_counts

    @property
    def uses_filter_cache(self) -> bool:
        """Whether the scan was served by a persistent filter cache."""
        return False

    @property
    def filter_cache_hit(self) -> bool:
        """Whether that cache was reused rather than rebuilt."""
        return False

    @property
    def scan_rank_count(self) -> int:
        """Ranks that participated in building the cache (0 on a hit)."""
        return 0

    @property
    def planner(self) -> ShardPlanner:
        """The shard planner built during construction."""
        if self._planner is None:
            raise RuntimeError(
                "Sharded file counts have not been initialized; "
                "_scan_files() must run during __init__"
            )
        return self._planner

    def _planner_for(self, world_size: int, rank: int) -> ShardPlanner:
        """Planner for the current environment.

        The process group may come up after construction (or a test may mock a
        different world), so re-derive the planner when the environment no
        longer matches the one captured in ``__init__``.
        """
        planner = self.planner
        if (planner.world_size, planner.rank) == (world_size, rank):
            return planner
        return ShardPlanner(
            file_counts=self._file_counts,
            world_size=world_size,
            rank=rank,
            drop_tail=self.drop_tail,
            rotate_tail=self.rotate_tail,
        )

    # ------------------------------------------------------------------ #
    # Epoch state                                                         #
    # ------------------------------------------------------------------ #
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for shard rotation and shuffling."""
        if epoch < 0:
            raise ValueError("epoch must be greater than or equal to zero")
        self._epoch_state[0] = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state[0].item())

    # ------------------------------------------------------------------ #
    # Length + iteration                                                  #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if not self.shard:
            return super().__len__()
        info = resolve_worker_info(self._world_size, self._rank)
        return self._planner_for(info.world_size, info.rank).rank_record_count()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.shard:
            return super().__iter__()

        iterator = self._iter_sharded_impl()
        # A sampler that already decided source membership must not run twice.
        if self.sampler is None or self.sampler is self._membership_filter():
            return iterator
        self.sampler.set_dataset_iterator(iterator)
        return iter(self.sampler)

    def _membership_filter(self):
        """The sampler, if any, whose decisions the scan already applied."""
        return None

    def _iter_sharded_impl(self) -> Iterator[dict[str, Any]]:
        info = resolve_worker_info(self._world_size, self._rank)
        epoch = self.epoch
        planner = self._planner_for(info.world_size, info.rank)
        segments = planner.worker_segments(
            worker_id=info.worker_id, num_workers=info.num_workers, epoch=epoch
        )
        rng = np.random.default_rng([self.seed, info.rank, info.worker_id, epoch])
        # Exposed so process() can make reproducible per-record random choices
        # (sequence slicing) that differ across workers, ranks and epochs.
        self._rng = rng
        if self._shuffle_files and len(segments) > 1:
            rng.shuffle(segments)

        for segment in segments:
            file = self.files[segment.file_index]
            records = self._read_segment(file, segment.start, segment.stop)
            if self.shuffle_pq and len(records) > 1:
                rng.shuffle(records)
            for record in records:
                processed_record = self.process(record)
                if processed_record is not None:
                    yield processed_record

    def _read_segment(self, file: str, start: int, stop: int) -> list[dict[str, Any]]:
        """Read the records of ``file`` this worker owns.

        Raises:
            RuntimeError: If the file yields fewer valid records than the scan
                promised. That means the scan and iteration predicates have
                diverged, and every rank's ownership is now wrong.
        """
        owned_records = []
        self._filter_nanoseconds = 0
        self._source_rows = 0

        valid_index = 0
        for record in self.iter_file_records(file):
            if start <= valid_index < stop:
                owned_records.append(record)
            valid_index += 1

        self._record_file_metrics(
            source_rows=self._source_rows,
            parquet_bytes=file_size(self.filesystem, file),
            filter_nanoseconds=self._filter_nanoseconds,
        )

        if valid_index < stop:
            raise RuntimeError(
                f"[shard] {file} yielded {valid_index} valid records during "
                f"iteration, but the scan counted at least {stop}; iteration "
                "filtering diverged from count_valid_records()."
            )
        return owned_records

    # ------------------------------------------------------------------ #
    # Optional per-epoch performance counters                             #
    # ------------------------------------------------------------------ #
    def configure_epoch_metrics(self, num_workers: int) -> None:
        """Allocate per-worker shared counters for optional benchmark metrics."""
        worker_slots = max(int(num_workers), 1)
        self._epoch_metric_counters = torch.zeros(
            (worker_slots, 3), dtype=torch.int64
        ).share_memory_()

    def reset_epoch_metrics(self) -> None:
        """Reset counters after the previous DataLoader iterator has completed."""
        if self._epoch_metric_counters is not None:
            self._epoch_metric_counters.zero_()

    def get_epoch_metrics(self) -> dict[str, float]:
        """Return counters accumulated by this rank's DataLoader workers."""
        if self._epoch_metric_counters is None:
            return {
                "source_rows_scanned": 0.0,
                "parquet_bytes_opened": 0.0,
                "filter_sec": 0.0,
            }
        totals = self._epoch_metric_counters.sum(dim=0).tolist()
        max_worker_filter_nanoseconds = (
            self._epoch_metric_counters[:, self._FILTER_NANOSECONDS].max().item()
        )
        return {
            "source_rows_scanned": float(totals[self._SOURCE_ROWS]),
            "parquet_bytes_opened": float(totals[self._PARQUET_BYTES]),
            "filter_sec": float(max_worker_filter_nanoseconds) / 1e9,
        }

    def _record_file_metrics(
        self, source_rows: int, parquet_bytes: int, filter_nanoseconds: int
    ) -> None:
        if self._epoch_metric_counters is None:
            return
        worker_info = torch.utils.data.get_worker_info()
        worker_slot = worker_info.id if worker_info is not None else 0
        if worker_slot >= len(self._epoch_metric_counters):
            raise RuntimeError(
                "Dataset epoch metrics were configured for fewer workers "
                "than the DataLoader uses"
            )
        counters = self._epoch_metric_counters[worker_slot]
        counters[self._SOURCE_ROWS] += source_rows
        counters[self._PARQUET_BYTES] += parquet_bytes
        counters[self._FILTER_NANOSECONDS] += filter_nanoseconds
