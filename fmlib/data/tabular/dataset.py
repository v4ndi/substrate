"""Sharded tabular dataset."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from functools import partial
from typing import Any

import numpy as np
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch

from fmlib.data.base.filter_cache import PersistentFilterCache
from fmlib.data.base.iterable import BaseShardedParquetDataset
from fmlib.data.base.parquet import parquet_num_rows, read_parquet_file
from fmlib.data.sampler import BaseSampler, MultiTaskColumnsFilterSampler

__all__ = ["TabularDataset"]


class TabularDataset(BaseShardedParquetDataset):
    """Tabular dataset over parquet files, sharded across ranks and workers.

    Records carry ``cat_features`` / ``num_features`` arrays and optional
    pre-computed hidden states; :meth:`process_tabular` turns them into tensors
    under a single ``tab_features`` key that
    :class:`~fmlib.data.tabular.collate.TabularCollateFn` stacks into a
    :class:`~fmlib.data.tabular.batch.TabularBatch`.

    Args:
        path: Directory (or list of directories) with parquet files; may be a
            local path or an ``hdfs://`` URI. Do not pass a single file.
        read_columns: Columns to read. ``None`` reads all of them.
        shuffle_files: Shuffle the order owned file segments are read in.
        shuffle_pq: Shuffle rows within each file.
        hidden_state_column: Single hidden-state column in the parquet schema.
        hidden_state_columns: Several hidden-state columns. Mutually exclusive
            with ``hidden_state_column``.
        lazy_process: Defer :meth:`process_tabular` to the collate function.
            Faster for multi-GPU training; requires ``drop_last=True`` on the
            DataLoader.
        sampler: Record-stream sampler. A
            :class:`~fmlib.data.sampler.MultiTaskColumnsFilterSampler` is
            special-cased: its decisions define which records exist at all, so
            the scan applies it once and the sampler does not run again.
        filesystem: Explicit :class:`pyarrow.fs.FileSystem` or a mapping of
            options. ``None`` infers it from ``path``.

    Sharding, caching and epoch behaviour are documented on
    :class:`~fmlib.data.base.iterable.BaseShardedParquetDataset`.
    """

    def __init__(
        self,
        path: str | list,
        read_columns: list[str] | None = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        hidden_state_column: str | None = None,
        hidden_state_columns: str | list[str] | None = None,
        lazy_process: bool = False,
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
        shard_by_rank: bool | None = None,
    ):
        assert not (
            hidden_state_column is not None and hidden_state_columns is not None
        ), "Only one of hidden_state_column and hidden_state_columns can be specified"

        if shard_by_rank is not None:
            warnings.warn(
                "shard_by_rank is deprecated; use shard instead",
                DeprecationWarning,
                stacklevel=2,
            )
            shard = shard_by_rank

        super().__init__(
            path=path,
            read_columns=read_columns,
            shuffle_files=shuffle_files,
            shuffle_pq=shuffle_pq,
            sampler=sampler,
            filesystem=filesystem,
            shard=shard,
            drop_tail=drop_tail,
            rotate_tail=rotate_tail,
            seed=seed,
            scan_workers=scan_workers,
            scan_ranks=scan_ranks,
            filter_cache=filter_cache,
            filter_cache_dir=filter_cache_dir,
            filter_cache_timeout_sec=filter_cache_timeout_sec,
        )

        if hidden_state_column is not None:
            hidden_state_columns = [hidden_state_column]
        elif hidden_state_columns is not None and isinstance(hidden_state_columns, str):
            hidden_state_columns = [hidden_state_columns]
        self.hidden_state_columns = hidden_state_columns
        self.lazy_process = lazy_process

        assert self.check_tabular_features(), (
            "Not found tabular features in parquet schema or hidden_state_column."
        )

        # A MultiTaskColumnsFilterSampler decides membership, so the scan has to
        # apply it too — otherwise the counts sharding is planned from would not
        # match what iteration yields.
        self._source_filter = (
            sampler if isinstance(sampler, MultiTaskColumnsFilterSampler) else None
        )
        self._cache: PersistentFilterCache | None = None
        if self.shard and self._source_filter is not None and filter_cache:
            self._cache = PersistentFilterCache(
                files=self.files,
                source_filter=self._source_filter,
                filesystem=self.filesystem,
                cache_dir=filter_cache_dir,
                scan_workers=scan_workers,
                scan_ranks=scan_ranks,
                timeout_sec=filter_cache_timeout_sec,
            )

        # Columns the filter needs but the caller did not ask for. Only relevant
        # without a cache; a cached index already encodes the filter's verdict.
        self._source_only_columns: set[str] = set()
        self._source_read_columns = self.read_columns
        if (
            self._source_filter is not None
            and self._cache is None
            and self.read_columns is not None
        ):
            self._source_only_columns = self._source_filter.filter_columns - set(
                self.read_columns
            )
            self._source_read_columns = [
                *self.read_columns,
                *sorted(self._source_only_columns),
            ]

        self._scan_files()

    # ------------------------------------------------------------------ #
    # Schema                                                              #
    # ------------------------------------------------------------------ #
    def check_tabular_features(self) -> bool:
        """Check parquet schema for presence of tabular features."""
        if self.filesystem is None:
            schema = pq.read_schema(self.files[0])
        else:
            with self.filesystem.open_input_file(self.files[0]) as source:
                schema = pq.read_schema(source)
        return (
            "cat_features" in schema.names
            or "num_features" in schema.names
            or all(
                hidden_state_col in schema.names
                for hidden_state_col in self.hidden_state_columns or []
            )
        )

    # ------------------------------------------------------------------ #
    # Scan / iterate predicate — must stay in agreement                   #
    # ------------------------------------------------------------------ #
    def _membership_filter(self):
        return self._source_filter

    def _collect_file_counts(self) -> np.ndarray:
        if self._cache is not None:
            return self._cache.prepare(self._world_size, self._rank)
        return super()._collect_file_counts()

    def _scan_suffix(self) -> str:
        if self._cache is None:
            return ""
        if self._cache.hit:
            return ", filter cache hit"
        return (
            f", filter cache built using {self._cache.effective_scan_ranks}/"
            f"{self._world_size} scan ranks"
        )

    @property
    def uses_filter_cache(self) -> bool:
        return self._cache is not None

    @property
    def filter_cache_hit(self) -> bool:
        return self._cache is not None and self._cache.hit

    @property
    def scan_rank_count(self) -> int:
        return self._cache.effective_scan_ranks if self._cache is not None else 0

    def count_valid_records(self, file: str) -> int:
        """Rows ``file`` contributes once the source filter has been applied."""
        if self._cache is not None:
            return self._cache.count_valid_records(file)
        if self._source_filter is None:
            return parquet_num_rows(file, self.filesystem)

        filter_columns = sorted(self._source_filter.filter_columns)
        return sum(
            self._source_filter.accepts(record)
            for record in read_parquet_file(
                file=file,
                columns=filter_columns,
                shuffle=False,
                filesystem=self.filesystem,
            )
        )

    def iter_file_records(self, file: str) -> Iterator[dict[str, Any]]:
        """Yield the accepted records of ``file`` in physical row order."""
        if self._cache is not None:
            yield from self._iter_cached(file)
            return

        for record in read_parquet_file(
            file=file,
            columns=self._source_read_columns,
            shuffle=False,
            filesystem=self.filesystem,
        ):
            self._source_rows += 1
            if self._source_filter is not None:
                with self._time_filter():
                    accepted = self._source_filter.accepts(record)
                if not accepted:
                    continue
                for column in self._source_only_columns:
                    del record[column]
            yield record

    def _iter_cached(self, file: str) -> Iterator[dict[str, Any]]:
        """Select accepted rows by their cached physical indexes."""
        valid_indices = self._cache.load_indexes(file)
        next_owned = 0
        for physical_index, record in enumerate(
            read_parquet_file(
                file=file,
                columns=self._source_read_columns,
                shuffle=False,
                filesystem=self.filesystem,
            )
        ):
            self._source_rows += 1
            if (
                next_owned < len(valid_indices)
                and physical_index == valid_indices[next_owned]
            ):
                next_owned += 1
                yield record
        if next_owned != len(valid_indices):
            raise RuntimeError(
                f"[shard] {file} ended before cached physical row "
                f"{valid_indices[next_owned]} could be read"
            )

    # ------------------------------------------------------------------ #
    # Record processing                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def process_tabular(record, hidden_state_columns: list[str] | None = None):
        """Process tabular features to torch.tensor."""
        tabular_features = {
            "cat_features": None,
            "num_features": None,
        }

        if "cat_features" in record:
            tabular_features["cat_features"] = torch.from_numpy(
                np.array(record["cat_features"])
            ).long()
            del record["cat_features"]

        if "num_features" in record:
            tabular_features["num_features"] = torch.from_numpy(
                np.array(record["num_features"])
            ).float()
            del record["num_features"]

        if hidden_state_columns is not None:
            tabular_features["_hidden_states"] = {}
            for hidden_state_col in hidden_state_columns:
                tabular_features["_hidden_states"][hidden_state_col] = torch.from_numpy(
                    np.array(record[hidden_state_col])
                ).float()
                del record[hidden_state_col]

        record["tab_features"] = tabular_features

        return record

    def process(self, record: dict):
        if not self.lazy_process:
            return TabularDataset.process_tabular(
                record=record, hidden_state_columns=self.hidden_state_columns
            )
        return partial(
            TabularDataset.process_tabular,
            record=record,
            hidden_state_columns=self.hidden_state_columns,
        )
