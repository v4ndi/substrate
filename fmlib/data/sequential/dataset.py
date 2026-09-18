"""Sharded event-sequence dataset."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import torch

from fmlib.data.base.iterable import BaseShardedParquetDataset
from fmlib.data.base.parquet import read_parquet_columns, read_parquet_file
from fmlib.data.sampler import BaseSampler
from fmlib.data.tabular.dataset import TabularDataset

__all__ = ["EventSequenceDataset"]


def _as_list_array(column) -> pa.Array:
    """Return ``column`` as a single contiguous list array."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
        if isinstance(column, pa.ChunkedArray):
            column = (
                column.chunk(0)
                if column.num_chunks == 1
                else pa.concat_arrays(column.chunks)
            )
    return column


class EventSequenceDataset(BaseShardedParquetDataset):
    """Event sequences from parquet, sharded across ranks and workers.

    Each record holds one entity's event history as parallel lists. The dataset

    - drops sequences shorter than ``min_length``,
    - keeps only the event types in ``selected_event_ids`` when given,
    - slices sequences longer than ``max_length``, from a random offset if
      ``random_slicing``,
    - converts the surviving lists to tensors, and optionally processes tabular
      features carried alongside the sequence.

    Args:
        path: Directory (or list of directories) with parquet files; may be a
            local path or an ``hdfs://`` URI.
        sequence_columns: Columns holding the parallel event lists. The last one
            defines a record's sequence length.
        event_time_column: Column of event timestamps, emitted as ``_timestamps``.
        event_ids_column: Column of event-type identifiers, emitted as
            ``_event_ids``.
        selected_event_ids: Event types to keep. ``None`` keeps all of them.
        min_length: Sequences shorter than this — after event-type filtering —
            are dropped.
        max_length: Sequences longer than this are sliced.
        random_slicing: Slice long sequences at a random offset instead of
            keeping the most recent ``max_length`` events. The offset is drawn
            from a generator seeded per ``(rank, worker, epoch)``, so runs are
            reproducible and workers do not draw in lockstep.
        has_tabular: Records also carry ``cat_features`` / ``num_features``.
        read_columns: Columns to read. ``None`` reads all of them.
        shuffle_files: Shuffle the order owned file segments are read in.
        shuffle_pq: Shuffle rows within each file.
        lazy_process: Defer processing to the collate function.
        sampler: Record-stream sampler.
        filesystem: Explicit :class:`pyarrow.fs.FileSystem` or a mapping of
            options. ``None`` infers it from ``path``.

    Sharding and epoch behaviour are documented on
    :class:`~fmlib.data.base.iterable.BaseShardedParquetDataset`.
    """

    def __init__(
        self,
        path: str | list,
        sequence_columns: list[str],
        event_time_column: str | None = None,
        event_ids_column: str | None = None,
        selected_event_ids: list[int] | None = None,
        min_length: int = 1,
        max_length: int = 512,
        random_slicing: bool = False,
        has_tabular: bool = False,
        read_columns: list[str] | None = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        lazy_process: bool = False,
        sampler: BaseSampler = None,
        filesystem: pafs.FileSystem | dict | None = None,
        shard: bool = True,
        drop_tail: bool = True,
        rotate_tail: bool = False,
        seed: int = 0,
        scan_workers: int = 4,
    ):
        assert min_length >= 0, "min_length must be greater than or equal to 0"
        assert max_length >= min_length, (
            "max_length must be greater than or equal to min_length; otherwise "
            "slicing could drop a record the scan counted as valid"
        )
        assert event_time_column not in sequence_columns, (
            "event_time_column in the sequence_columns"
        )
        if event_ids_column is not None:
            assert event_ids_column not in sequence_columns, (
                "event_ids_column in the sequence_columns"
            )

        self.min_length = min_length
        self.max_length = max_length
        self.random_slicing = random_slicing
        self.sequence_columns = list(sequence_columns)
        self.has_tabular = has_tabular
        self.event_ids_column = event_ids_column
        self.event_time_column = event_time_column
        self.lazy_process = lazy_process

        self._filters_modalities = (
            selected_event_ids is not None and event_ids_column is not None
        )
        if selected_event_ids is not None:
            self.selected_event_ids = torch.tensor(selected_event_ids).long()
            self._selected_event_ids = np.asarray(selected_event_ids)
        else:
            self.selected_event_ids = None
            self._selected_event_ids = None

        if read_columns is not None:
            # The filter and the processing both need these regardless of what
            # the caller asked for.
            required = [*self.sequence_columns]
            if event_time_column is not None:
                required.append(event_time_column)
            if event_ids_column is not None:
                required.append(event_ids_column)
            read_columns = list(dict.fromkeys([*read_columns, *required]))

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
            filter_cache=False,
        )
        self._scan_files()

    # ------------------------------------------------------------------ #
    # Scan / iterate predicate — must stay in agreement                   #
    # ------------------------------------------------------------------ #
    @property
    def _length_column(self) -> str:
        """The column whose list length defines a record's sequence length."""
        return self.sequence_columns[-1]

    def count_valid_records(self, file: str) -> int:
        """Vectorised count of the records of ``file`` that pass the filter."""
        columns = [self._length_column]
        if self._filters_modalities:
            columns.append(self.event_ids_column)
        table = read_parquet_columns(file, columns, self.filesystem)
        if table.num_rows == 0:
            return 0
        return int(self._valid_mask(table).sum())

    def _valid_mask(self, table: pa.Table) -> np.ndarray:
        """Row mask matching :meth:`_accepts`, evaluated over a whole file."""
        lengths = pc.list_value_length(table[self._length_column])
        lengths = pc.fill_null(lengths, 0).to_numpy(zero_copy_only=False)
        mask = lengths >= self.min_length
        if not self._filters_modalities:
            return mask

        events = _as_list_array(table[self.event_ids_column])
        values = pc.list_flatten(events).to_numpy(zero_copy_only=False)
        parents = pc.list_parent_indices(events).to_numpy(zero_copy_only=False)
        selected = np.isin(values, self._selected_event_ids)
        selected_counts = np.bincount(
            parents[selected], minlength=table.num_rows
        ).astype(np.int64)
        return mask & (selected_counts >= self.min_length)

    def _accepts(self, record: dict[str, Any]) -> bool:
        """Row-wise equivalent of :meth:`_valid_mask`."""
        if len(record[self._length_column]) < self.min_length:
            return False
        if not self._filters_modalities:
            return True
        events = np.asarray(record[self.event_ids_column])
        selected = np.isin(events, self._selected_event_ids)
        return int(selected.sum()) >= self.min_length

    def iter_file_records(self, file: str) -> Iterator[dict[str, Any]]:
        """Yield the accepted records of ``file`` in physical row order."""
        for record in read_parquet_file(
            file=file,
            columns=self.read_columns,
            shuffle=False,
            filesystem=self.filesystem,
        ):
            self._source_rows += 1
            with self._time_filter():
                accepted = self._accepts(record)
            if accepted:
                yield record

    # ------------------------------------------------------------------ #
    # Record processing                                                   #
    # ------------------------------------------------------------------ #
    def _slice_bounds(self, sequence_length: int) -> tuple[int, int | None]:
        """Bounds used to trim a sequence down to ``max_length``.

        ``sequence_length`` is the length *after* event-type filtering, so the
        bounds always land inside the sequence that will actually be sliced.
        Deriving them from the raw length instead would let a random offset run
        past the end of a modality-filtered sequence and silently empty it.
        """
        if sequence_length > self.max_length and self.random_slicing:
            start = int(self.rng.integers(0, sequence_length - self.max_length + 1))
            return start, start + self.max_length
        return -self.max_length, None

    def _modalities_mask(self, record: dict) -> torch.Tensor | None:
        """Move event ids under ``_event_ids`` and build the event-type mask.

        Returns:
            The boolean mask over the raw event ids when event-type filtering is
            active, otherwise ``None``. The mask is applied to every sequence
            column before slicing.
        """
        if self.event_ids_column is None:
            return None

        event_data = record[self.event_ids_column]
        record["_event_ids"] = torch.from_numpy(np.array(event_data)).long()
        del record[self.event_ids_column]

        if not self._filters_modalities:
            return None
        return torch.isin(record["_event_ids"], self.selected_event_ids)

    @staticmethod
    def _prepare_record(record, start: int, end: int | None, modalities_mask=None):
        """Convert one record column to a tensor, then mask and slice it."""
        record = torch.from_numpy(np.array(record))
        if len(record.shape) == 0:
            return record
        if modalities_mask is not None:
            record = record[modalities_mask]
        return record[start:end]

    def _process_record(
        self, record: dict, start: int, end: int | None, modalities_mask
    ):
        """Convert every sequence, timestamp and tabular field to tensors."""
        partial_prepare = partial(
            self._prepare_record,
            start=start,
            end=end,
            modalities_mask=modalities_mask,
        )
        for column in self.sequence_columns:
            record[column] = partial_prepare(record[column])

            if record[column].dtype in (torch.int16, torch.int32, torch.int64):
                record[column] = record[column].long()
            elif record[column].dtype in (torch.float16, torch.float32, torch.float64):
                record[column] = record[column].float()

        if self.event_time_column is not None:
            # expected that self.event_time_column contains float values
            record["_timestamps"] = partial_prepare(
                record[self.event_time_column]
            ).float()
            del record[self.event_time_column]

        if "_event_ids" in record:
            events = record["_event_ids"]
            if modalities_mask is not None:
                events = events[modalities_mask]
            record["_event_ids"] = events[start:end].long()

        if self.has_tabular:
            TabularDataset.process_tabular(record)
        return record

    def process(self, record: dict):
        """Filter, slice and tensorise one record.

        Returns ``None`` only when the record fails the length filter, which
        can happen only with ``shard=False``; the sharded path filters in
        :meth:`iter_file_records` so that ownership is decided on the same set
        of records that iteration yields.
        """
        modalities_mask = self._modalities_mask(record)
        if modalities_mask is not None:
            sequence_length = int(modalities_mask.sum())
        else:
            sequence_length = len(record[self._length_column])
        if sequence_length < self.min_length:
            return None

        start, end = self._slice_bounds(sequence_length)
        process_record_func = partial(
            self._process_record,
            record=record,
            start=start,
            end=end,
            modalities_mask=modalities_mask,
        )
        if self.lazy_process:
            return process_record_func
        return process_record_func()
