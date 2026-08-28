"""Parquet scanning helpers shared by the local preprocessing backend."""

from __future__ import annotations

import glob
import os
from typing import Iterator, Optional, Sequence, Union

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PathLike = Union[str, "os.PathLike[str]"]
Source = Union[PathLike, Sequence[PathLike], ds.Dataset]


def _resolve_files(source: Union[PathLike, Sequence[PathLike]]) -> list[str]:
    """Expand a path / glob / directory / list into a sorted list of parquet files."""

    def _one(path: str) -> list[str]:
        path = os.fspath(path)
        if os.path.isdir(path):
            found = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        elif any(ch in path for ch in "*?["):
            found = glob.glob(path, recursive=True)
        else:
            found = [path]
        return found

    if isinstance(source, (str, os.PathLike)):
        files = _one(source)
    else:
        files = []
        for item in source:
            files.extend(_one(item))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No parquet files found for source: {source!r}")
    return files


def as_dataset(source: Source, columns: Optional[Sequence[str]] = None) -> ds.Dataset:
    """Return a :class:`pyarrow.dataset.Dataset` for ``source``.

    ``source`` may be a path, a glob, a directory, a list of any of those, or an
    already-built :class:`pyarrow.dataset.Dataset`. ``columns`` is accepted for
    symmetry but projection happens in :func:`iter_record_batches`.
    """
    if isinstance(source, ds.Dataset):
        return source
    files = _resolve_files(source)
    return ds.dataset(files, format="parquet", partitioning="hive")


def dataset_num_rows(source: Source) -> int:
    """Total number of rows across ``source`` (reads only parquet footers)."""
    if isinstance(source, ds.Dataset):
        return source.count_rows()
    return sum(pq.ParquetFile(f).metadata.num_rows for f in _resolve_files(source))


def iter_record_batches(
    source: Source,
    columns: Optional[Sequence[str]] = None,
    batch_rows: int = 250_000,
) -> Iterator[pa.RecordBatch]:
    """Yield :class:`pyarrow.RecordBatch` chunks of at most ``batch_rows`` rows.

    Memory is bounded by ``batch_rows * len(columns)``. ``columns=None`` reads
    every column (used by ``transform`` to pass non-feature columns through).
    Missing requested columns raise a clear error before iteration starts.
    """
    dataset = as_dataset(source)
    if columns is not None:
        columns = list(dict.fromkeys(columns))  # de-dup, keep order
        missing = [c for c in columns if c not in dataset.schema.names]
        if missing:
            raise KeyError(
                f"Columns {missing} not present in dataset schema "
                f"{dataset.schema.names}"
            )
    scanner = dataset.scanner(columns=columns, batch_size=batch_rows)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        yield batch
