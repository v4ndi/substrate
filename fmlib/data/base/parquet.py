"""Filesystem-aware parquet reading primitives.

Every function takes an optional ``filesystem``; when it is ``None`` pyarrow
resolves the path itself, which preserves the original local-only behaviour of
these helpers exactly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

__all__ = [
    "open_parquet_file",
    "parquet_num_rows",
    "read_parquet_columns",
    "read_parquet_file",
]


@contextmanager
def open_parquet_file(
    path: str, filesystem: pafs.FileSystem | None = None
) -> Iterator[pq.ParquetFile]:
    """Open ``path`` as a :class:`pyarrow.parquet.ParquetFile`."""
    if filesystem is None:
        with pq.ParquetFile(path) as parquet_file:
            yield parquet_file
        return
    with filesystem.open_input_file(path) as source:
        with pq.ParquetFile(source) as parquet_file:
            yield parquet_file


def parquet_num_rows(path: str, filesystem: pafs.FileSystem | None = None) -> int:
    """Returns the number of rows in a parquet file.

    Args:
        path: Path to the parquet file.
        filesystem: Filesystem the path lives on. ``None`` means pyarrow
            resolves the path itself (local disk).

    Returns:
        Number of rows recorded in the file's footer metadata.
    """
    with open_parquet_file(path, filesystem) as parquet_file:
        return parquet_file.metadata.num_rows


def read_parquet_columns(
    path: str,
    columns: list[str],
    filesystem: pafs.FileSystem | None = None,
) -> pa.Table:
    """Read a column subset of one parquet file into a table.

    Used by the scan phase, which only ever needs the columns a filter or a
    length check reads — never the payload.
    """
    return ds.dataset(
        path, filesystem=filesystem, format="parquet", partitioning="hive"
    ).to_table(use_threads=False, columns=columns)


def read_parquet_file(
    file: str,
    columns: list[str] | None = None,
    shuffle: bool = True,
    filesystem: pafs.FileSystem | None = None,
) -> Iterator[dict[str, Any]]:
    """Reads a parquet file and yields records as dictionaries with optional shuffling.

    Args:
        file: Path to the parquet file to read
        columns: Optional list of column names to read. If None, reads all columns.
        shuffle: Whether to randomly shuffle the rows before yielding. Defaults to True.
        filesystem: Filesystem the path lives on. ``None`` means pyarrow
            resolves the path itself (local disk).

    Yields:
        Dictionary for each row, where keys are column names and values are numpy arrays
        containing the row values. Note that scalar values will still be returned as
        numpy arrays (use .item() to convert to Python scalar if needed).

    Note:
        - When shuffle=True, the entire file is loaded into memory temporarily
        - For very large files, consider using shuffle=False or specifying columns
        - Numpy arrays are returned even for scalar values for consistency

    Example:
        >>> # Read specific columns with shuffling
        >>> for record in read_parquet_file("data.parquet", columns=["id", "value"]):
        ...     print(record["id"].item(), record["value"].item())
        >>>
        >>> # Read all columns without shuffling
        >>> for record in read_parquet_file("data.parquet", shuffle=False):
        ...     print(record)
    """
    table = ds.dataset(
        file, filesystem=filesystem, format="parquet", partitioning="hive"
    ).to_table(use_threads=False, columns=columns)
    if columns is None:
        columns = table.column_names
    if shuffle and table.num_rows != 0:
        row_indexes = [i for i in range(table.num_rows)]
        np.random.shuffle(row_indexes)
        table = table.take(row_indexes)

    for rb in table.to_batches():
        col_arrays = [rb.column(x) for x in columns]
        col_arrays = [x.to_numpy(zero_copy_only=False) for x in col_arrays]
        for row in zip(*col_arrays, strict=False):
            record = {}
            for col, arr in zip(columns, row, strict=False):
                record[col] = arr
            yield record
