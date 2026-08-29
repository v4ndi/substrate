from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parquet_num_rows(path: str) -> int:
    """Returns the number of rows in a parquet file or dataset.

    Args:
        path: Path to the parquet file or directory containing parquet files.
              Can be either a string or Path object.

    Returns:
        Number of rows in the parquet file(s). For directories, returns total
        row count across all files in the dataset.
    """
    with pq.ParquetFile(path) as f:
        return f.metadata.num_rows


def read_parquet_file(
    file: str, columns: list[str] | None = None, shuffle: bool = True
) -> Iterator[dict[str, Any]]:
    """Reads a parquet file and yields records as dictionaries with optional shuffling.

    Args:
        file: Path to the parquet file to read
        columns: Optional list of column names to read. If None, reads all columns.
        shuffle: Whether to randomly shuffle the rows before yielding. Defaults to True.

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
    # table = pq.read_table(file, use_threads=False, columns=columns)
    table = ds.dataset(file, format="parquet", partitioning="hive").to_table(
        use_threads=False, columns=columns
    )
    if columns is None:
        columns = [col._name for col in table.columns]
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
