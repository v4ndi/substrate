"""Parquet source discovery and Hive partition reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import polars as pl

from avatar.automl.exceptions import SchemaError
from avatar.automl.types import ParquetPath


@dataclass(frozen=True)
class ParquetSource:
    """Represent a deterministic collection of parquet shards.

    Attributes:
        path: Resolved input file or directory.
        files: Recursively discovered parquet files in deterministic order.
    """

    path: Path
    files: tuple[Path, ...]

    @classmethod
    def resolve(cls, path: ParquetPath) -> ParquetSource:
        """Resolve a parquet file or recursively discover a directory.

        Args:
            path: Parquet file or directory containing parquet shards.

        Returns:
            A source with absolute paths sorted deterministically.

        Raises:
            SchemaError: If the path is invalid or contains no parquet files.
        """
        resolved = Path(path).expanduser().resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".parquet":
            files = (resolved,)
        elif resolved.is_dir():
            files = tuple(
                sorted(item for item in resolved.rglob("*.parquet") if item.is_file())
            )
        else:
            msg = f"Parquet path does not exist or is not a parquet file/directory: {resolved}"
            raise SchemaError(msg)
        if not files:
            msg = f"No parquet files found under: {resolved}"
            raise SchemaError(msg)
        return cls(path=resolved, files=files)

    def scan(self) -> pl.LazyFrame:
        """Scan all files and restore ``key=value`` Hive partitions as columns.

        Partition segments are collected from the complete file path, so fields
        located both above and below the path passed to :meth:`resolve` remain
        available in the returned frame.

        Returns:
            A lazy union of shards and reconstructed partition columns.

        Raises:
            SchemaError: If parquet reading or frame concatenation fails.
        """
        try:
            partition_groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
            for path in self.files:
                partitions = tuple(
                    (name, unquote(value))
                    for part in path.parent.parts
                    if "=" in part
                    for name, value in [part.split("=", 1)]
                    if name and value
                )
                partition_groups.setdefault(partitions, []).append(str(path))

            frames: list[pl.LazyFrame] = []
            for partitions, files in partition_groups.items():
                frame = pl.scan_parquet(files, hive_partitioning=False)
                names = frame.collect_schema().names()
                missing_partitions = {
                    name: value for name, value in partitions if name not in names
                }
                if missing_partitions:
                    frame = frame.with_columns([
                        pl.lit(value).alias(name)
                        for name, value in missing_partitions.items()
                    ])
                frames.append(frame)
            return (
                frames[0]
                if len(frames) == 1
                else pl.concat(frames, how="diagonal_relaxed")
            )
        except Exception as exc:
            msg = f"Failed to read parquet source {self.path}: {exc}"
            raise SchemaError(msg) from exc

    def read(self, columns: tuple[str, ...] | None = None) -> pl.DataFrame:
        """Collect a projected lazy source without retaining eager shard frames."""
        try:
            frame = self.scan()
            if columns is not None:
                available = frame.collect_schema().names()
                frame = frame.select([name for name in columns if name in available])
            return frame.collect(engine="streaming")
        except SchemaError:
            raise
        except Exception as exc:
            msg = f"Failed to read parquet source {self.path}: {exc}"
            raise SchemaError(msg) from exc

    def unique_column_values(self, column: str) -> tuple[tuple[Any, ...], bool]:
        """Read one physical or Hive-partition column without loading other data.

        Returns:
            Deterministically ordered non-null values and whether nulls exist.

        Raises:
            SchemaError: If the column is absent from every shard or cannot be read.
        """
        values: list[Any] = []
        has_nulls = False
        column_seen = False
        try:
            for path in self.files:
                schema = pl.read_parquet_schema(path)
                if column in schema:
                    series = pl.read_parquet(path, columns=[column])[column]
                    column_seen = True
                    has_nulls = has_nulls or bool(series.null_count())
                    for value in series.drop_nulls().unique().to_list():
                        if value not in values:
                            values.append(value)
                    continue

                partitions = {
                    name: unquote(value)
                    for part in path.parent.parts
                    if "=" in part
                    for name, value in [part.split("=", 1)]
                    if name and value
                }
                if column in partitions:
                    column_seen = True
                    value = partitions[column]
                    if value not in values:
                        values.append(value)
                else:
                    has_nulls = True
            if not column_seen:
                msg = f"Column {column!r} is missing from parquet source {self.path}"
                raise SchemaError(msg)
            return tuple(sorted(values, key=str)), has_nulls
        except SchemaError:
            raise
        except Exception as exc:
            msg = f"Failed to read column {column!r} from parquet source {self.path}: {exc}"
            raise SchemaError(msg) from exc
