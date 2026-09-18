"""Modality-independent dataset machinery: sharding, parquet IO, filesystems."""

from fmlib.data.base.batch import move_to_device
from fmlib.data.base.collate import BaseCollateFn
from fmlib.data.base.distributed import (
    WorkerInfo,
    resolve_dist_info,
    resolve_worker_info,
)
from fmlib.data.base.filter_cache import PersistentFilterCache
from fmlib.data.base.fs import (
    discover_parquet_files,
    is_local_filesystem,
    resolve_filesystem,
    resolve_filesystems,
)
from fmlib.data.base.iterable import BaseParquetDataset, BaseShardedParquetDataset
from fmlib.data.base.parquet import (
    parquet_num_rows,
    read_parquet_columns,
    read_parquet_file,
)
from fmlib.data.base.shard import ShardPlanner, ShardSegment

__all__ = [
    "BaseCollateFn",
    "BaseParquetDataset",
    "BaseShardedParquetDataset",
    "PersistentFilterCache",
    "ShardPlanner",
    "ShardSegment",
    "WorkerInfo",
    "discover_parquet_files",
    "is_local_filesystem",
    "move_to_device",
    "parquet_num_rows",
    "read_parquet_columns",
    "read_parquet_file",
    "resolve_dist_info",
    "resolve_filesystem",
    "resolve_filesystems",
    "resolve_worker_info",
]
