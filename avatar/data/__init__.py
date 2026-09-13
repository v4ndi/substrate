"""Datasets, batches and collate functions.

Organised by modality: :mod:`avatar.data.tabular` and
:mod:`avatar.data.sequential` build on the shared machinery in
:mod:`avatar.data.base`, which owns parquet access, filesystem resolution and
record-level sharding across ranks and DataLoader workers.

The whole public API is re-exported here, so ``from avatar.data import
TabularDataset`` and ``_target_: avatar.data.TabularDataset`` both work.
"""

from avatar.data import base, sampler, sequential, tabular
from avatar.data.base import (
    BaseCollateFn,
    BaseParquetDataset,
    BaseShardedParquetDataset,
    ShardPlanner,
    ShardSegment,
    move_to_device,
    parquet_num_rows,
    read_parquet_file,
    resolve_filesystem,
)
from avatar.data.sequential import (
    EventSequenceBatch,
    EventSequenceCollateFn,
    EventSequenceDataset,
)
from avatar.data.tabular import (
    MultiTaskSupervisedCollateFn,
    MultiTaskUpliftCollateFn,
    SupervisedCollateFn,
    TabularBatch,
    TabularCollateFn,
    TabularDataset,
    UpliftCollateFn,
    UpliftTabularBatch,
)

__all__ = [
    "BaseCollateFn",
    "BaseParquetDataset",
    "BaseShardedParquetDataset",
    "EventSequenceBatch",
    "EventSequenceCollateFn",
    "EventSequenceDataset",
    "MultiTaskSupervisedCollateFn",
    "MultiTaskUpliftCollateFn",
    "ShardPlanner",
    "ShardSegment",
    "SupervisedCollateFn",
    "TabularBatch",
    "TabularCollateFn",
    "TabularDataset",
    "UpliftCollateFn",
    "UpliftTabularBatch",
    "base",
    "move_to_device",
    "parquet_num_rows",
    "read_parquet_file",
    "resolve_filesystem",
    "sampler",
    "sequential",
    "tabular",
]
