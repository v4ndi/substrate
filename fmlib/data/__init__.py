"""Datasets, batches and collate functions.

Organised by modality: :mod:`fmlib.data.tabular` and
:mod:`fmlib.data.sequential` build on the shared machinery in
:mod:`fmlib.data.base`, which owns parquet access, filesystem resolution and
record-level sharding across ranks and DataLoader workers.

The whole public API is re-exported here, so ``from fmlib.data import
TabularDataset`` and ``_target_: fmlib.data.TabularDataset`` both work.
"""

from fmlib.data import base, campaign, sampler, sequential, tabular
from fmlib.data.base import (
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
from fmlib.data.campaign import CampaignTaskChannelBatches
from fmlib.data.sequential import (
    EventSequenceBatch,
    EventSequenceCollateFn,
    EventSequenceDataset,
)
from fmlib.data.tabular import (
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
    "CampaignTaskChannelBatches",
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
    "campaign",
    "move_to_device",
    "parquet_num_rows",
    "read_parquet_file",
    "resolve_filesystem",
    "sampler",
    "sequential",
    "tabular",
]
