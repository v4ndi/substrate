"""Deprecated module path.

``avatar.data.dataset`` moved to :mod:`avatar.data.base`,
:mod:`avatar.data.tabular` and :mod:`avatar.data.sequential`. This shim keeps
existing Hydra ``_target_`` strings resolving for one release.

``ShardTabularDataset`` and `ShardEventSequenceDataset` are gone: the plain
``TabularDataset`` / ``EventSequenceDataset`` now shard by default, so the
separate classes had nothing left to add.
"""

import warnings

from avatar.data.base import BaseParquetDataset, BaseShardedParquetDataset
from avatar.data.sequential import EventSequenceDataset
from avatar.data.tabular import TabularDataset

warnings.warn(
    "avatar.data.dataset has moved: use avatar.data.tabular.TabularDataset, "
    "avatar.data.sequential.EventSequenceDataset, or the flat re-exports on "
    "avatar.data. This shim will be removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

# Both sharded variants are now the default behaviour of the base classes.
ShardTabularDataset = TabularDataset
ShardEventSequenceDataset = EventSequenceDataset

# The old two-level base (BaseIterDataset -> IterDataset) collapsed into one.
BaseIterDataset = BaseParquetDataset
IterDataset = BaseParquetDataset

__all__ = [
    "BaseIterDataset",
    "BaseParquetDataset",
    "BaseShardedParquetDataset",
    "EventSequenceDataset",
    "IterDataset",
    "ShardEventSequenceDataset",
    "ShardTabularDataset",
    "TabularDataset",
]
