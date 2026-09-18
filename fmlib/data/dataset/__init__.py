"""Deprecated module path.

``fmlib.data.dataset`` moved to :mod:`fmlib.data.base`,
:mod:`fmlib.data.tabular` and :mod:`fmlib.data.sequential`. This shim keeps
existing Hydra ``_target_`` strings resolving for one release.

``ShardTabularDataset`` and `ShardEventSequenceDataset` are gone: the plain
``TabularDataset`` / ``EventSequenceDataset`` now shard by default, so the
separate classes had nothing left to add.
"""

import warnings

from fmlib.data.base import BaseParquetDataset, BaseShardedParquetDataset
from fmlib.data.sequential import EventSequenceDataset
from fmlib.data.tabular import TabularDataset

warnings.warn(
    "fmlib.data.dataset has moved: use fmlib.data.tabular.TabularDataset, "
    "fmlib.data.sequential.EventSequenceDataset, or the flat re-exports on "
    "fmlib.data. This shim will be removed in the next release.",
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
