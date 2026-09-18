"""Deprecated module path.

``fmlib.data.dataset.collate_fn`` moved to :mod:`fmlib.data.tabular.collate`
and :mod:`fmlib.data.sequential.collate`. This shim keeps existing Hydra
``_target_`` strings resolving for one release.
"""

import warnings

from fmlib.data.base.collate import BaseCollateFn
from fmlib.data.sequential.collate import EventSequenceCollateFn
from fmlib.data.tabular.collate import (
    MultiTaskSupervisedCollateFn,
    MultiTaskUpliftCollateFn,
    SupervisedCollateFn,
    TabularCollateFn,
    UpliftCollateFn,
)

warnings.warn(
    "fmlib.data.dataset.collate_fn has moved: use fmlib.data.tabular.collate, "
    "fmlib.data.sequential.collate, or the flat re-exports on fmlib.data. "
    "This shim will be removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BaseCollateFn",
    "EventSequenceCollateFn",
    "MultiTaskSupervisedCollateFn",
    "MultiTaskUpliftCollateFn",
    "SupervisedCollateFn",
    "TabularCollateFn",
    "UpliftCollateFn",
]
