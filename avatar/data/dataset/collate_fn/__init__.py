"""Deprecated module path.

``avatar.data.dataset.collate_fn`` moved to :mod:`avatar.data.tabular.collate`
and :mod:`avatar.data.sequential.collate`. This shim keeps existing Hydra
``_target_`` strings resolving for one release.
"""

import warnings

from avatar.data.base.collate import BaseCollateFn
from avatar.data.sequential.collate import EventSequenceCollateFn
from avatar.data.tabular.collate import (
    MultiTaskSupervisedCollateFn,
    MultiTaskUpliftCollateFn,
    SupervisedCollateFn,
    TabularCollateFn,
    UpliftCollateFn,
)

warnings.warn(
    "avatar.data.dataset.collate_fn has moved: use avatar.data.tabular.collate, "
    "avatar.data.sequential.collate, or the flat re-exports on avatar.data. "
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
