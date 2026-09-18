"""Pipelines over tabular batches — one row per record.

Two of them, and the second is the first plus a treatment feature:

* :class:`~fmlib.pipeline.tabular.supervised.SupervisedLearner` — binary
  classification, regression and multi-class, told apart by ``num_classes``
  and ``task_type``;
* :class:`~fmlib.pipeline.tabular.uplift.SLearner` — the same model with the
  treatment flag fed in as a feature, scored on the difference between a
  treated and a control pass.

:mod:`~fmlib.pipeline.tabular.interaction` holds the blocks that decide how
an extra token — treatment or group — meets the feature tokens.
"""

from .interaction import (
    BaseTreatmentInteraction,
    ConcatTreatmentInteraction,
    ElementwiseTreatmentInteraction,
    IgnoreTreatmentInteraction,
    SumTreatmentInteraction,
)
from .supervised import SupervisedLearner
from .uplift import SLearner

__all__ = [
    "BaseTreatmentInteraction",
    "ConcatTreatmentInteraction",
    "ElementwiseTreatmentInteraction",
    "IgnoreTreatmentInteraction",
    "SLearner",
    "SumTreatmentInteraction",
    "SupervisedLearner",
]
