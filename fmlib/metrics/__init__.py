"""Metrics: everything scored during evaluation, plus the wrappers that compose them.

All of them implement the ``update`` / ``compute`` / ``reset`` contract of
:class:`~fmlib.metrics.base.BaseMetric` and are referenced from the
``metrics:`` block of a config. See ``docs/guides/metrics.md`` for the contract
and ``docs/reference/metrics.md`` for the catalogue.

Two kinds, told apart by their base class rather than by a convention:

* :class:`~fmlib.metrics.base.ScalarMetric` returns numbers —
  ``UpliftMetrics``, ``ResponseMetrics``, ``RegressionMetrics``,
  ``MultiClassMetrics``, ``MultiLossMetric``, ``UniversalLossesMetric`` and the
  two wrappers;
* :class:`~fmlib.metrics.base.ArtifactMetric` returns a file —
  ``CollectEmbeddings``, ``InferenceMultiTaskCampaignMetrics``,
  ``InferenceSupervisedMetrics``.
"""

from fmlib.metrics.loss_logging import UniversalLossesMetric

from .base import ArtifactMetric, BaseMetric, ScalarMetric
from .campaign import CollectEmbeddings, InferenceMultiTaskCampaignMetrics
from .multi_loss import MultiLossMetric
from .supervised import (
    InferenceSupervisedMetrics,
    MultiClassMetrics,
    RegressionMetrics,
    ResponseMetrics,
)
from .uplift import UpliftMetrics
from .utils import GroupAverageMetricWrapper, GroupDevidedMetricsWrapper

__all__ = [
    # Base classes
    "ArtifactMetric",
    "BaseMetric",
    # Collectors — the product is a file
    "CollectEmbeddings",
    # Wrappers
    "GroupAverageMetricWrapper",
    "GroupDevidedMetricsWrapper",
    "InferenceMultiTaskCampaignMetrics",
    "InferenceSupervisedMetrics",
    # Loss reporting
    "MultiClassMetrics",
    "MultiLossMetric",
    "RegressionMetrics",
    # Scoring metrics — the product is numbers
    "ResponseMetrics",
    "ScalarMetric",
    "UniversalLossesMetric",
    "UpliftMetrics",
]
