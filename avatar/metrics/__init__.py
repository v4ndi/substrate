"""Metrics: everything scored during evaluation, plus the wrappers that compose them.

All of them implement the ``update`` / ``compute`` / ``reset`` contract of
:class:`~avatar.metrics.base.BaseMetric` and are referenced from the
``metrics:`` block of a config. See ``docs/guides/metrics.md`` for the contract
and ``docs/reference/metrics.md`` for the catalogue.

Three groups worth telling apart:

* **scoring metrics** — return numbers (``RocAucScore``, ``UpliftMetrics``,
  ``ResponseMetrics``, ``RegressionMetrics``);
* **collectors** — return files (``ClassificationInferenceMetrics``,
  ``CollectEmbeddings``, the campaign ``Inference*`` classes);
* **wrappers and diagnostics** — reshape or explain the above
  (``GroupAverageMetricWrapper``, ``GroupDevidedMetricsWrapper``,
  ``MultiLossMetric``, ``SequenceStats``, ``Entropy``, ``Importance``).
"""

from avatar.metrics.loss_logging import UniversalLossesMetric

from .base import BaseMetric, ClassificationInferenceMetrics
from .campaign import CatboostCampaignBenchmark, CollectEmbeddings, MLPCampaignBenchmark
from .classification import RocAucScore
from .moe_reg import Entropy, Importance
from .multi_loss import MultiLossMetric
from .sequence_stats import ExpertsWorkload, HiddensNorm, SequenceStats
from .supervised import RegressionMetrics, ResponseMetrics
from .uplift import UpliftMetrics
from .utils import GroupAverageMetricWrapper, GroupDevidedMetricsWrapper

__all__ = [
    # Base classes
    "BaseMetric",
    "CatboostCampaignBenchmark",
    "ClassificationInferenceMetrics",
    # Campaign tools
    "CollectEmbeddings",
    "Entropy",
    "ExpertsWorkload",
    # Wrappers
    "GroupAverageMetricWrapper",
    "GroupDevidedMetricsWrapper",
    "HiddensNorm",
    "Importance",
    "MLPCampaignBenchmark",
    # Core metrics
    "MultiLossMetric",
    "RegressionMetrics",
    "ResponseMetrics",
    "RocAucScore",
    "SequenceStats",
    # log all losses from outputs
    "UniversalLossesMetric",
    # Specialized metrics
    "UpliftMetrics",
]
