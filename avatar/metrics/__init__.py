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
    "ClassificationInferenceMetrics",
    # Core metrics
    "MultiLossMetric",
    "Entropy",
    "Importance",
    # Specialized metrics
    "UpliftMetrics",
    "ResponseMetrics",
    "RegressionMetrics",
    # Campaign tools
    "CollectEmbeddings",
    "CatboostCampaignBenchmark",
    "MLPCampaignBenchmark",
    # Wrappers
    "GroupAverageMetricWrapper",
    "GroupDevidedMetricsWrapper",
    "SequenceStats",
    "ExpertsWorkload",
    "RocAucScore",
    "HiddensNorm",
    # log all losses from outputs
    "UniversalLossesMetric",
]
