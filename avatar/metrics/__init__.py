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
