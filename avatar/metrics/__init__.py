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

# Optional Horizon metrics with graceful fallback
try:
    from .horizon_metric import (
        HorizonInference,
        HorizonMetric,
        KeyEventsInference,
        compute_f_score,
    )

    _HAS_HORIZON = True
except ImportError:
    _HAS_HORIZON = False
    import warnings

    warnings.warn(
        "HoTPP benchmark not installed. Horizon metrics will be unavailable. ",
        category=ImportWarning,
        stacklevel=2,
    )
    # Create null objects to prevent import errors
    HorizonInference = None
    HorizonMetric = None
    KeyEventsInference = None
    compute_f_score = None

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
    # Optional Horizon metrics (None if not available)
    "HorizonMetric",
    "HorizonInference",
    "KeyEventsInference",
    "compute_f_score",
    "SequenceStats",
    "ExpertsWorkload",
    "RocAucScore",
    "HiddensNorm",
    # log all losses from outputs
    "UniversalLossesMetric",
]
