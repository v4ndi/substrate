"""Public task and configuration classes for fmlib AutoML."""

from .config import (
    BinaryTaskConfig,
    EnvironmentConfig,
    MulticlassTaskConfig,
    RegressionTaskConfig,
    ResponseTaskConfig,
    UpliftTaskConfig,
)
from .tasks import BinaryTask, MulticlassTask, RegressionTask, ResponseTask, UpliftTask
from .types import (
    CalibrationResult,
    EvaluationResult,
    PredictionResult,
    TrainingResult,
)

__all__ = [
    "BinaryTask",
    "BinaryTaskConfig",
    "CalibrationResult",
    "EnvironmentConfig",
    "EvaluationResult",
    "MulticlassTask",
    "MulticlassTaskConfig",
    "PredictionResult",
    "RegressionTask",
    "RegressionTaskConfig",
    "ResponseTask",
    "ResponseTaskConfig",
    "TrainingResult",
    "UpliftTask",
    "UpliftTaskConfig",
]
