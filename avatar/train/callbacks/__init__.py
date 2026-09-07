"""Built-in trainer callbacks.

Order matters in one place only: :class:`EarlyStoppingCallback` must run before
:class:`CheckpointCallback`, because it is what vetoes a save when the
monitored metric did not improve.
"""

from avatar.train.callbacks.base import EVENTS, CallbackHandler, TrainerCallback
from avatar.train.callbacks.checkpoint import CheckpointCallback
from avatar.train.callbacks.early_stopping import EarlyStoppingCallback
from avatar.train.callbacks.ema import EMACallback
from avatar.train.callbacks.metrics import TrainMetricsCallback
from avatar.train.callbacks.mlflow import MLflowCallback
from avatar.train.callbacks.perf_metrics import PerfMetricsCallback
from avatar.train.callbacks.profiler import ProfilerCallback
from avatar.train.callbacks.progress import ProgressBarCallback
from avatar.train.callbacks.throughput import ThroughputCallback
from avatar.train.callbacks.train_stats import TrainStatsCallback

__all__ = [
    "EVENTS",
    "CallbackHandler",
    "CheckpointCallback",
    "EMACallback",
    "EarlyStoppingCallback",
    "MLflowCallback",
    "PerfMetricsCallback",
    "ProfilerCallback",
    "ProgressBarCallback",
    "ThroughputCallback",
    "TrainMetricsCallback",
    "TrainStatsCallback",
    "TrainerCallback",
]
