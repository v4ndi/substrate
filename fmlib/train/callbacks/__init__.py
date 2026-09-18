"""Built-in trainer callbacks.

Order matters in one place only: :class:`EarlyStoppingCallback` must run before
:class:`CheckpointCallback`, because it is what vetoes a save when the
monitored metric did not improve.
"""

from fmlib.train.callbacks.base import EVENTS, CallbackHandler, TrainerCallback
from fmlib.train.callbacks.checkpoint import CheckpointCallback
from fmlib.train.callbacks.early_stopping import EarlyStoppingCallback
from fmlib.train.callbacks.ema import EMACallback
from fmlib.train.callbacks.metrics import TrainMetricsCallback
from fmlib.train.callbacks.mlflow import MLflowCallback
from fmlib.train.callbacks.perf_metrics import PerfMetricsCallback
from fmlib.train.callbacks.profiler import ProfilerCallback
from fmlib.train.callbacks.progress import ProgressBarCallback
from fmlib.train.callbacks.throughput import ThroughputCallback
from fmlib.train.callbacks.train_stats import TrainStatsCallback

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
