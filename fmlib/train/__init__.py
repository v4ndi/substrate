"""Training on plain ``torch.distributed``.

The loop itself is :class:`~fmlib.train.loop.Trainer`; everything
cross-cutting (logging, checkpoints, early stopping, EMA, profiling,
throughput) is a :class:`~fmlib.train.callbacks.base.TrainerCallback`.

Launch with::

    torchrun --standalone --nproc_per_node=2 -m fmlib.train \
        --config-dir=configs --config-name=my_run
"""

from fmlib.train.callbacks import (
    CallbackHandler,
    CheckpointCallback,
    EarlyStoppingCallback,
    EMACallback,
    MLflowCallback,
    PerfMetricsCallback,
    ProfilerCallback,
    ProgressBarCallback,
    ThroughputCallback,
    TrainerCallback,
    TrainMetricsCallback,
    TrainStatsCallback,
)
from fmlib.train.checkpoint import (
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    save_model,
)
from fmlib.train.config import (
    DDPConfig,
    DistributedConfig,
    RunConfig,
    resolve_run_config,
)
from fmlib.train.dist import DistEnv, seed_everything, unwrap_model
from fmlib.train.early_stopping import EarlyStopping
from fmlib.train.evaluate import evaluate, predict
from fmlib.train.factory import build_callbacks, build_default_callbacks
from fmlib.train.loop import Trainer
from fmlib.train.loss_reduce import calculate_output_loss
from fmlib.train.state import CallbackContext, TrainerControl, TrainerState
from fmlib.train.utils import (
    apply_basic_loader_checks,
    flatten_dict,
    get_default_log_params,
    move_to_device,
    normalize_log_param,
    prefix_metrics,
    wrap_metrics,
)

__all__ = [
    "CallbackContext",
    "CallbackHandler",
    "CheckpointCallback",
    "DDPConfig",
    "DistEnv",
    "DistributedConfig",
    "EMACallback",
    "EarlyStopping",
    "EarlyStoppingCallback",
    "MLflowCallback",
    "PerfMetricsCallback",
    "ProfilerCallback",
    "ProgressBarCallback",
    "RunConfig",
    "ThroughputCallback",
    "TrainMetricsCallback",
    "TrainStatsCallback",
    "Trainer",
    "TrainerCallback",
    "TrainerControl",
    "TrainerState",
    "apply_basic_loader_checks",
    "build_callbacks",
    "build_default_callbacks",
    "calculate_output_loss",
    "evaluate",
    "flatten_dict",
    "get_default_log_params",
    "load_checkpoint",
    "move_to_device",
    "normalize_log_param",
    "predict",
    "prefix_metrics",
    "resolve_run_config",
    "rotate_checkpoints",
    "save_checkpoint",
    "save_model",
    "seed_everything",
    "unwrap_model",
    "wrap_metrics",
]
