"""Training on plain ``torch.distributed``.

The loop itself is :class:`~avatar.train.loop.Trainer`; everything
cross-cutting (logging, checkpoints, early stopping, EMA, profiling,
throughput) is a :class:`~avatar.train.callbacks.base.TrainerCallback`.

Launch with::

    torchrun --standalone --nproc_per_node=2 -m avatar.train \
        --config-dir=configs --config-name=my_run
"""

from avatar.train.callbacks import (
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
from avatar.train.checkpoint import (
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    save_model,
)
from avatar.train.config import (
    DDPConfig,
    DistributedConfig,
    RunConfig,
    resolve_run_config,
)
from avatar.train.dist import DistEnv, seed_everything, unwrap_model
from avatar.train.early_stopping import EarlyStopping
from avatar.train.evaluate import evaluate, predict
from avatar.train.factory import build_callbacks, build_default_callbacks
from avatar.train.loop import Trainer
from avatar.train.loss_reduce import calculate_output_loss
from avatar.train.state import CallbackContext, TrainerControl, TrainerState
from avatar.train.utils import (
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
