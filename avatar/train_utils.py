"""Deprecated alias for :mod:`avatar.train`.

The training stack moved into the ``avatar.train`` package when ``accelerate``
was replaced by plain ``torch.distributed``. This module keeps the old import
paths working for configs that still name ``avatar.train_utils.EarlyStopping``.
"""

import warnings

from avatar.train.checkpoint import rotate_checkpoints, save_checkpoint, save_model
from avatar.train.early_stopping import EarlyStopping
from avatar.train.loss_reduce import calculate_output_loss
from avatar.train.utils import (
    IGNORE_CONFIG_KEYS,
    PRESERVE_CONFIG_KEYS,
    apply_basic_loader_checks,
    flatten_dict,
    get_default_log_params,
    move_to_device,
    normalize_log_param,
    wrap_metrics,
)

warnings.warn(
    "avatar.train_utils is deprecated; import from avatar.train instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "IGNORE_CONFIG_KEYS",
    "PRESERVE_CONFIG_KEYS",
    "EarlyStopping",
    "apply_basic_loader_checks",
    "calculate_output_loss",
    "flatten_dict",
    "get_default_log_params",
    "move_to_device",
    "normalize_log_param",
    "rotate_checkpoints",
    "save_checkpoint",
    "save_model",
    "wrap_metrics",
]
