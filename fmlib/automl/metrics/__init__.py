"""Registered model-quality metrics used by fmlib AutoML tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from fmlib.automl.exceptions import ConfigError

from .base import Metric, MetricInput
from .binary import (
    DEFAULT_TOP_K,
    BinaryRocAuc,
    BinaryTopK,
    ClassWiseRocAuc,
    binary_roc_auc,
    binary_top_k_metrics,
)
from .multiclass import (
    MulticlassMetric,
    multiclass_class_metrics,
    multiclass_metrics,
    multiclass_objective,
)
from .regression import RegressionMetric
from .uplift import (
    UpliftArmRocAuc,
    UpliftMetric,
    perfect_qini_curve,
    perfect_uplift_curve,
    qini_auc_score,
    qini_curve,
    uplift_at_k,
    uplift_auc_score,
    uplift_curve,
)

MetricUse = Literal["optimization", "evaluation"]

_METRICS: tuple[Metric, ...] = (
    BinaryRocAuc(),
    *(BinaryTopK(kind, k) for k in DEFAULT_TOP_K for kind in ("precision", "recall")),
    ClassWiseRocAuc(),
    RegressionMetric("mse", "minimize"),
    RegressionMetric("mae", "minimize"),
    RegressionMetric("mape", None),
    MulticlassMetric("roc_auc_ovr_macro", "maximize"),
    MulticlassMetric("log_loss", "minimize"),
    MulticlassMetric("accuracy", "maximize"),
    MulticlassMetric("f1_macro", "maximize"),
    MulticlassMetric("f1_weighted", None),
    UpliftMetric("qini_auc", "maximize"),
    UpliftMetric("uplift_auc", "maximize"),
    UpliftMetric("uplift_at_10", None),
    UpliftMetric("uplift_at_20", None),
    UpliftMetric("uplift_at_50", None),
    UpliftArmRocAuc("treatment_roc_auc"),
    UpliftArmRocAuc("control_roc_auc"),
)
METRIC_REGISTRY: dict[str, Metric] = {metric.name: metric for metric in _METRICS}

DEFAULT_OPTIMIZATION_METRICS: dict[str, str] = {
    "binary": "roc_auc",
    "response": "roc_auc",
    "regression": "mse",
    "multiclass": "roc_auc_ovr_macro",
    "uplift": "qini_auc",
}
DEFAULT_EVALUATION_METRICS: dict[str, tuple[str, ...]] = {
    "binary": (
        "roc_auc",
        *(name for k in DEFAULT_TOP_K for name in (f"precision@{k}", f"recall@{k}")),
    ),
    "response": (
        "roc_auc",
        *(name for k in DEFAULT_TOP_K for name in (f"precision@{k}", f"recall@{k}")),
    ),
    "regression": ("mse", "mae", "mape"),
    "multiclass": (
        "roc_auc_ovr_macro",
        "log_loss",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "class_wise_roc_auc",
        *(name for k in DEFAULT_TOP_K for name in (f"precision@{k}", f"recall@{k}")),
    ),
    "uplift": (
        "qini_auc",
        "uplift_auc",
        "uplift_at_10",
        "uplift_at_20",
        "uplift_at_50",
        "treatment_roc_auc",
        "control_roc_auc",
    ),
}


def resolve_metric(name: str, task: str, use: MetricUse) -> Metric:
    """Resolve and validate one public metric name for a task and use mode."""
    if not isinstance(name, str) or not name:
        msg = f"Metric name must be a non-empty string; got {name!r}"
        raise ConfigError(msg)
    metric = METRIC_REGISTRY.get(name)
    if metric is None:
        msg = f"Unknown metric {name!r}; expected one of {sorted(METRIC_REGISTRY)}"
        raise ConfigError(msg)
    if task not in metric.supported_tasks:
        msg = f"Metric {name!r} is incompatible with task {task!r}"
        raise ConfigError(msg)
    if use == "optimization" and metric.optimization_direction is None:
        msg = f"Metric {name!r} is available only for evaluation and cannot be used for optimization"
        raise ConfigError(msg)
    return metric


def default_optimization_metric(task: str) -> str:
    """Return the registered default optimization metric for a task."""
    try:
        return DEFAULT_OPTIMIZATION_METRICS[task]
    except KeyError as exc:
        msg = f"No default optimization metric is registered for task {task!r}"
        raise ConfigError(msg) from exc


def resolve_evaluation_metrics(
    names: Sequence[str] | None, task: str
) -> tuple[Metric, ...]:
    """Normalize and resolve a complete evaluation metric selection."""
    if names is None:
        selected = DEFAULT_EVALUATION_METRICS[task]
    else:
        if isinstance(names, str | bytes):
            msg = "metrics must be a non-empty sequence of unique metric names, not a string"
            raise ConfigError(msg)
        try:
            selected = tuple(names)
        except TypeError as exc:
            msg = "metrics must be a non-empty sequence of unique metric names or None"
            raise ConfigError(msg) from exc
        if not selected:
            msg = "metrics must not be empty; omit it or pass None to use task defaults"
            raise ConfigError(msg)
        if not all(isinstance(name, str) and name for name in selected):
            msg = f"metrics must contain non-empty string names; got {selected!r}"
            raise ConfigError(msg)
        if len(selected) != len(set(selected)):
            msg = f"metrics must contain unique names; got {selected!r}"
            raise ConfigError(msg)
    return tuple(resolve_metric(name, task, "evaluation") for name in selected)


__all__ = [
    "DEFAULT_EVALUATION_METRICS",
    "DEFAULT_OPTIMIZATION_METRICS",
    "DEFAULT_TOP_K",
    "METRIC_REGISTRY",
    "Metric",
    "MetricInput",
    "binary_roc_auc",
    "binary_top_k_metrics",
    "default_optimization_metric",
    "multiclass_class_metrics",
    "multiclass_metrics",
    "multiclass_objective",
    "perfect_qini_curve",
    "perfect_uplift_curve",
    "qini_auc_score",
    "qini_curve",
    "resolve_evaluation_metrics",
    "resolve_metric",
    "uplift_at_k",
    "uplift_auc_score",
    "uplift_curve",
]
