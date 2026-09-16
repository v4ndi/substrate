"""Task-neutral metric contracts."""

import numpy as np
import pytest

from avatar.automl.exceptions import ConfigError
from avatar.automl.metrics import (
    DEFAULT_EVALUATION_METRICS,
    METRIC_REGISTRY,
    MetricInput,
    binary_roc_auc,
    binary_top_k_metrics,
    multiclass_metrics,
    multiclass_objective,
    resolve_evaluation_metrics,
    resolve_metric,
)

_EVALUATION_ONLY_CASES = tuple(
    (task, name)
    for name, metric in METRIC_REGISTRY.items()
    if metric.optimization_direction is None
    for task in sorted(metric.supported_tasks)
)


def test_binary_metrics_are_score_ordered() -> None:
    target = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])

    assert binary_roc_auc(target, scores) == 1.0
    assert binary_top_k_metrics(target, scores, ks=(50,)) == {
        "precision@50": 1.0,
        "recall@50": 1.0,
    }


@pytest.mark.parametrize(
    "name", ["roc_auc_ovr_macro", "accuracy", "f1_macro", "log_loss"]
)
def test_multiclass_objectives_and_evaluation(name: str) -> None:
    target = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ])

    assert np.isfinite(
        multiclass_objective(name, target, probabilities, ("a", "b", "c"))
    )
    metrics = multiclass_metrics(target, probabilities, ("a", "b", "c"), grouped=False)
    assert metrics["roc_auc_ovr_macro"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc_class_0"] == 1.0


def test_registry_resolves_scalar_matrix_and_treatment_aware_inputs() -> None:
    binary = MetricInput(np.array([0, 1]), np.array([0.1, 0.9]))
    assert resolve_metric("roc_auc", "binary", "optimization").compute(binary) == 1.0

    multiclass = MetricInput(
        np.array([0, 1, 2]),
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        class_order=("a", "b", "c"),
    )
    assert (
        resolve_metric("accuracy", "multiclass", "optimization").compute(multiclass)
        == 1.0
    )

    uplift = MetricInput(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.2, 0.9]),
        treatment=np.array([0, 1, 0, 1]),
    )
    assert np.isfinite(
        resolve_metric("qini_auc", "uplift", "optimization").compute(uplift)
    )


def test_registry_owns_compatibility_directions_and_defaults() -> None:
    assert (
        resolve_metric("log_loss", "multiclass", "optimization").optimization_direction
        == "minimize"
    )
    assert (
        resolve_metric("roc_auc", "response", "optimization").optimization_direction
        == "maximize"
    )
    assert (
        tuple(metric.name for metric in resolve_evaluation_metrics(None, "multiclass"))
        == (DEFAULT_EVALUATION_METRICS["multiclass"])
    )


@pytest.mark.parametrize(("task", "name"), _EVALUATION_ONLY_CASES)
def test_every_evaluation_only_metric_is_rejected_for_optimization(
    task: str, name: str
) -> None:
    with pytest.raises(ConfigError, match="only for evaluation"):
        resolve_metric(name, task, "optimization")


@pytest.mark.parametrize("task", DEFAULT_EVALUATION_METRICS)
def test_unknown_evaluation_metric_is_rejected_for_every_task(task: str) -> None:
    with pytest.raises(ConfigError, match="Unknown metric"):
        resolve_evaluation_metrics(["not_registered"], task)


@pytest.mark.parametrize(
    ("task", "name"),
    [
        ("binary", "mse"),
        ("response", "qini_auc"),
        ("regression", "roc_auc"),
        ("multiclass", "mse"),
        ("uplift", "accuracy"),
    ],
)
def test_incompatible_evaluation_metric_is_rejected_for_every_task(
    task: str, name: str
) -> None:
    with pytest.raises(ConfigError, match="incompatible"):
        resolve_evaluation_metrics([name], task)
