"""The metric bridge must report the number AutoML would report itself."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from fmlib.automl.backends.tabnn.metric import AutoMLMetric, to_scores
from fmlib.automl.exceptions import ConfigError
from fmlib.automl.metrics import MetricInput, resolve_metric


class _Output:
    def __init__(self, logits):
        self.logits = logits


def _feed(metric, targets, logits, chunk=3):
    for start in range(0, len(targets), chunk):
        metric.update(
            inputs={"targets": torch.as_tensor(targets[start : start + chunk])},
            outputs=_Output(torch.as_tensor(logits[start : start + chunk])),
        )


def test_binary_scores_match_sklearn_on_the_whole_population():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(40, 1)).astype(np.float32)
    targets = (rng.random(40) < 1 / (1 + np.exp(-logits[:, 0]))).astype(np.int64)

    metric = AutoMLMetric("roc_auc", "binary", score_transform="sigmoid")
    _feed(metric, targets, logits)

    expected = roc_auc_score(targets, 1 / (1 + np.exp(-logits[:, 0])))
    assert metric.compute() == {"roc_auc": pytest.approx(expected)}


def test_the_bridge_agrees_with_the_registry_it_wraps():
    """Composition, not reimplementation: same numbers, same input."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 1)).astype(np.float32)
    targets = rng.integers(0, 2, 50)

    metric = AutoMLMetric("roc_auc", "binary")
    _feed(metric, targets, logits)

    direct = resolve_metric("roc_auc", "binary", "optimization").compute(
        MetricInput(
            target=targets, scores=to_scores(torch.as_tensor(logits), "sigmoid")
        )
    )
    assert metric.compute()["roc_auc"] == pytest.approx(direct)


def test_multiclass_keeps_the_probability_matrix_and_the_class_order():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(60, 3)).astype(np.float32)
    targets = rng.integers(0, 3, 60)

    metric = AutoMLMetric(
        "roc_auc_ovr_macro",
        "multiclass",
        score_transform="softmax",
        class_order=("a", "b", "c"),
    )
    _feed(metric, targets, logits)
    value = metric.compute()["roc_auc_ovr_macro"]

    probabilities = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    expected = resolve_metric(
        "roc_auc_ovr_macro", "multiclass", "optimization"
    ).compute(
        MetricInput(target=targets, scores=probabilities, class_order=("a", "b", "c"))
    )
    assert value == pytest.approx(expected)


def test_batching_does_not_change_the_answer():
    """ROC-AUC is not an average over batches, which is why it needs them all."""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(64, 1)).astype(np.float32)
    targets = rng.integers(0, 2, 64)

    coarse = AutoMLMetric("roc_auc", "binary")
    _feed(coarse, targets, logits, chunk=64)
    fine = AutoMLMetric("roc_auc", "binary")
    _feed(fine, targets, logits, chunk=7)
    assert coarse.compute() == fine.compute()
    assert coarse.needs_full_population is True


def test_reset_forgets_everything():
    metric = AutoMLMetric("roc_auc", "binary")
    _feed(metric, np.array([0, 1, 0, 1]), np.array([[0.1], [0.9], [0.2], [0.8]]))
    assert metric.compute()
    metric.reset()
    assert metric.compute() == {}


def test_the_name_and_direction_are_what_early_stopping_is_told():
    maximized = AutoMLMetric("roc_auc", "binary")
    assert (maximized.name, maximized.direction) == ("roc_auc", "max")
    minimized = AutoMLMetric("mse", "regression", score_transform="identity")
    assert (minimized.name, minimized.direction) == ("mse", "min")


def test_the_declared_fields_are_the_two_a_distributed_run_gathers():
    metric = AutoMLMetric("roc_auc", "binary", target_key="targets")
    assert metric.required_inputs == ("targets",)
    assert metric.required_outputs == ("logits",)


def test_an_unusable_metric_fails_when_the_run_is_configured():
    with pytest.raises(ConfigError):
        AutoMLMetric("roc_auc", "regression")
    with pytest.raises(ConfigError):
        AutoMLMetric("not_a_metric", "binary")


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        ("sigmoid", np.array([0.5, 0.7310586])),
        ("identity", np.array([0.0, 1.0])),
    ],
)
def test_score_transforms_flatten_a_single_column_head(transform, expected):
    scores = to_scores(torch.tensor([[0.0], [1.0]]), transform)
    assert scores.shape == (2,)
    assert np.allclose(scores, expected)


def test_softmax_keeps_the_matrix_shape():
    scores = to_scores(torch.zeros(4, 3), "softmax")
    assert scores.shape == (4, 3)
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_an_unknown_transform_is_named():
    with pytest.raises(ValueError, match="Unknown score transform"):
        to_scores(torch.zeros(2, 1), "logit")
