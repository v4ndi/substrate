"""What the metrics refuse, and what they return null for instead.

A metric is how a model gets chosen. One that quietly returns a number on data
it cannot describe picks the wrong model and leaves no trace, so the
interesting behaviour is at the edges: a validation split missing a class, a
selection with only one treatment arm, an uplift curve with no attainable area.

The distinction that matters here, and the one that was untested: **overall
metrics raise, grouped metrics warn and return null**. A group too small to
support a metric must not fail the whole evaluation, and the whole evaluation
must not silently skip a metric it was asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmlib.automl.exceptions import ConfigError, SchemaError
from fmlib.automl.metrics.multiclass import (
    multiclass_class_metrics,
    multiclass_metrics,
    multiclass_objective,
)
from fmlib.automl.metrics.uplift import (
    perfect_qini_curve,
    qini_auc_score,
    uplift_at_k,
    uplift_auc_score,
)

CLASSES = ("a", "b", "c")


def _probabilities(rows: int, classes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(size=(rows, classes))
    return raw / raw.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Multiclass                                                                   #
# --------------------------------------------------------------------------- #
def test_an_unknown_multiclass_metric_is_a_config_error():
    target = np.array([0, 1, 2, 0])
    with pytest.raises(ConfigError, match="Unsupported multiclass metric"):
        multiclass_objective("auc", target, _probabilities(4, 3), CLASSES)


@pytest.mark.parametrize("name", ["accuracy", "f1_macro", "log_loss"])
def test_the_argmax_metrics_work_on_a_split_missing_a_class(name):
    """These do not need every class present, and must not pretend otherwise."""
    target = np.array([0, 0, 1, 1])
    value = multiclass_objective(name, target, _probabilities(4, 3), CLASSES)
    assert np.isfinite(value)


def test_the_objective_auc_refuses_a_split_it_cannot_score():
    """One class in the validation split makes a one-vs-rest AUC meaningless."""
    target = np.zeros(6, dtype=int)
    with pytest.raises(SchemaError, match="undefined for the validation split"):
        multiclass_objective("roc_auc_ovr_macro", target, _probabilities(6, 3), CLASSES)


def test_overall_evaluation_refuses_a_split_missing_a_class():
    """Named in the message, because the fix is to change the split."""
    target = np.array([0, 0, 1, 1, 1, 0])
    with pytest.raises(SchemaError, match="'c'"):
        multiclass_metrics(target, _probabilities(6, 3), CLASSES, grouped=False)


def test_a_group_missing_a_class_reports_null_rather_than_failing(caplog):
    """One unscoreable group must not take the rest of the evaluation with it."""
    target = np.array([0, 0, 1, 1, 1, 0])
    with caplog.at_level("WARNING"):
        metrics = multiclass_metrics(
            target, _probabilities(6, 3), CLASSES, grouped=True
        )

    assert metrics["roc_auc_ovr_macro"] is None
    # The other metrics are still real numbers: only the undefined one is null.
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["accuracy"])
    assert "missing classes" in caplog.text


def test_class_wise_auc_is_null_for_a_class_with_no_positives(caplog):
    target = np.array([0, 0, 1, 1])
    with caplog.at_level("WARNING"):
        metrics = multiclass_class_metrics(
            target, _probabilities(4, 3), CLASSES, grouped=True
        )

    assert metrics["roc_auc_class_2"] is None
    assert metrics["n_positives_class_2"] == 0
    assert metrics["roc_auc_class_0"] is not None
    assert "class_index=2" in caplog.text


def test_class_wise_auc_stays_quiet_when_it_is_not_a_group(caplog):
    """The warning is for grouped evaluation; overall already raised by then."""
    target = np.array([0, 0, 1, 1])
    with caplog.at_level("WARNING"):
        metrics = multiclass_class_metrics(
            target, _probabilities(4, 3), CLASSES, grouped=False
        )
    assert metrics["roc_auc_class_2"] is None
    assert caplog.text == ""


# --------------------------------------------------------------------------- #
# Uplift                                                                       #
# --------------------------------------------------------------------------- #
def _uplift_sample(rows: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, rows)
    score = rng.uniform(size=rows)
    target = (rng.uniform(size=rows) < 0.2 + 0.4 * treatment * score).astype(int)
    target[0], target[1] = 0, 1
    return target, score, treatment


@pytest.mark.parametrize("metric", [qini_auc_score, uplift_auc_score])
def test_uplift_metrics_refuse_a_single_treatment_arm(metric):
    target, score, _ = _uplift_sample()
    with pytest.raises(ValueError, match="treatment must contain both"):
        metric(target, score, np.zeros_like(target))


@pytest.mark.parametrize("metric", [qini_auc_score, uplift_auc_score])
def test_uplift_metrics_refuse_a_single_outcome(metric):
    _, score, treatment = _uplift_sample()
    with pytest.raises(ValueError, match="y_true must contain both"):
        metric(np.zeros(len(score), dtype=int), score, treatment)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_uplift_metrics_refuse_a_non_finite_score(bad):
    target, score, treatment = _uplift_sample()
    score = score.copy()
    score[0] = bad
    with pytest.raises(ValueError, match="only finite values"):
        qini_auc_score(target, score, treatment)


def test_the_perfect_curve_flag_has_to_be_a_bool():
    """`negative_effect=1` would pick a different curve and score differently."""
    target, _, treatment = _uplift_sample()
    with pytest.raises(TypeError, match="negative_effect must be bool"):
        perfect_qini_curve(target, treatment, negative_effect=1)


@pytest.mark.parametrize("negative_effect", [True, False])
def test_both_perfect_curves_are_scoreable(negative_effect):
    target, score, treatment = _uplift_sample()
    value = qini_auc_score(target, score, treatment, negative_effect=negative_effect)
    assert np.isfinite(value)


def test_uplift_at_k_refuses_an_unknown_strategy():
    target, score, treatment = _uplift_sample()
    with pytest.raises(ValueError, match="strategy must be"):
        uplift_at_k(target, score, treatment, strategy="top", k=0.3)


@pytest.mark.parametrize("k", [0.0, 1.0, -0.5, 1.5])
def test_uplift_at_k_refuses_a_fraction_outside_the_open_unit_interval(k):
    target, score, treatment = _uplift_sample()
    with pytest.raises(ValueError, match=r"float k must lie in \(0, 1\)"):
        uplift_at_k(target, score, treatment, k=k)


@pytest.mark.parametrize("k", [0, -3, 10_000])
def test_uplift_at_k_refuses_an_integer_k_outside_the_sample(k):
    target, score, treatment = _uplift_sample()
    with pytest.raises(ValueError, match="integer k must be positive"):
        uplift_at_k(target, score, treatment, k=k)


@pytest.mark.parametrize("strategy", ["overall", "by_group"])
@pytest.mark.parametrize("k", [0.3, 50])
def test_uplift_at_k_scores_both_strategies_and_both_kinds_of_k(strategy, k):
    target, score, treatment = _uplift_sample()
    assert np.isfinite(uplift_at_k(target, score, treatment, strategy, k))


def test_uplift_at_k_refuses_a_selection_with_one_arm():
    """Top-scored rows that are all treated give a difference of nothing."""
    rows = 100
    treatment = np.array([1] * 50 + [0] * 50)
    # Score ranks every treated row above every control row, so the top decile
    # is entirely treated.
    score = np.linspace(1.0, 0.0, rows)
    target = np.array([0, 1] * 50)
    with pytest.raises(ValueError, match="both treatment arms"):
        uplift_at_k(target, score, treatment, "overall", 0.1)
