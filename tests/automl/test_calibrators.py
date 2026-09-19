"""The two calibrators, including everything they refuse.

Both were around 70% covered, and the uncovered part was all of it: the guard
clauses. A calibrator that accepts bad input does not fail — it returns
probabilities, and nothing downstream can tell those from good ones. So the
refusals are the behaviour worth testing, alongside the property that makes the
whole thing usable: a calibrator persisted to JSON and read back scores
identically.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmlib.automl.calibrators import (
    BetaCalibrator,
    IsotonicCalibrator,
    calibrator_class,
)
from fmlib.automl.exceptions import ConfigError, SchemaError

CALIBRATORS = (BetaCalibrator, IsotonicCalibrator)


def _sample(rows: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores in [0, 1] with both classes present, which both fits accept."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.01, 0.99, size=rows)
    targets = (rng.uniform(size=rows) < scores).astype(int)
    # Both classes, whatever the draw did.
    targets[0], targets[1] = 0, 1
    return scores, targets


@pytest.mark.parametrize("calibrator", CALIBRATORS)
def test_a_fitted_calibrator_survives_a_json_round_trip(calibrator):
    """The stored form is the calibrator: same numbers, not merely close ones."""
    scores, targets = _sample()
    fitted = calibrator.fit(scores, targets)

    restored = calibrator.from_dict(fitted.to_dict())

    assert restored == fitted
    np.testing.assert_array_equal(restored.predict(scores), fitted.predict(scores))


@pytest.mark.parametrize("calibrator", CALIBRATORS)
def test_calibrated_scores_stay_probabilities(calibrator):
    scores, targets = _sample()
    calibrated = calibrator.fit(scores, targets).predict(scores)

    assert calibrated.shape == scores.shape
    assert np.isfinite(calibrated).all()
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


@pytest.mark.parametrize("calibrator", CALIBRATORS)
def test_calibration_is_monotone_in_the_score(calibrator):
    """Neither mapping may reorder anyone: calibration changes scale, not rank."""
    scores, targets = _sample()
    fitted = calibrator.fit(scores, targets)

    grid = np.linspace(0.01, 0.99, 200)
    calibrated = fitted.predict(grid)
    assert np.all(np.diff(calibrated) >= -1e-12), "calibration reordered the scores"


@pytest.mark.parametrize("calibrator", CALIBRATORS)
@pytest.mark.parametrize(
    ("scores", "targets", "expected"),
    [
        pytest.param(
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([0, 1]),
            "one-dimensional",
            id="two-dimensional",
        ),
        pytest.param(
            np.array([0.1, 0.2, 0.3]), np.array([0, 1]), None, id="length-mismatch"
        ),
        pytest.param(np.array([0.5]), np.array([1]), None, id="single-row"),
        pytest.param(
            np.array([0.1, np.nan, 0.3]), np.array([0, 1, 1]), None, id="nan-score"
        ),
        pytest.param(
            np.array([0.1, np.inf, 0.3]), np.array([0, 1, 1]), None, id="inf-score"
        ),
        pytest.param(
            np.array([0.1, 0.2, 0.3]),
            np.array([0, 0, 0]),
            "both target classes",
            id="one-class",
        ),
        pytest.param(
            np.array([0.1, 0.2, 0.3]),
            np.array([0, 1, 2]),
            "both target classes",
            id="three-classes",
        ),
    ],
)
def test_a_fit_refuses_data_it_cannot_calibrate(calibrator, scores, targets, expected):
    with pytest.raises(SchemaError, match=expected):
        calibrator.fit(scores, targets)


def test_isotonic_refuses_a_constant_score():
    """One distinct value carries no ordering, so there is nothing to fit."""
    with pytest.raises(SchemaError, match="two distinct scores"):
        IsotonicCalibrator.fit(np.full(10, 0.5), np.tile([0, 1], 5))


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_beta_refuses_scores_outside_the_unit_interval(bad):
    """Beta calibration is defined on probabilities; a raw margin is not one."""
    scores = np.array([0.2, 0.4, 0.6, bad])
    with pytest.raises(SchemaError, match=r"\[0, 1\]"):
        BetaCalibrator.fit(scores, np.array([0, 1, 0, 1]))


@pytest.mark.parametrize("calibrator", CALIBRATORS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_predict_refuses_what_it_cannot_map(calibrator, bad):
    scores, targets = _sample()
    fitted = calibrator.fit(scores, targets)
    with pytest.raises(SchemaError):
        fitted.predict(np.array([0.5, bad]))


def test_beta_predict_refuses_scores_outside_the_unit_interval():
    scores, targets = _sample()
    fitted = BetaCalibrator.fit(scores, targets)
    with pytest.raises(SchemaError, match=r"\[0, 1\]"):
        fitted.predict(np.array([0.5, 1.5]))


def test_isotonic_clips_rather_than_extrapolating():
    """Outside the fitted range the answer is the edge, never an invented one."""
    scores = np.linspace(0.2, 0.8, 50)
    targets = (scores > 0.5).astype(int)
    fitted = IsotonicCalibrator.fit(scores, targets)

    below, above = fitted.predict(np.array([-5.0, 5.0]))
    assert below == pytest.approx(fitted.y_thresholds[0])
    assert above == pytest.approx(fitted.y_thresholds[-1])


@pytest.mark.parametrize(
    ("name", "expected"),
    [("beta_calibration", BetaCalibrator), ("isotonic_regression", IsotonicCalibrator)],
)
def test_the_registry_resolves_the_documented_names(name, expected):
    assert calibrator_class(name) is expected


@pytest.mark.parametrize("name", ["isotonic", "platt", "", None])
def test_an_unknown_strategy_names_the_ones_that_exist(name):
    """The message has to be actionable: 'isotonic' is a plausible wrong guess."""
    with pytest.raises(ConfigError) as error:
        calibrator_class(name)
    assert "beta_calibration" in str(error.value)
    assert "isotonic_regression" in str(error.value)
