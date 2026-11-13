"""When UpliftMetrics is allowed to report a calibrated number, and when not.

A calibrator fitted on the rows it is then scored on measures nothing, so the
rule is narrow: calibrated metrics exist only for a held-out ``test`` slice.
Everything here pins down a case where the metric must stay silent rather than
produce a number that looks fine.
"""

import logging
import types

import numpy as np
import pytest
import torch

from avatar.metrics import UpliftMetrics


def uplift_batch(conversion, t_probs, c_probs, treatment, split_type=None):
    """One batch as :class:`UpliftMetrics` sees it, optionally without a split."""
    size = len(conversion)
    inputs = {"epk_id": np.arange(size)}
    if split_type is not None:
        inputs["split_type"] = np.array(split_type)
    outputs = types.SimpleNamespace(
        uplift=torch.tensor(t_probs) - torch.tensor(c_probs),
        treatment=torch.tensor(treatment),
        conversion=torch.tensor(conversion),
        control_probs=torch.tensor(c_probs),
        treatment_probs=torch.tensor(t_probs),
        group=torch.zeros(size, dtype=torch.int64),
    )
    return inputs, outputs


def population(size=400, seed=0, split_type=None, conversion=None):
    """A population where both heads rank conversions."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, size)
    t_probs = rng.uniform(0.1, 0.9, size)
    c_probs = rng.uniform(0.1, 0.9, size)
    if conversion is None:
        probs = np.where(treatment == 1, t_probs, c_probs)
        conversion = (rng.uniform(size=size) < probs).astype(np.float64)
    return uplift_batch(
        conversion=conversion,
        t_probs=t_probs,
        c_probs=c_probs,
        treatment=treatment,
        split_type=split_type,
    )


def halves(size=400):
    """Half the population calibrates, half is held out."""
    return ["calib"] * (size // 2) + ["test"] * (size // 2)


def calibrated_names(scores) -> list[str]:
    return [name for name in scores if "calibrated" in name]


def test_without_a_split_type_column_nothing_calibrated_is_reported():
    """Every row is then the calibration slice, and scoring it is in-sample."""
    metric = UpliftMetrics(require_calibration=True)
    metric.update(*population())
    scores = metric.compute()

    assert calibrated_names(scores) == []
    assert "mean_calibrated_qini_auc_score" not in scores
    assert "mean_qini_auc_score" in scores


def test_asking_for_calibration_without_a_held_out_slice_says_so(caplog):
    """The run is not left guessing why the number it asked for is missing."""
    metric = UpliftMetrics(require_calibration=True)
    metric.update(*population())
    with caplog.at_level(logging.WARNING, logger="avatar.metrics.uplift"):
        metric.compute()

    assert any(
        "require_calibration is set" in record.message for record in caplog.records
    )


def test_the_fitted_slice_gets_no_calibrated_metrics():
    """``calib`` is what the calibrator saw, so its calibrated score is in-sample."""
    metric = UpliftMetrics(require_calibration=True)
    metric.update(*population(split_type=halves()))
    scores = metric.compute()

    assert "test_group_0_calibrated_qini_auc_score" in scores
    assert "calib_group_0_calibrated_qini_auc_score" not in scores
    assert "calib_group_0_qini_auc_score" in scores


def test_the_fit_is_still_diagnosed_on_the_slice_it_used():
    """The gap between calibrated probability and observed rate belongs there."""
    metric = UpliftMetrics(require_calibration=True)
    metric.update(*population(split_type=halves()))
    scores = metric.compute()

    assert "calib_group_0_calibration_gap_control" in scores
    assert "calib_group_0_calibration_rank_shift_control" in scores
    # Out of sample there is no rate to reproduce, only a ranking to preserve.
    assert "test_group_0_calibration_gap_control" not in scores
    assert "test_group_0_calibration_rank_shift_control" in scores


def test_calibration_off_fits_nothing_at_all():
    """``require_calibration=False`` now means what it says."""
    metric = UpliftMetrics(require_calibration=False)
    metric.update(*population(split_type=halves()))
    scores = metric.compute()

    assert calibrated_names(scores) == []
    assert "test_group_0_control_roc_auc_score" not in scores
    assert "test_group_0_qini_auc_score" in scores


def test_a_slice_without_conversions_produces_no_metrics(caplog):
    """It used to get one invented conversion so the call would not raise."""
    size = 200
    metric = UpliftMetrics()
    metric.update(*population(size=size, conversion=np.zeros(size)))
    with caplog.at_level(logging.WARNING, logger="avatar.metrics.uplift"):
        scores = metric.compute()

    assert scores == {}
    assert any("no conversions" in record.message for record in caplog.records)


def test_a_slice_with_one_arm_produces_no_metrics(caplog):
    """Uplift compares two arms; with one there is nothing to compare."""
    size = 200
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.1, 0.9, size)
    metric = UpliftMetrics()
    metric.update(
        *uplift_batch(
            conversion=(rng.uniform(size=size) < probs).astype(np.float64),
            t_probs=probs,
            c_probs=rng.uniform(0.1, 0.9, size),
            treatment=np.ones(size, dtype=np.int64),
        )
    )
    with caplog.at_level(logging.WARNING, logger="avatar.metrics.uplift"):
        scores = metric.compute()

    assert scores == {}
    assert any("no control records" in record.message for record in caplog.records)


def test_a_group_whose_calib_slice_cannot_be_fitted_keeps_its_raw_metrics(caplog):
    """No calibrator is not the same as no metrics."""
    size = 400
    rng = np.random.default_rng(2)
    conversion = np.zeros(size)
    # Conversions only in the held-out half: the calib half cannot be fitted.
    conversion[size // 2 :] = (rng.uniform(size=size // 2) < 0.5).astype(np.float64)

    metric = UpliftMetrics(require_calibration=True)
    metric.update(
        *population(size=size, split_type=halves(size), conversion=conversion)
    )
    with caplog.at_level(logging.WARNING, logger="avatar.metrics.uplift"):
        scores = metric.compute()

    assert "test_group_0_qini_auc_score" in scores
    assert calibrated_names(scores) == []
    assert any(
        "no calibrator could be fitted" in record.message for record in caplog.records
    )


def test_the_conversions_are_never_rewritten():
    """The buffered labels come out of ``compute`` exactly as they went in."""
    size = 200
    conversion = np.zeros(size)
    metric = UpliftMetrics()
    metric.update(*population(size=size, conversion=conversion))
    before = metric.preds[0]["y_true"].copy()

    metric.compute()

    assert metric.preds[0]["y_true"].tolist() == before.tolist()
    assert metric.preds[0]["y_true"].sum() == pytest.approx(0.0)
