"""The group/split skeleton shared by the uplift and supervised metrics.

These metrics buffer batches and only produce numbers at the end of an epoch,
so what is worth pinning down is the naming — one key per metric per group per
split — and the choice of which slice feeds ``mean_{main_metric}``.
"""

import logging
import math
import types

import numpy as np
import pytest
import torch

from avatar.metrics import RegressionMetrics, ResponseMetrics, UpliftMetrics


def supervised_batch(targets, logits, group, split_type):
    """One batch as the supervised metrics see it."""
    inputs = {
        "epk_id": np.arange(len(targets)),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "group": torch.tensor(group),
        "split_type": np.array(split_type),
    }
    outputs = types.SimpleNamespace(
        logits=torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
    )
    return inputs, outputs


def uplift_batch(conversion, t_probs, c_probs, treatment, group, split_type):
    """One batch as :class:`UpliftMetrics` sees it."""
    inputs = {
        "epk_id": np.arange(len(conversion)),
        "split_type": np.array(split_type),
    }
    outputs = types.SimpleNamespace(
        uplift=torch.tensor(t_probs) - torch.tensor(c_probs),
        treatment=torch.tensor(treatment),
        conversion=torch.tensor(conversion),
        control_probs=torch.tensor(c_probs),
        treatment_probs=torch.tensor(t_probs),
        group=torch.tensor(group),
    )
    return inputs, outputs


@pytest.fixture
def two_split_response():
    """Eighty records in one group, split into calib and test halves.

    Big enough that the top five percent is a whole record — below that
    ``precision_at_5`` has no top-k to look at and is not reported at all.
    """
    half = 40
    targets = [index % 2 for index in range(2 * half)]
    return supervised_batch(
        targets=targets,
        logits=[2.0 if target else -2.0 for target in targets],
        group=[0] * (2 * half),
        split_type=["calib"] * half + ["test"] * half,
    )


def test_response_names_every_metric_per_group_and_split(two_split_response):
    """A slice's numbers are named ``{split}_group_{group}_{metric}``."""
    metric = ResponseMetrics()
    metric.update(*two_split_response)
    scores = metric.compute()

    assert "calib_group_0_roc_auc_score" in scores
    assert "test_group_0_roc_auc_score" in scores
    assert "test_group_0_precision_at_5" in scores


def test_no_metric_name_carries_a_task(two_split_response):
    """``task_name`` is gone: nothing is reported per task any more."""
    metric = ResponseMetrics()
    metric.update(*two_split_response)

    assert not [name for name in metric.compute() if name.startswith("task_")]


def test_the_held_out_slice_is_the_one_that_counts():
    """With both splits present, ``mean_`` follows ``test``, not ``calib``."""
    metric = ResponseMetrics(main_metric="roc_auc_score")
    # calib ranks perfectly, test ranks backwards: the two cannot be confused.
    metric.update(
        *supervised_batch(
            targets=[0, 0, 1, 1, 0, 0, 1, 1],
            logits=[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            group=[0] * 8,
            split_type=["calib"] * 4 + ["test"] * 4,
        )
    )
    scores = metric.compute()

    assert scores["calib_group_0_roc_auc_score"] == pytest.approx(1.0)
    assert scores["test_group_0_roc_auc_score"] == pytest.approx(0.0)
    assert scores["mean_roc_auc_score"] == pytest.approx(0.0)


def test_a_calib_only_group_still_contributes_to_the_mean():
    """No held-out slice is not the same as no data."""
    metric = ResponseMetrics(main_metric="roc_auc_score")
    metric.update(
        *supervised_batch(
            targets=[0, 0, 1, 1],
            logits=[-1.0, -1.0, 1.0, 1.0],
            group=[0] * 4,
            split_type=["calib"] * 4,
        )
    )
    scores = metric.compute()

    assert "test_group_0_roc_auc_score" not in scores
    assert scores["mean_roc_auc_score"] == pytest.approx(1.0)


def test_groups_are_scored_independently():
    """Each group gets its own key, and the mean averages over them."""
    metric = ResponseMetrics(main_metric="roc_auc_score")
    metric.update(
        *supervised_batch(
            targets=[0, 0, 1, 1, 0, 0, 1, 1],
            logits=[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
            group=[0, 0, 0, 0, 7, 7, 7, 7],
            split_type=["calib"] * 8,
        )
    )
    scores = metric.compute()

    assert scores["calib_group_0_roc_auc_score"] == pytest.approx(1.0)
    assert scores["calib_group_7_roc_auc_score"] == pytest.approx(0.0)
    assert scores["mean_roc_auc_score"] == pytest.approx(0.5)


def test_regression_keeps_its_three_metrics_apart():
    """Every regression metric gets its own key.

    The name used to be built as ``f"{task}_" if len(task) != 0 else "" + key``,
    which parses as ``(f"{task}_") if cond else ("" + key)`` — so under a
    non-empty task all three metrics collapsed into the single key ``"task_"``
    and only the last one survived.
    """
    metric = RegressionMetrics()
    metric.update(
        *supervised_batch(
            targets=[1.0, 2.0, 3.0, 4.0],
            logits=[1.5, 2.5, 2.5, 3.5],
            group=[0] * 4,
            split_type=["calib"] * 4,
        )
    )
    scores = metric.compute()

    assert {
        "calib_group_0_mse",
        "calib_group_0_rmse",
        "calib_group_0_mae",
        "calib_group_0_mape",
        "calib_group_0_r2",
    } <= set(scores)
    assert scores["calib_group_0_mae"] == pytest.approx(0.5)
    assert scores["mean_mae"] == pytest.approx(0.5)


def test_mape_ignores_the_records_whose_target_is_zero(caplog):
    """A zero target makes the percentage error meaningless, not enormous.

    ``mean_absolute_percentage_error`` divides by ``max(|y|, eps)``, so one zero
    target used to push the reported mape to about ``1e16``. Being finite, it
    passed the undefined-value filter and went on to represent the model.
    """
    metric = RegressionMetrics()
    metric.update(
        *supervised_batch(
            targets=[0.0, 0.0, 10.0, 20.0],
            logits=[1.0, 2.0, 11.0, 18.0],
            group=[0] * 4,
            split_type=["calib"] * 4,
        )
    )
    with caplog.at_level(logging.WARNING):
        scores = metric.compute()

    # |11 - 10| / 10 and |18 - 20| / 20, averaged.
    assert scores["calib_group_0_mape"] == pytest.approx(0.1)
    assert "skips 2 of 4" in caplog.text


def test_a_target_column_of_zeros_reports_no_mape_at_all(caplog):
    metric = RegressionMetrics()
    metric.update(
        *supervised_batch(
            targets=[0.0, 0.0, 0.0, 0.0],
            logits=[1.0, 2.0, 0.5, 0.0],
            group=[0] * 4,
            split_type=["calib"] * 4,
        )
    )
    with caplog.at_level(logging.WARNING):
        scores = metric.compute()

    assert "calib_group_0_mape" not in scores
    assert "calib_group_0_mae" in scores
    assert "every target is zero" in caplog.text


def test_batches_accumulate_and_reset_clears_them(two_split_response):
    """``update`` buffers; ``reset`` drops the buffer."""
    metric = ResponseMetrics()
    metric.update(*two_split_response)
    metric.update(*two_split_response)
    assert len(metric.preds) == 2

    metric.reset()
    assert metric.preds == []


@pytest.fixture
def uplift_population():
    """A population where the treatment head genuinely ranks conversions."""
    rng = np.random.default_rng(0)
    size = 400
    treatment = rng.integers(0, 2, size)
    t_probs = rng.uniform(0.1, 0.9, size)
    c_probs = rng.uniform(0.1, 0.9, size)
    probs = np.where(treatment == 1, t_probs, c_probs)
    conversion = (rng.uniform(size=size) < probs).astype(np.float64)
    return uplift_batch(
        conversion=conversion,
        t_probs=t_probs,
        c_probs=c_probs,
        treatment=treatment,
        group=np.zeros(size, dtype=np.int64),
        split_type=np.array(["calib"] * (size // 2) + ["test"] * (size // 2)),
    )


def test_uplift_reports_the_raw_family_on_both_splits(uplift_population):
    """Raw metrics need no calibrator and are reported wherever there is data."""
    metric = UpliftMetrics()
    metric.update(*uplift_population)
    scores = metric.compute()

    assert "calib_group_0_qini_auc_score" in scores
    assert "test_group_0_qini_auc_score" in scores
    assert not [name for name in scores if name.startswith("task_")]


def test_the_raw_and_calibrated_means_are_separate_numbers(uplift_population):
    """``mean_X`` follows the raw family, ``mean_calibrated_X`` the calibrated one.

    They used to be written from the same list, so the run could not report an
    uncalibrated summary at all.
    """
    metric = UpliftMetrics(require_calibration=True)
    metric.update(*uplift_population)
    scores = metric.compute()

    assert scores["mean_qini_auc_score"] == pytest.approx(
        scores["test_group_0_qini_auc_score"]
    )
    assert scores["mean_calibrated_qini_auc_score"] == pytest.approx(
        scores["test_group_0_calibrated_qini_auc_score"]
    )


def test_uplift_writes_a_submit_file(uplift_population, tmp_path):
    """The submit carries the calibrated uplift the group loop filled in."""
    import pandas as pd

    metric = UpliftMetrics(
        require_calibration=True, save_submit_path=str(tmp_path / "submit")
    )
    metric.update(*uplift_population)
    metric.compute()

    frame = pd.read_parquet(tmp_path / "submit" / "predict.parquet")
    assert "calibrated_uplift" in frame.columns
    assert "task_name" not in frame.columns
    assert not frame["calibrated_uplift"].isna().any()


def test_a_metric_that_cannot_be_computed_is_not_reported(caplog):
    """An undefined number is dropped, not published as ``nan``.

    ``precision_at_k`` divides by a ``k`` that rounds to zero on a small slice.
    The ``nan`` used to travel into the training loop, where every comparison
    against it is false and early stopping took it for an improvement.
    """
    metric = ResponseMetrics()
    metric.update(
        *supervised_batch(
            targets=[0, 1, 0, 1],
            logits=[-1.0, 1.0, -1.0, 1.0],
            group=[0] * 4,
            split_type=["calib"] * 4,
        )
    )
    with caplog.at_level(logging.WARNING, logger="avatar.metrics.grouped"):
        scores = metric.compute()

    assert "calib_group_0_roc_auc_score" in scores
    assert "calib_group_0_precision_at_5" not in scores
    assert not [name for name, value in scores.items() if math.isnan(value)]
    assert any("is not reported" in record.message for record in caplog.records)
