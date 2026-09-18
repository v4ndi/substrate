"""Metrics for a head wider than one column.

What makes the multiclass case different from the rest of
:mod:`fmlib.metrics.supervised` is that the prediction kept per record is a
row of probabilities, not a number. Everything downstream of that — the group
loop, the mean, the submit file — has to cope with a column that is itself a
matrix.
"""

import logging
import types

import numpy as np
import pandas as pd
import pytest
import torch

from fmlib.metrics import InferenceSupervisedMetrics, MultiClassMetrics

NUM_CLASSES = 3


def multiclass_batch(targets, logits, group=None, split_type=None):
    """One batch as :class:`MultiClassMetrics` sees it."""
    size = len(targets)
    inputs = {
        "epk_id": np.arange(size),
        "targets": torch.tensor(targets, dtype=torch.long),
        "group": torch.tensor(group if group is not None else [0] * size),
        "split_type": np.array(
            split_type if split_type is not None else ["calib"] * size
        ),
    }
    outputs = types.SimpleNamespace(logits=torch.tensor(logits, dtype=torch.float32))
    return inputs, outputs


def confident(label, size=NUM_CLASSES, strength=4.0):
    """Logits that put nearly all the mass on ``label``."""
    return [strength if index == label else 0.0 for index in range(size)]


@pytest.fixture
def perfect_population():
    """Thirty records, every class present, every prediction right."""
    targets = [index % NUM_CLASSES for index in range(30)]
    return multiclass_batch(
        targets=targets,
        logits=[confident(target) for target in targets],
        split_type=["calib"] * 15 + ["test"] * 15,
    )


def test_every_metric_is_named_per_group_and_split(perfect_population):
    metric = MultiClassMetrics(num_classes=NUM_CLASSES)
    metric.update(*perfect_population)

    scores = metric.compute()

    for split in ("calib", "test"):
        for name in (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "roc_auc_score_ovr",
            "log_loss",
        ):
            assert f"{split}_group_0_{name}" in scores
    assert scores["mean_balanced_accuracy"] == pytest.approx(1.0)


def test_a_perfect_classifier_scores_one_and_a_confused_one_does_not():
    right = MultiClassMetrics(num_classes=NUM_CLASSES)
    right.update(
        *multiclass_batch(
            [0, 1, 2, 0, 1, 2], [confident(c) for c in [0, 1, 2, 0, 1, 2]]
        )
    )

    wrong = MultiClassMetrics(num_classes=NUM_CLASSES)
    wrong.update(
        *multiclass_batch(
            [0, 1, 2, 0, 1, 2], [confident(c) for c in [1, 2, 0, 1, 2, 0]]
        )
    )

    assert right.compute()["calib_group_0_accuracy"] == pytest.approx(1.0)
    assert wrong.compute()["calib_group_0_accuracy"] == pytest.approx(0.0)


def test_balanced_accuracy_sees_through_a_majority_class():
    """Always answering the majority class is 80% accurate and worth nothing."""
    targets = [0] * 8 + [1, 2]
    metric = MultiClassMetrics(num_classes=NUM_CLASSES)
    metric.update(*multiclass_batch(targets, [confident(0) for _ in targets]))

    scores = metric.compute()

    assert scores["calib_group_0_accuracy"] == pytest.approx(0.8)
    assert scores["calib_group_0_balanced_accuracy"] == pytest.approx(1 / 3)


def test_groups_are_scored_independently():
    targets = [0, 1, 2, 0, 1, 2]
    metric = MultiClassMetrics(num_classes=NUM_CLASSES)
    metric.update(
        *multiclass_batch(
            targets=targets,
            # Group 0 is right, group 1 is wrong.
            logits=[confident(c) for c in [0, 1, 2]]
            + [confident(c) for c in [1, 2, 0]],
            group=[0, 0, 0, 1, 1, 1],
        )
    )

    scores = metric.compute()

    assert scores["calib_group_0_accuracy"] == pytest.approx(1.0)
    assert scores["calib_group_1_accuracy"] == pytest.approx(0.0)
    assert scores["mean_balanced_accuracy"] == pytest.approx(0.5)


def test_a_class_with_no_contrast_is_left_out_of_the_auc(caplog):
    """A rare class missing from a group must not destroy the whole number.

    ``roc_auc_score(..., multi_class="ovr")`` raises when a label is absent from
    ``y_true``; here the remaining classes are still scored.
    """
    targets = [0, 0, 1, 1]  # class 2 never occurs
    metric = MultiClassMetrics(num_classes=NUM_CLASSES)
    metric.update(*multiclass_batch(targets, [confident(target) for target in targets]))

    with caplog.at_level(logging.WARNING):
        scores = metric.compute()

    assert scores["calib_group_0_roc_auc_score_ovr"] == pytest.approx(1.0)
    assert "leaves out class(es) [2]" in caplog.text


def test_a_single_class_slice_reports_no_auc_at_all(caplog):
    targets = [1, 1, 1, 1]
    metric = MultiClassMetrics(num_classes=NUM_CLASSES)
    metric.update(*multiclass_batch(targets, [confident(target) for target in targets]))

    with caplog.at_level(logging.WARNING):
        scores = metric.compute()

    assert "calib_group_0_roc_auc_score_ovr" not in scores
    assert "calib_group_0_accuracy" in scores


def test_the_held_out_slice_is_the_one_that_feeds_the_mean():
    targets = [0, 1, 2, 0, 1, 2]
    metric = MultiClassMetrics(num_classes=NUM_CLASSES)
    metric.update(
        *multiclass_batch(
            targets=targets,
            logits=[confident(c) for c in [0, 1, 2]]
            + [confident(c) for c in [1, 2, 0]],
            split_type=["calib"] * 3 + ["test"] * 3,
        )
    )

    scores = metric.compute()

    assert scores["calib_group_0_accuracy"] == pytest.approx(1.0)
    assert scores["test_group_0_accuracy"] == pytest.approx(0.0)
    assert scores["mean_balanced_accuracy"] == pytest.approx(0.0)


def test_the_submit_file_spreads_the_probabilities_into_one_column_per_class(
    perfect_population, tmp_path
):
    """``pd.DataFrame`` will not take a column that is itself a matrix."""
    metric = MultiClassMetrics(num_classes=NUM_CLASSES, save_submit_path=str(tmp_path))
    metric.update(*perfect_population)

    metric.compute()

    frame = pd.read_parquet(tmp_path / "predict.parquet")
    assert [f"y_pred_{index}" for index in range(NUM_CLASSES)] == [
        name for name in frame.columns if name.startswith("y_pred")
    ]
    assert len(frame) == 30
    assert (
        frame[[f"y_pred_{i}" for i in range(NUM_CLASSES)]]
        .sum(axis=1)
        .round(5)
        .eq(1)
        .all()
    )


def test_a_one_wide_head_is_the_response_setting_and_says_so():
    with pytest.raises(ValueError, match="ResponseMetrics"):
        MultiClassMetrics(num_classes=1)


# -- the inference collector -------------------------------------------------


def test_inference_saves_the_chosen_class_and_the_whole_distribution(tmp_path):
    """Which class won is rarely all a downstream campaign wants to know."""
    metric = InferenceSupervisedMetrics(
        path_to_save=str(tmp_path), save_steps=10, task_type="multi_clf"
    )
    metric.update(
        inputs={"epk_id": [1, 2, 3]},
        outputs=types.SimpleNamespace(
            logits=torch.tensor(
                [confident(0), confident(2), confident(1)], dtype=torch.float32
            )
        ),
    )

    metric.flush()

    frame = pd.read_parquet(next(tmp_path.glob("*.parquet")))
    assert list(frame["prediction"]) == [0, 2, 1]
    assert [f"probability_{index}" for index in range(NUM_CLASSES)] == [
        name for name in frame.columns if name.startswith("probability")
    ]
    assert (
        frame[[f"probability_{i}" for i in range(NUM_CLASSES)]]
        .sum(axis=1)
        .round(5)
        .eq(1)
        .all()
    )


def test_inference_takes_an_id_column_that_arrived_as_a_tensor(tmp_path):
    """``add_extra_columns`` makes tensors, and by now they live on the device."""
    metric = InferenceSupervisedMetrics(
        path_to_save=str(tmp_path), save_steps=10, task_type="binary_clf"
    )
    metric.update(
        inputs={
            "epk_id": torch.tensor([7, 8]),
            "target_attr_2": torch.tensor([1, 0]),
        },
        outputs=types.SimpleNamespace(logits=torch.zeros(2, 1)),
    )

    metric.flush()

    frame = pd.read_parquet(next(tmp_path.glob("*.parquet")))
    assert list(frame["epk_id"]) == [7, 8]
    assert list(frame["target_attr_2"]) == [1, 0]
    assert frame["prediction"].tolist() == pytest.approx([0.5, 0.5])


def test_an_unknown_task_type_is_rejected_by_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown task_type"):
        InferenceSupervisedMetrics(
            path_to_save=str(tmp_path), save_steps=1, task_type="ranking"
        )
