"""What early stopping does with a score it cannot compare.

Every comparison against ``nan`` is false, so the plain implementation took one
for an improvement: it overwrote the best score, reset the patience counter
and — since the checkpoint callback saves whenever that counter is zero — wrote
a checkpoint labelled best. One undefined epoch was enough to erase the record
a later collapse should have been measured against.
"""

import logging

import pytest

from fmlib.train import EarlyStopping


def feed(stopper, scores):
    for score in scores:
        stopper({"m": score})
    return stopper


def test_a_nan_does_not_become_the_record():
    """The 0.9 stands, so the drop to 0.1 is still a drop."""
    stopper = feed(
        EarlyStopping(main_metric="m", patience=2, strategy="max"),
        [0.9, float("nan"), 0.1, 0.1],
    )

    assert stopper.best_score == pytest.approx(0.9)
    assert stopper.early_stop


def test_a_nan_counts_as_no_improvement():
    """A run whose metric is permanently undefined still ends."""
    stopper = feed(
        EarlyStopping(main_metric="m", patience=3, strategy="max"),
        [0.5] + [float("nan")] * 3,
    )

    assert stopper.counter == 3
    assert stopper.early_stop


def test_a_nan_does_not_ask_for_a_checkpoint():
    """``counter == 0`` is what the checkpoint callback reads as "new best"."""
    stopper = feed(
        EarlyStopping(main_metric="m", patience=5, strategy="max"),
        [0.9, float("nan")],
    )

    assert stopper.counter == 1


def test_a_missing_metric_is_survivable(caplog):
    """A metric that could not be computed omits its key rather than lying."""
    stopper = EarlyStopping(main_metric="m", patience=2, strategy="max")
    stopper({"m": 0.9})
    with caplog.at_level(logging.WARNING, logger="fmlib.train.early_stopping"):
        stopper({"other": 0.1})

    assert stopper.best_score == pytest.approx(0.9)
    assert stopper.counter == 1
    assert any("not among the reported metrics" in r.message for r in caplog.records)


def test_none_is_treated_like_nan():
    """Whatever cannot be compared counts the same way."""
    stopper = feed(
        EarlyStopping(main_metric="m", patience=2, strategy="max"), [0.9, None]
    )

    assert stopper.best_score == pytest.approx(0.9)
    assert stopper.counter == 1


def test_a_real_improvement_still_resets_the_counter():
    """The ordinary path is unchanged."""
    stopper = feed(
        EarlyStopping(main_metric="m", patience=2, strategy="max"),
        [0.1, 0.2, float("nan"), 0.9],
    )

    assert stopper.best_score == pytest.approx(0.9)
    assert stopper.counter == 0
    assert not stopper.early_stop


def test_minimisation_is_unchanged():
    """``strategy="min"`` still stops when the loss stops falling."""
    stopper = feed(
        EarlyStopping(main_metric="m", patience=2, strategy="min"),
        [1.0, 0.5, 0.6, 0.7],
    )

    assert stopper.best_score == pytest.approx(-0.5)
    assert stopper.early_stop
