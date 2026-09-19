"""How often a trial validates, asserted as a number.

The bug this covers was found on a two-rank run and looked like a crash: the
trial trained, never validated once, and came back with no metric — which is
indistinguishable from a failure. The cause was a cadence computed over the
whole corpus while the loop counts steps *per rank*, so on two ranks the
interval was longer than the epoch.

The regression test that followed asserted the metric was not ``None``. That
catches the symptom on the one shape it was found in. This asserts the property
instead: **the cadence never exceeds one per-rank epoch**, on any shape.
"""

from __future__ import annotations

import math

import pytest

from fmlib.automl.backends.tabnn.assembly import _steps_before_evaluation


def _steps_per_rank_epoch(rows: int, batch_size: int, world_size: int) -> int:
    return max(1, math.ceil(max(1, math.ceil(rows / world_size)) / batch_size))


@pytest.mark.parametrize("rows", [1, 7, 800, 4000, 3_000_000, 100_000_000])
@pytest.mark.parametrize("batch_size", [1, 64, 4096])
@pytest.mark.parametrize("world_size", [1, 2, 8, 64])
@pytest.mark.parametrize("per_epoch", [1, 4, 100])
def test_the_cadence_fits_inside_one_per_rank_epoch(
    rows, batch_size, world_size, per_epoch
):
    """A trial that finishes an epoch must have validated at least once."""
    cadence = _steps_before_evaluation(rows, batch_size, per_epoch, world_size)
    assert cadence >= 1
    assert cadence <= _steps_per_rank_epoch(rows, batch_size, world_size), (
        f"rows={rows} batch={batch_size} ranks={world_size}: a trial would "
        "finish an epoch without validating, and report no metric"
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_more_ranks_never_make_the_cadence_longer(world_size):
    """Adding ranks shortens the per-rank epoch, so it cannot lengthen the wait.

    This is the direction the bug went wrong in: the cadence was computed from
    a number that does not shrink with the rank count, so it stayed put while
    the epoch it had to fit inside got shorter.
    """
    rows, batch_size, per_epoch = 4_000_000, 4096, 4
    single = _steps_before_evaluation(rows, batch_size, per_epoch, 1)
    many = _steps_before_evaluation(rows, batch_size, per_epoch, world_size)
    assert many <= single


def test_the_cadence_asks_for_about_the_requested_number_of_validations():
    """On data large enough for the ratio to mean anything, it is honoured."""
    rows, batch_size, world_size = 100_000_000, 4096, 8
    for per_epoch in (1, 2, 5, 10):
        cadence = _steps_before_evaluation(rows, batch_size, per_epoch, world_size)
        epoch = _steps_per_rank_epoch(rows, batch_size, world_size)
        assert round(epoch / cadence) == pytest.approx(per_epoch, abs=1)
