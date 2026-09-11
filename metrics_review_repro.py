"""Reproduce the defects still open in ``avatar.metrics``.

Run it from the repository root with the interpreter that has ``avatar``
importable::

    python metrics_review_repro.py

Everything the review found and the purge fixed is covered by ``tests/metrics``
and ``tests/train/test_metric_fields.py`` instead — a test that fails is worth
more than a script that prints. What is left here is the open list: each block
prints the value the module actually produces next to the value it should.
Nothing is asserted; the point is to make the open findings checkable in one
command.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

import avatar
from avatar.metrics import ScalarMetric
from avatar.metrics.supervised import calculate_response_metrics
from avatar.metrics.utils import GroupAverageMetricWrapper
from avatar.train.early_stopping import EarlyStopping

assert avatar.__file__.startswith(os.getcwd()), (
    f"avatar came from {avatar.__file__}, not from {os.getcwd()}"
)


@dataclass
class Output:
    """Stand-in for a pipeline output dataclass."""

    logits: torch.Tensor | None = None


class TwoGroupsOnSecondEpoch(ScalarMetric):
    """A metric whose group set grows between epochs, as lazy discovery does."""

    def __init__(self):
        self.epoch = 0

    def update(self, inputs, outputs) -> None:
        """Accept and ignore a batch."""

    def compute(self) -> dict[str, float]:
        """Return one group on the first epoch and two on the second."""
        self.epoch += 1
        scores = {"g_0_auc": 1.0}
        if self.epoch > 1:
            scores["g_1_auc"] = 0.0
        return scores

    def reset(self) -> None:
        """Nothing to reset."""


def s4_nan_erases_the_best_score() -> None:
    """S4: a one-class slice yields nan, and a nan wipes the early-stopping record."""
    print("\n--- S4  nan из вырожденного среза стирает рекорд ---")
    degenerate = calculate_response_metrics(np.ones(3), np.array([0.1, 0.9, 0.5]))
    print(f"  срез из одних позитивов -> roc_auc_score={degenerate['roc_auc_score']}")

    for label, scores in (
        ("nan есть", [0.9, float("nan"), 0.1, 0.1, 0.1, 0.1]),
        ("nan нет ", [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ):
        stopper = EarlyStopping(main_metric="m", patience=2, strategy="max")
        for score in scores:
            stopper({"m": score})
        print(
            f"  {label} best={stopper.best_score!r:>5}  early_stop={stopper.early_stop}"
        )
    print("  один nan забывает рекорд 0.9, и просадка проходит незамеченной")


def s5_frozen_regex_groups() -> None:
    """S5: regex groups are resolved once and never revisited."""
    print("\n--- S5  GroupAverageMetricWrapper: группы фиксируются ---")
    wrapper = GroupAverageMetricWrapper(
        TwoGroupsOnSecondEpoch(), avg_over_regulars={"avg_auc": r"g_\d+_auc"}
    )
    print(f"  эпоха 1 -> {wrapper.compute()}")
    print(f"  эпоха 2 -> {wrapper.compute()}")
    print("  g_1_auc появился, но в avg_auc не попал")


def main() -> None:
    """Run every reproduction in the order the review lists them."""
    print(f"avatar: {avatar.__file__}")
    s4_nan_erases_the_best_score()
    s5_frozen_regex_groups()


if __name__ == "__main__":
    main()
