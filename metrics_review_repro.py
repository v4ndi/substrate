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

import copy
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

import avatar
from avatar.metrics import ScalarMetric
from avatar.metrics.supervised import calculate_response_metrics
from avatar.metrics.utils import GroupAverageMetricWrapper, GroupDevidedMetricsWrapper
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


def b3_regression_key_collapse() -> None:
    """B3: operator precedence collapses every regression metric into one key."""
    print("\n--- B3  RegressionMetrics: схлопывание ключей ---")
    scores = {"mse": 1.0, "mae": 2.0, "mape": 3.0}
    task_name = "prod_a"
    collapsed = {
        f"{task_name}_" if len(task_name) != 0 else "" + key: value
        for key, value in scores.items()
    }
    print(f"  было: {scores}")
    print(f"  стало: {collapsed}   — ожидались три ключа, названные по метрикам")


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


def s12_reset_keeps_groups() -> None:
    """S12: reset() clears the inner metrics but not the group dictionary."""
    print("\n--- S12  GroupDevidedMetricsWrapper.reset не забывает группы ---")
    wrapper = GroupDevidedMetricsWrapper(
        metric_class=TwoGroupsOnSecondEpoch, columns_to_devide=["channel"]
    )
    wrapper.group2metrics["a",] = TwoGroupsOnSecondEpoch()
    wrapper.reset()
    print(f"  после reset() группы: {list(wrapper.group2metrics)}")
    print("  контракт требует состояния «как после создания метрики»")


def p5_deepcopy_on_train_outputs() -> None:
    """P5: the group wrapper deep-copies outputs, which fails on non-leaf tensors."""
    print("\n--- P5  GroupDevidedMetricsWrapper: deepcopy на train-выходе ---")
    leaf = torch.randn(4, 3, requires_grad=True)
    try:
        copy.deepcopy(Output(logits=leaf * 2))
        print("  deepcopy: OK")
    except RuntimeError as error:
        print(f"  deepcopy -> RuntimeError: {str(error).splitlines()[0]}")


def main() -> None:
    """Run every reproduction in the order the review lists them."""
    print(f"avatar: {avatar.__file__}")
    b3_regression_key_collapse()
    s4_nan_erases_the_best_score()
    s5_frozen_regex_groups()
    s12_reset_keeps_groups()
    p5_deepcopy_on_train_outputs()


if __name__ == "__main__":
    main()
