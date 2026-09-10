"""Reproduce the confirmed defects listed in ``metrics_review.md``.

Run it from the repository root with the interpreter that has ``avatar`` importable::

    python metrics_review_repro.py

Every block prints the value the module actually produces next to the value the
docstring or the contract promises. Nothing is asserted: the point is to make the
review checkable in one command, not to fail a build.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

import avatar
from avatar.metrics.base import BaseMetric
from avatar.metrics.classification import RocAucScore
from avatar.metrics.loss_logging import UniversalLossesMetric
from avatar.metrics.moe_reg import Entropy
from avatar.metrics.multi_loss import MultiLossMetric
from avatar.metrics.utils import GroupAverageMetricWrapper

assert avatar.__file__.startswith(os.getcwd()), (
    f"avatar came from {avatar.__file__}, not from {os.getcwd()}"
)


@dataclass
class Output:
    """Stand-in for a pipeline output dataclass."""

    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None
    num_items: dict[str, torch.Tensor] | None = None


class TwoGroupsOnSecondEpoch(BaseMetric):
    """A metric whose group set grows between epochs, as lazy group discovery does."""

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


def s1_entropy_sign() -> None:
    """S1: normalised entropy comes back negated."""
    print("\n--- S1  Entropy: знак ---")
    entropy = Entropy()
    uniform = entropy.calc_normalized_entropy(np.full(4, 0.25))
    collapsed = entropy.calc_normalized_entropy(np.array([1.0, 1e-12, 1e-12, 1e-12]))
    print(f"  равномерно по 4 экспертам -> {uniform:+.4f}   докстринг обещает ~ +1")
    print(f"  коллапс на одного         -> {collapsed:+.4f}   докстринг обещает ~  0")


def s2_multi_loss() -> None:
    """S2: mean of ratios instead of ratio of sums, and compute() mutates state."""
    print("\n--- S2  MultiLossMetric: накопитель и идемпотентность ---")
    metric = MultiLossMetric()
    for loss, items in [(2.0, 4.0), (4.0, 2.0)]:
        metric.update(
            None,
            Output(
                losses={"a": torch.tensor(loss)}, num_items={"a": torch.tensor(items)}
            ),
        )
    print(f"  compute() первый раз  -> {dict(metric.compute())}")
    print(f"  compute() второй раз  -> {dict(metric.compute())}")
    print(f"  корректное sum/sum    -> {(2.0 + 4.0) / (4.0 + 2.0)}")


def s3_universal_losses() -> None:
    """S3: an intermittent loss is divided by the global batch counter."""
    print("\n--- S3  UniversalLossesMetric: делитель ---")
    metric = UniversalLossesMetric()
    for step in range(4):
        metric.update(
            None,
            Output(
                loss=torch.tensor(1.0),
                aux_loss=torch.tensor(1.0) if step == 3 else None,
            ),
        )
    print(f"  {dict(metric.compute())}")
    print("  aux_loss встречался один раз и был равен 1.0; докстринг обещает 1.0")


def s4_roc_auc_nan() -> None:
    """S4: an all-positive slice yields nan, which then breaks EarlyStopping."""
    print("\n--- S4  RocAucScore: срез из одних позитивов ---")
    metric = RocAucScore()
    metric.update(
        {"targets": torch.tensor([1.0, 1.0, 1.0])},
        Output(logits=torch.tensor([0.1, 0.9, 0.5])),
    )
    print(f"  compute() -> {metric.compute()}")
    best = float("nan")
    print(f"  EarlyStopping: score < best + delta  ->  {0.9 < best + 0.0}")
    print("  nan в любом сравнении даёт False, поэтому любой скор считается рекордом")


def s5_frozen_regex_groups() -> None:
    """S5: regex groups are resolved once and never revisited."""
    print("\n--- S5  GroupAverageMetricWrapper: группы фиксируются ---")
    wrapper = GroupAverageMetricWrapper(
        TwoGroupsOnSecondEpoch(), avg_over_regulars={"avg_auc": r"g_\d+_auc"}
    )
    print(f"  эпоха 1 -> {wrapper.compute()}")
    print(f"  эпоха 2 -> {wrapper.compute()}")
    print("  g_1_auc появился, но в avg_auc не попал")


def b1_collectors_return_none() -> None:
    """B1: a collector's compute() returns None and the loop calls scores.update on it."""
    print("\n--- B1  compute() -> None ломает scores.update ---")
    try:
        {}.update(None)
    except TypeError as error:
        print(f"  scores.update(None) -> TypeError: {error}")


def b3_regression_key_collapse() -> None:
    """B3: operator precedence collapses every regression metric into one key."""
    print("\n--- B3  RegressionMetrics: схлопывание ключей ---")
    scores = {"mse": 1.0, "mae": 2.0, "mape": 3.0}
    task_name = "prod_a"
    collapsed = {
        f"{task_name}_" if len(task_name) != 0 else "" + key: value
        for key, value in scores.items()
    }
    print(f"  {scores}")
    print(f"  ->  {collapsed}")


def p5_deepcopy_on_train_outputs() -> None:
    """P5: the group wrapper deep-copies outputs, which fails on non-leaf tensors."""
    print("\n--- P5  GroupDevidedMetricsWrapper: deepcopy на train-выходе ---")
    import copy

    leaf = torch.randn(4, 3, requires_grad=True)
    try:
        copy.deepcopy(Output(logits=leaf * 2))
        print("  deepcopy: OK")
    except RuntimeError as error:
        print(f"  deepcopy -> RuntimeError: {str(error).splitlines()[0]}")


def main() -> None:
    """Run every reproduction in the order the review lists them."""
    print(f"avatar: {avatar.__file__}")
    b1_collectors_return_none()
    b3_regression_key_collapse()
    s1_entropy_sign()
    s2_multi_loss()
    s3_universal_losses()
    s4_roc_auc_nan()
    s5_frozen_regex_groups()
    p5_deepcopy_on_train_outputs()


if __name__ == "__main__":
    main()
