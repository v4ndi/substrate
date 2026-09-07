"""Diagnostics for mixture-of-experts gating.

Both metrics read ``outputs.task_gated_weights`` — the per-task distribution
over experts — and answer the same question from two directions: is the gate
actually routing, or has it collapsed onto one expert?
"""

from collections import defaultdict

import numpy as np

from .base import BaseMetric


class Entropy(BaseMetric):
    """Normalised entropy of each task's average gating distribution.

    Close to 1 means the task spreads evenly over experts; close to 0 means it
    has collapsed onto a single one.

    Returns from :meth:`compute`:
        ``{task}_normalized_entropy`` per task.
    """

    def __init__(self):
        self.reset()

    def update(self, inputs, outputs):
        for task_name in outputs.task_gated_weights.keys():
            self.task_to_weights[task_name].append(
                outputs.task_gated_weights[task_name].detach().cpu().numpy()
            )

    def calc_normalized_entropy(self, arr) -> float:
        return (np.sum(arr * np.log(arr)) / np.log(len(arr))).item()

    def compute(self):
        """Return the accumulated statistic per task."""
        result = {}
        for task_name, weights_list in self.task_to_weights.items():
            result[task_name + "_normalized_entropy"] = self.calc_normalized_entropy(
                np.mean(weights_list, axis=0)
            )

        return result

    def reset(self):
        self.task_to_weights = defaultdict(list)


class Importance(BaseMetric):
    """Squared coefficient of variation of each task's gating weights.

    The load-balancing statistic from the MoE literature: variance over squared
    mean. 0 means every expert is used equally; large values mean a few experts
    take most of the traffic. The complement of :class:`Entropy` — same input,
    penalises imbalance rather than rewarding spread.

    Returns from :meth:`compute`:
        ``{task}_importance`` per task.
    """

    def __init__(self):
        self.reset()

    def update(self, inputs, outputs):
        for task_name in outputs.task_gated_weights.keys():
            self.task_to_weights[task_name].append(
                outputs.task_gated_weights[task_name].detach().cpu().numpy()
            )

    def calc_importance(self, arr) -> float:
        return np.var(arr).item() / (np.mean(arr) ** 2).item()

    def compute(self):
        """Return the accumulated statistic per task."""
        result = {}
        for task_name, weights_list in self.task_to_weights.items():
            result[task_name + "_importance"] = self.calc_importance(
                np.mean(weights_list, axis=0)
            )

        return result

    def reset(self):
        self.task_to_weights = defaultdict(list)
