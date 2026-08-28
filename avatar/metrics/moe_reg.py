from collections import defaultdict

import numpy as np

from .base import BaseMetric


class Entropy(BaseMetric):
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
        """return Dict(metric_name: value)"""

        result = {}
        for task_name, weights_list in self.task_to_weights.items():
            result[task_name + "_normalized_entropy"] = self.calc_normalized_entropy(
                np.mean(weights_list, axis=0)
            )

        return result

    def reset(self):
        self.task_to_weights = defaultdict(list)


class Importance(BaseMetric):
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
        """return Dict(metric_name: value)"""

        result = {}
        for task_name, weights_list in self.task_to_weights.items():
            result[task_name + "_importance"] = self.calc_importance(
                np.mean(weights_list, axis=0)
            )

        return result

    def reset(self):
        self.task_to_weights = defaultdict(list)
