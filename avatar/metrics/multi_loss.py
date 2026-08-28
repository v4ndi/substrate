from collections import defaultdict

import numpy as np

from .base import BaseMetric


class MultiLossMetric(BaseMetric):
    def __init__(self):
        self.losses = defaultdict(list)
        self.num_items = defaultdict(list)

    def update(self, inputs, outputs) -> None:
        losses = outputs.losses

        for attribute, loss in losses.items():
            self.losses[attribute].append(loss.detach().cpu().item())
            self.num_items[attribute].append(outputs.num_items[attribute].cpu().item())

    def compute(self) -> dict[str, float]:
        for attribute in self.losses.keys():
            if self.num_items[attribute] != 0:
                self.losses[attribute] = (
                    np.array(self.losses[attribute])
                    / np.array(self.num_items[attribute])
                ).mean()
            else:
                self.losses[attribute] = 0.0
        return self.losses

    def reset(self) -> None:
        self.losses = defaultdict(list)
        self.num_items = defaultdict(list)
