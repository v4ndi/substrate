"""Report the individual components of a composite loss.

A multi-head model returns one loss per head; the trainer only ever sees their
sum. This metric surfaces the parts, which is what tells you *which* head
stopped learning.
"""

from collections import defaultdict

import numpy as np

from .base import BaseMetric


class MultiLossMetric(BaseMetric):
    """Token-weighted mean of each loss component reported by the model.

    Consumes ``outputs.losses`` and ``outputs.num_items`` — the per-component
    dictionaries produced by :class:`~avatar.losses.base.Loss`. Each component
    is divided by its own item count, so heads with different numbers of valid
    targets stay comparable.

    Returns from :meth:`compute`:
        One entry per component, keyed by the name the model used.
    """

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
