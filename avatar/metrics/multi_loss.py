"""Report the individual components of a composite loss.

A multi-head model returns one loss per head; the trainer only ever sees their
sum. This metric surfaces the parts, which is what tells you *which* head
stopped learning.
"""

from collections import defaultdict

from .base import ScalarMetric


class MultiLossMetric(ScalarMetric):
    """Token-weighted mean of each loss component reported by the model.

    Consumes ``outputs.losses`` and ``outputs.num_items`` — the per-component
    dictionaries produced by :class:`~avatar.losses.base.Loss`.

    Each component is the **sum of its losses over the sum of its item counts**,
    not the mean of the per-batch ratios. The two differ whenever batches carry
    different numbers of valid targets, which is exactly the case this metric
    exists for: on batches ``(loss=2, n=4)`` and ``(loss=4, n=2)`` the mean of
    ratios gives 1.25 and the ratio of sums gives the correct 1.0.

    Returns from :meth:`compute`:
        One entry per component, keyed by the name the model used.
    """

    required_inputs = ()
    required_outputs = ("losses", "num_items")

    def __init__(self):
        self.reset()

    def update(self, inputs, outputs) -> None:
        """Add one batch's losses and item counts to the running totals."""
        for attribute, loss in outputs.losses.items():
            self.loss_sums[attribute] += loss.detach().cpu().item()
            self.item_counts[attribute] += outputs.num_items[attribute].cpu().item()

    def compute(self) -> dict[str, float]:
        """Divide each component's total loss by its total item count.

        Leaves the accumulators alone, so calling it twice gives the same
        answer.
        """
        return {
            attribute: (
                self.loss_sums[attribute] / self.item_counts[attribute]
                if self.item_counts[attribute]
                else 0.0
            )
            for attribute in self.loss_sums
        }

    def reset(self) -> None:
        """Clear the running totals."""
        self.loss_sums: defaultdict[str, float] = defaultdict(float)
        self.item_counts: defaultdict[str, float] = defaultdict(float)
