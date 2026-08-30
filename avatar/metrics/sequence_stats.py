"""Diagnostics for sequence models: what went in, and how big the vectors are.

These do not measure quality. They answer "is the data what I think it is" —
how long the contexts actually are after slicing, how the event types are
distributed, whether hidden states are drifting in scale.
"""

import numpy as np
import torch

from avatar.outputs import SequenceOutput

from .base import BaseMetric


class SequenceStats(BaseMetric):
    """A class to compute statistics for sequence data, such as mean context length and event ratio.

    Attributes:
        _context_lengths (list): List of total lengths of contexts accumulated over batches.
        _events_lengths (dict): Dictionary mapping event IDs to a list of their corresponding lengths.
        _batch_size (int): The size of the current batch.
    """

    def __init__(self):
        """Initialize the SequenceStats instance with empty lists and None values."""
        self._context_lengths = []
        self._events_lengths = {}
        self._batch_size = None

    def update(self, inputs, outputs) -> None:
        """Update internal statistics with new data from a batch."""
        seq_features = inputs["seq_features"]
        event_ids = seq_features.event_ids
        if self._batch_size is None:
            self._batch_size = event_ids.size(0)

        event_ids = event_ids.unique().detach().cpu().numpy()
        event_ids = event_ids[event_ids >= 0]
        self._context_lengths.append(float(seq_features.seq_len.sum().detach().cpu()))

        for event_id in event_ids:
            event_stat = float(seq_features.num_items(event_id).sum().detach().cpu())
            if event_id in self._events_lengths:
                self._events_lengths[event_id].append(event_stat)
            else:
                self._events_lengths[event_id] = [event_stat]

    def compute(self) -> dict[str, float]:
        """Compute and return statistics based on the collected data.

        Returns:
            dict[str, float]: A dictionary with computed statistics, including mean context length
            and normalized event counts.
        """
        if not self._context_lengths:
            return {}
        total_context_length = sum(self._context_lengths)
        mean_context_length = total_context_length / (
            len(self._context_lengths) * self._batch_size
        )
        event_stats = {
            f"event_{key}": (sum(self._events_lengths[key]) / total_context_length)
            if total_context_length > 0
            else 0.0
            for key in self._events_lengths.keys()
        }
        event_stats["mean_context_length"] = mean_context_length
        print(event_stats)
        return event_stats

    def reset(self) -> None:
        """Reset all internal state variables to prepare for a new round of stats collection."""
        self._context_lengths.clear()
        for key in self._events_lengths.keys():
            self._events_lengths[key].clear()
        self._batch_size = None
        self._event_ids = None


class ExpertsWorkload(BaseMetric):
    """Mean per-layer entropy of the router's expert distribution.

    Reads ``outputs.router_logits`` and reports how evenly traffic is spread
    over experts, averaged across layers.

    Args:
        epsilon: Guard added inside the logarithm so an unused expert does not
            produce ``-inf``.

    Warning:
        Unlike every other metric here, :meth:`compute` returns a **scalar**,
        not a ``{name: value}`` dict, so it cannot be used directly as a
        ``valid_metrics`` entry — the evaluation loop does
        ``scores.update(metric.compute())``. No config uses it today. Wrap it
        or fix the return type before putting it in one.
    """

    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon

        self.reset()

    def update(self, inputs, outputs) -> None:
        assert hasattr(outputs, "router_logits") is not None, (
            "Output must have router_logits"
        )
        router_logits = outputs.router_logits
        logits_array = np.array([i.detach().cpu().numpy() for i in router_logits]).sum(
            axis=1
        )
        self._experts_workload.append(logits_array)

    def compute(self) -> dict[str, float]:
        total_logits = np.array(self._experts_workload).sum(axis=0)
        norm_logits = (total_logits.T / total_logits.sum(axis=-1)).T
        norm_entropy = (norm_logits * np.log(norm_logits + self.epsilon)).sum(
            axis=-1
        ) / norm_logits.shape[1]
        return np.mean(-(norm_entropy))

    def reset(self) -> None:
        self._experts_workload = []


class HiddensNorm(BaseMetric):
    """Running mean of the L1, L2 and L-infinity norms of the pooled embedding.

    Reads ``outputs.aggregated_hidden_state``. Useful as a drift alarm: a
    representation whose norm climbs steadily across epochs is usually a sign
    of a missing normalisation layer rather than of learning.

    Returns from :meth:`compute`:
        ``l1_norm``, ``l2_norm``, ``l_inf_norm``.
    """

    def __init__(self):
        self.reset()
        pass

    def _update_metric(self, metric_name: str, new_value: float):
        self.norms[metric_name] = (
            self.norms[metric_name] * self.update_counter + new_value
        ) / (self.update_counter + 1)

    def _calc_norms(self, hidden_state: torch.Tensor):
        l1_norm = hidden_state.abs().sum(dim=-1).mean()
        l2_norm = hidden_state.norm(p=2, dim=-1).mean()
        l_inf_norm = hidden_state.abs().max(dim=-1).values.mean()
        for metric_name, value in zip(
            ["l1_norm", "l2_norm", "l_inf_norm"],
            [l1_norm, l2_norm, l_inf_norm],
            strict=False,
        ):
            self._update_metric(metric_name, value.detach().cpu().item())

    def update(self, inputs, outputs: SequenceOutput):
        self.update_counter += 1
        self._calc_norms(outputs.aggregated_hidden_state)

    def compute(self) -> dict[str, float]:
        return self.norms.copy()

    def reset(self) -> None:
        self.update_counter = 0
        self.norms = {"l1_norm": 0, "l2_norm": 0, "l_inf_norm": 0}
