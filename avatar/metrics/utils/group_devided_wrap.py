"""Split a batch by column values and score each slice separately."""

import copy

import numpy as np
import torch

from avatar.metrics.base import BaseMetric


class GroupDevidedMetricsWrapper(BaseMetric):
    """Compute one metric independently for every combination of column values.

    Each batch is masked into slices — one per observed combination — and each
    slice is fed to its **own** metric instance. Instances are created lazily,
    the first time a combination is seen, which is why ``metric_class`` must be
    a factory rather than an object.

    Args:
        metric_class: Callable returning a fresh :class:`BaseMetric`. From
            Hydra this means the metric block carries ``_partial_: true``;
            without it every group would share one accumulator and the scores
            would silently be wrong.
        columns_to_devide: Columns of ``inputs`` whose values define the groups.
        columns_desc: Human-readable names used when building metric names;
            defaults to ``columns_to_devide``.

    Returns from :meth:`compute`:
        ``{desc}_{value}_..._{inner metric name}`` for every combination — so
        ``columns_desc=["channel", "group"]`` wrapping ``RocAucScore`` gives
        names like ``channel_0_group_1_roc_auc_score``. Those names are what
        :class:`~avatar.metrics.utils.group_average_wrap.GroupAverageMetricWrapper`
        then averages over.

    Note:
        Groups are discovered from the data, so a combination absent from
        validation simply produces no metric — the name will be missing rather
        than zero.
    """

    def __init__(
        self,
        metric_class,  # partial
        columns_to_devide: list[str],
        columns_desc: list[str] | None = None,
    ):
        if columns_desc is None:
            columns_desc = columns_to_devide
        self.columns_to_devide = columns_to_devide
        self.columns_desc = columns_desc
        self.metric_class = metric_class
        self.group2metrics = {}

    def _get_mask(self, inputs, groups_tuple: tuple):
        mask = None
        for column, value in zip(self.columns_to_devide, groups_tuple, strict=False):
            current_mask = np.array(inputs[column]) == value
            mask = mask & current_mask if mask is not None else current_mask
        return mask

    def _build_input_output_by_mask(self, inputs, outputs, mask):
        device = outputs.logits.device
        ids = torch.tensor(np.where(mask)[0]).to(device)
        current_inputs = {}
        current_outputs = copy.deepcopy(outputs)
        for key in inputs.keys():
            if isinstance(inputs[key], dict):
                current_inputs[key] = {}
                for inner_key in inputs[key].keys():
                    current_inputs[key][inner_key] = inputs[key][inner_key][ids]
            elif isinstance(inputs[key], list):
                current_inputs[key] = [inputs[key][i] for i in ids]
            elif isinstance(inputs[key], torch.Tensor):
                current_inputs[key] = inputs[key][ids]
        current_outputs.logits = outputs.logits[ids]

        return current_inputs, current_outputs

    def _update_current_groups(self, inputs, outputs, groups_tuple: tuple):
        mask = self._get_mask(inputs, groups_tuple)
        current_inputs, current_outputs = self._build_input_output_by_mask(
            inputs, outputs, mask
        )
        if groups_tuple not in self.group2metrics.keys():
            self.group2metrics[groups_tuple] = self.metric_class()
        self.group2metrics[groups_tuple].update(current_inputs, current_outputs)

    def update(self, inputs, outputs):
        unique_combinations = set(
            zip(*([inputs[column] for column in self.columns_to_devide]), strict=False)
        )
        for groups_tuple in unique_combinations:
            self._update_current_groups(inputs, outputs, groups_tuple)

    def _build_metric_name(self, groups_tuple, suff=""):
        name = ""
        for column_desc, value in zip(self.columns_desc, groups_tuple, strict=False):
            name += f"{column_desc}_{value}_"
        name = name + suff
        return name

    def compute(self):
        result = {}
        for key, metric in self.group2metrics.items():
            metric_result = metric.compute()
            for inner_metric_name, inner_metric_value in metric_result.items():
                result[self._build_metric_name(key, inner_metric_name)] = (
                    inner_metric_value
                )
        return result

    def reset(self):
        for metric in self.group2metrics.values():
            metric.reset()
