"""Split a batch by column values and score each slice separately."""

import copy
import dataclasses

import numpy as np
import torch

from avatar.metrics.base import ScalarMetric


def slice_value(value, positions: np.ndarray, batch_size: int):
    """Take ``positions`` out of one batch column, whatever it is made of.

    Anything whose leading dimension is not the batch size is passed through
    untouched — a scalar loss, a per-task constant, a string tag. Slicing those
    would be meaningless, and dropping them (which is what the previous
    type-by-type branching did to numpy columns) loses data the inner metric
    may well need.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            key: slice_value(inner, positions, batch_size)
            for key, inner in value.items()
        }
    if isinstance(value, torch.Tensor):
        if value.dim() == 0 or value.shape[0] != batch_size:
            return value
        return value[torch.as_tensor(positions, device=value.device)]
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != batch_size:
            return value
        return value[positions]
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            return value
        return type(value)(value[position] for position in positions)
    return value


def output_fields(outputs) -> list[str]:
    """Names of the fields on a model output object."""
    if dataclasses.is_dataclass(outputs):
        return [field.name for field in dataclasses.fields(outputs)]
    return list(vars(outputs))


class GroupDevidedMetricsWrapper(ScalarMetric):
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
        ``columns_desc=["channel", "group"]`` wrapping ``ResponseMetrics`` gives
        names like ``channel_0_group_1_calib_group_0_roc_auc_score``. Those
        names are what
        :class:`~avatar.metrics.utils.group_average_wrap.GroupAverageMetricWrapper`
        then averages over.

    Note:
        Groups are discovered from the data, so a combination absent from
        validation simply produces no metric — the name will be missing rather
        than zero. :meth:`reset` drops the groups along with their metrics, so
        each epoch rediscovers them; a group that has left the data leaves with
        it instead of reaching ``compute`` with an empty accumulator.

        The inner metric is a factory, so this wrapper cannot ask it which
        fields it reads before the first group appears. Both
        ``required_inputs`` and ``required_outputs`` therefore stay ``None`` and
        a distributed run gathers the whole batch for it.
    """

    required_inputs = None
    required_outputs = None

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
        """Cut both sides of the batch down to the masked records.

        The output object is copied shallowly and every field replaced by its
        slice. It used to be ``copy.deepcopy(outputs)`` with only ``logits``
        replaced, which meant a metric reading any other field got the whole
        batch back — and a deepcopy of a training step's output raises, because
        those tensors are not graph leaves.
        """
        positions = np.where(mask)[0]
        batch_size = len(mask)

        current_inputs = {
            key: slice_value(value, positions, batch_size)
            for key, value in inputs.items()
        }

        current_outputs = copy.copy(outputs)
        for name in output_fields(outputs):
            setattr(
                current_outputs,
                name,
                slice_value(getattr(outputs, name), positions, batch_size),
            )

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
        """Return to the state of a freshly built wrapper.

        The groups go with their metrics. Keeping them meant a group that
        disappeared between epochs survived into the next ``compute`` with an
        empty accumulator, where the inner metric raises.
        """
        self.group2metrics = {}
