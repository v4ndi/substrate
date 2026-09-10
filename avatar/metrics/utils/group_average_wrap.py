"""Average already-computed metrics into summary numbers."""

import re
from statistics import mean

from avatar.metrics.base import BaseMetric, ScalarMetric


class GroupAverageMetricWrapper(ScalarMetric):
    """Add averages over the metrics the wrapped metric produced.

    Purely a post-processing step: ``update`` is forwarded untouched, and the
    averaging happens on the names returned by the inner ``compute``. Usually
    wrapped around
    :class:`~avatar.metrics.utils.group_devided_wrap.GroupDevidedMetricsWrapper`,
    whose per-group names are what there is to average.

    Args:
        metric: Any metric. Its ``compute`` output is both kept and averaged.
        groups: ``new name -> explicit list of metric names`` to average.
        avg_over_regulars: ``new name -> regular expression``; every produced
            metric name matching it is averaged. Resolved once, against the
            first ``compute``, and reused afterwards.

    Raises:
        AssertionError: A new name collides with a metric the inner metric
            already produces, or is defined in both ``groups`` and
            ``avg_over_regulars``.

    Note:
        A group that matches nothing averages to ``0``, not to a missing key.
        A regular expression that silently stops matching after a rename
        therefore shows up as a metric that went to zero.
    """

    def __init__(
        self,
        metric: BaseMetric,
        groups: dict[str, list[str]] | None = None,
        avg_over_regulars: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        self.metric = metric
        self.groups = groups if groups is not None else {}
        self.avg_over_regulars = avg_over_regulars
        self._init_groups_flag = False

    @property
    def required_inputs(self):
        """Whatever the wrapped metric needs — this wrapper reads nothing itself."""
        return self.metric.required_inputs

    @property
    def required_outputs(self):
        """Whatever the wrapped metric needs — this wrapper reads nothing itself."""
        return self.metric.required_outputs

    def init_groups(self, result):
        if self.avg_over_regulars is not None:
            for metric_name, regular in self.avg_over_regulars.items():
                assert metric_name not in result.keys(), (
                    f"provided new metric name:{metric_name} also exists in metrics"
                )
                assert metric_name not in self.groups.keys(), (
                    f"provided new metric name:{metric_name} exists both in avg_over_regulars and in groups"
                )
                self.groups[metric_name] = [
                    name for name in result.keys() if re.search(regular, name)
                ]
        self._init_groups_flag = True

    def update(self, inputs, outputs):
        self.metric.update(inputs, outputs)

    def compute(self):
        result = self.metric.compute()
        if not self._init_groups_flag:
            self.init_groups(result)
        if self.groups is None:  # epmty wrapper
            return result
        for metric_name, group in self.groups.items():
            assert metric_name not in result.keys(), (
                f"provided new metric name:{metric_name} also exists in metrics"
            )
            result[metric_name] = (
                mean([result[name] for name in group]) if len(group) > 0 else 0
            )
        return result

    def reset(self):
        self.metric.reset()
