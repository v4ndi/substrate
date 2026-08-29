import re
from statistics import mean

from avatar.metrics.base import BaseMetric


class GroupAverageMetricWrapper(BaseMetric):
    """GroupAverageMetricWrapper
    Wrap metric, after usual compute, compute average for groups of metrics

    metric (BaseMetric): any metric with methods update, compute, reset.
    gropus: (dict[str, list[str]]): new_metric_name -> list of metric, provided by metric to average
    avg_over_regulars: dict[str, str] - new_metric_name -> regular (regular means subset of metrics to average)
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
