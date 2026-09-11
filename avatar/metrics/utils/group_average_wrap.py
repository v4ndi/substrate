"""Average already-computed metrics into summary numbers."""

import logging
import re
from statistics import mean

from avatar.metrics.base import BaseMetric, ScalarMetric

logger = logging.getLogger(__name__)


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
            metric name matching it is averaged. Resolved against **every**
            ``compute``, because the inner wrapper discovers its groups from
            the data: a group that first appears in the second epoch used to be
            left out of the average for the rest of the run, and nothing in the
            log said so.

    Raises:
        AssertionError: A new name collides with a metric the inner metric
            already produces, or is defined in both ``groups`` and
            ``avg_over_regulars``.

    Note:
        A group that matches nothing produces **no key**, and the reason is
        logged. It used to average to ``0``, so a regular expression that
        stopped matching after a rename looked like a metric that collapsed.
    """

    def __init__(
        self,
        metric: BaseMetric,
        groups: dict[str, list[str]] | None = None,
        avg_over_regulars: dict[str, str] | None = None,
    ):
        super().__init__()
        self.metric = metric
        self.groups = groups if groups is not None else {}
        self.avg_over_regulars = avg_over_regulars

    @property
    def required_inputs(self):
        """Whatever the wrapped metric needs — this wrapper reads nothing itself."""
        return self.metric.required_inputs

    @property
    def required_outputs(self):
        """Whatever the wrapped metric needs — this wrapper reads nothing itself."""
        return self.metric.required_outputs

    def resolve_groups(self, result: dict) -> dict[str, list[str]]:
        """The explicit groups plus the regular expressions, against *this* result.

        Nothing is cached: the inner metric may have produced names this epoch
        that did not exist last epoch, and those belong in the average.
        """
        groups = {name: list(members) for name, members in self.groups.items()}
        for metric_name, regular in (self.avg_over_regulars or {}).items():
            assert metric_name not in result.keys(), (
                f"provided new metric name:{metric_name} also exists in metrics"
            )
            assert metric_name not in self.groups.keys(), (
                f"provided new metric name:{metric_name} exists both in "
                f"avg_over_regulars and in groups"
            )
            groups[metric_name] = [
                name for name in result.keys() if re.search(regular, name)
            ]
        return groups

    def update(self, inputs, outputs):
        self.metric.update(inputs, outputs)

    def compute(self):
        result = self.metric.compute()
        for metric_name, group in self.resolve_groups(result).items():
            assert metric_name not in result.keys(), (
                f"provided new metric name:{metric_name} also exists in metrics"
            )
            members = [name for name in group if name in result]
            missing = [name for name in group if name not in result]
            if missing:
                logger.warning(
                    "%s averages over %d name(s) the metric did not produce this "
                    "time (%s)",
                    metric_name,
                    len(missing),
                    ", ".join(missing[:5]),
                )
            if not members:
                logger.warning("%s matched nothing and is not reported", metric_name)
                continue
            result[metric_name] = mean([result[name] for name in members])
        return result

    def reset(self):
        self.metric.reset()
