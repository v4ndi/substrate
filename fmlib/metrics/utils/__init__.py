"""Metric wrappers: split a population into groups, then average back up.

The two compose in one direction — ``GroupDevidedMetricsWrapper`` produces the
per-group names, ``GroupAverageMetricWrapper`` averages over them.
"""

from .group_average_wrap import GroupAverageMetricWrapper
from .group_devided_wrap import GroupDevidedMetricsWrapper

__all__ = ["GroupAverageMetricWrapper", "GroupDevidedMetricsWrapper"]
