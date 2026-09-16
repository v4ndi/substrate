"""Metrics for scalar regression models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from .base import MetricInput


@dataclass(frozen=True)
class RegressionMetric:
    """One registered scalar regression metric."""

    name: Literal["mse", "mae", "mape"]
    optimization_direction: Literal["minimize"] | None
    supported_tasks: frozenset[str] = frozenset({"regression"})

    def compute(self, data: MetricInput) -> float:
        if self.name == "mse":
            return float(mean_squared_error(data.target, data.scores))
        if self.name == "mae":
            return float(mean_absolute_error(data.target, data.scores))
        return float(mean_absolute_percentage_error(data.target, data.scores))
