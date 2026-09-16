"""Public structural contract for AutoML metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class MetricInput:
    """Task-level arrays consumed by a registered metric."""

    target: np.ndarray
    scores: np.ndarray
    treatment: np.ndarray | None = None
    class_order: tuple[object, ...] | None = None


class Metric(Protocol):
    """Structural interface implemented by every registered AutoML metric."""

    name: str
    supported_tasks: frozenset[str]
    optimization_direction: Literal["maximize", "minimize"] | None

    def compute(self, data: MetricInput) -> float | None:
        """Calculate the metric from task-level arrays."""
