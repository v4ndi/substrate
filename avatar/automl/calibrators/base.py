"""Typed interface shared by declarative score calibrators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, Self

import numpy as np


class Calibrator(Protocol):
    """Fit, apply, and serialize one scalar calibration mapping."""

    @classmethod
    def fit(cls, scores: np.ndarray, target: np.ndarray) -> Self: ...

    def predict(self, scores: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self: ...
