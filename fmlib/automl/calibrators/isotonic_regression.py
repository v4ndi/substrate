"""Declarative isotonic-regression calibration mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fmlib.automl.exceptions import MissingDependencyError, SchemaError


@dataclass(frozen=True)
class IsotonicCalibrator:
    """JSON-safe clipped piecewise-linear isotonic mapping."""

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    @classmethod
    def fit(cls, scores: np.ndarray, target: np.ndarray) -> IsotonicCalibrator:
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            msg = "Isotonic calibration requires the 'scikit-learn' package"
            raise MissingDependencyError(msg) from exc
        values = np.asarray(scores, dtype=float)
        targets = np.asarray(target)
        if values.ndim != 1 or values.size != targets.size or values.size < 2:
            msg = "Calibration scores must be a one-dimensional array matching at least two target rows"
            raise SchemaError(msg)
        if (
            not np.isfinite(values).all()
            or not np.isfinite(targets.astype(float)).all()
        ):
            msg = "Calibration scores and targets must be finite"
            raise SchemaError(msg)
        if np.unique(values).size < 2:
            msg = "Isotonic calibration requires at least two distinct scores"
            raise SchemaError(msg)
        if set(np.unique(targets).tolist()) != {0, 1}:
            msg = "Calibration data must contain both target classes 0 and 1"
            raise SchemaError(msg)
        model = IsotonicRegression(out_of_bounds="clip").fit(
            values, targets.astype(float)
        )
        return cls(
            x_thresholds=tuple(float(value) for value in model.X_thresholds_),
            y_thresholds=tuple(float(value) for value in model.y_thresholds_),
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all():
            msg = "Scores to calibrate must be finite"
            raise SchemaError(msg)
        return np.interp(
            values,
            np.asarray(self.x_thresholds),
            np.asarray(self.y_thresholds),
            left=self.y_thresholds[0],
            right=self.y_thresholds[-1],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_thresholds": list(self.x_thresholds),
            "y_thresholds": list(self.y_thresholds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IsotonicCalibrator:
        return cls(
            x_thresholds=tuple(float(value) for value in payload["x_thresholds"]),
            y_thresholds=tuple(float(value) for value in payload["y_thresholds"]),
        )
