"""Declarative beta-calibration mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from avatar.automl.exceptions import MissingDependencyError, SchemaError


@dataclass(frozen=True)
class BetaCalibrator:
    """JSON-safe numerical state sufficient for beta-calibration inference."""

    map_values: tuple[float, float, float]
    coefficients: tuple[float, ...]
    intercept: float

    @classmethod
    def fit(cls, scores: np.ndarray, target: np.ndarray) -> BetaCalibrator:
        try:
            from betacal import BetaCalibration
        except ImportError as exc:
            msg = "Beta calibration requires the 'betacal' package"
            raise MissingDependencyError(msg) from exc
        values = np.asarray(scores, dtype=float)
        targets = np.asarray(target)
        if (
            values.ndim != 1
            or values.size != targets.size
            or values.size < 2
            or not np.isfinite(values).all()
        ):
            msg = "Calibration scores must be a finite one-dimensional array matching target rows"
            raise SchemaError(msg)
        if np.any(values < 0) or np.any(values > 1):
            msg = "Calibration probabilities must lie in [0, 1]"
            raise SchemaError(msg)
        if set(np.unique(targets).tolist()) != {0, 1}:
            msg = "Calibration data must contain both target classes 0 and 1"
            raise SchemaError(msg)
        model = (
            BetaCalibration()
            .fit(values.reshape(-1, 1), targets.astype(int))
            .calibrator_
        )
        return cls(
            map_values=tuple(float(value) for value in model.map_),
            coefficients=tuple(float(value) for value in model.lr_.coef_[0]),
            intercept=float(model.lr_.intercept_[0]),
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            msg = "Scores to calibrate must be finite probabilities in [0, 1]"
            raise SchemaError(msg)
        epsilon = np.finfo(values.dtype).eps
        values = np.clip(values, epsilon, 1 - epsilon)
        transformed = np.hstack((np.log(values), -np.log(1 - values)))
        if self.map_values[0] == 0:
            transformed = transformed[:, 1:2]
        elif self.map_values[1] == 0:
            transformed = transformed[:, 0:1]
        logits = transformed @ np.asarray(self.coefficients) + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_values": list(self.map_values),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BetaCalibrator:
        return cls(
            map_values=tuple(float(value) for value in payload["map_values"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
        )
