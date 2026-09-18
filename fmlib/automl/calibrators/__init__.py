"""Publicly selected scalar score-calibration strategies."""

from typing import Literal

from fmlib.automl.exceptions import ConfigError

from .base import Calibrator
from .beta_calibration import BetaCalibrator
from .isotonic_regression import IsotonicCalibrator

CalibrationStrategy = Literal["beta_calibration", "isotonic_regression"]


def calibrator_class(strategy: str) -> type[Calibrator]:
    """Resolve a complete public strategy name to its implementation."""
    strategies: dict[str, type[Calibrator]] = {
        "beta_calibration": BetaCalibrator,
        "isotonic_regression": IsotonicCalibrator,
    }
    try:
        return strategies[strategy]
    except (KeyError, TypeError) as exc:
        msg = f"Unsupported calibration_strategy={strategy!r}; expected 'beta_calibration' or 'isotonic_regression'"
        raise ConfigError(msg) from exc


__all__ = [
    "BetaCalibrator",
    "CalibrationStrategy",
    "Calibrator",
    "IsotonicCalibrator",
    "calibrator_class",
]
