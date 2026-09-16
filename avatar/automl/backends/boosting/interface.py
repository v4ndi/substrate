"""Operations shared by fitted native models and uplift composites."""

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Self

import numpy as np
import polars as pl

from avatar.automl.data.schema import FeatureSchema
from avatar.automl.exceptions import ConfigError


@dataclass
class BoostingBackend(ABC):
    """Common runtime and persistence contract, independent of training strategy."""

    engine: str
    params: Mapping[str, Any]
    random_state: int
    device: str = "gpu"
    verbose: bool | int = False

    @abstractmethod
    def predict_score(self, frame: pl.DataFrame, schema: FeatureSchema) -> np.ndarray:
        """Return task-specific scores with the input row order preserved."""

    @abstractmethod
    def feature_importance(self, schema: FeatureSchema) -> pl.DataFrame | None:
        """Return feature importances, or None when unavailable."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted backend and all state required for inference."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path, *, device: str = "cpu") -> Self:
        """Restore the concrete backend with the requested runtime device."""

    def set_runtime_device(self, device: str) -> None:
        """Validate and select the runtime device."""
        if device not in {"cpu", "gpu"}:
            msg = f"Unsupported boosting runtime device: {device!r}"
            raise ConfigError(msg)
        self.device = device

    def for_execution(self, device: str) -> Self:
        """Reuse this backend unless an operation requests a different device."""
        if device == self.device:
            return self
        backend = copy(self)
        backend.set_runtime_device(device)
        return backend
