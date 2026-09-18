"""Contracts a model backend implements, independent of the model family.

Two halves, because two different callers need two different things:

* :class:`ModelBackend` -- what a *fitted* model must do: score a frame,
  report importances, persist and restore. The task layer, the artifact
  repository and the prediction path speak only this.
* :class:`TrainableBackend` -- what :func:`fmlib.automl.backends.search.fit_model`
  calls while searching. A ``Protocol`` rather than a base class: it records
  the surface the search machinery depends on without forcing a second
  inheritance edge on adapters that already have one.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import polars as pl

from fmlib.automl.data.schema import FeatureSchema
from fmlib.automl.exceptions import ConfigError


@dataclass
class ModelBackend(ABC):
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
            msg = f"Unsupported runtime device: {device!r}"
            raise ConfigError(msg)
        self.device = device

    def for_execution(self, device: str) -> Self:
        """Reuse this backend unless an operation requests a different device."""
        if device == self.device:
            return self
        backend = copy(self)
        backend.set_runtime_device(device)
        return backend


@runtime_checkable
class PreparedFitData(Protocol):
    """Engine-ready train and validation data shared across trials of one search."""

    valid_prediction_features: Any


@runtime_checkable
class TrainableBackend(Protocol):
    """The surface :func:`fmlib.automl.backends.search.fit_model` trains through.

    Preparation is separate from fitting because every trial of a search reuses
    the same prepared data; scoring takes prepared features rather than a frame
    for the same reason.
    """

    def can_prequantize(self, search_space: Mapping[str, Any] | None) -> bool:
        """Return whether one preparation can serve every trial of the search."""
        ...

    def prepare_fit_data(
        self,
        train_frame: pl.DataFrame,
        train_target: np.ndarray,
        schema: FeatureSchema,
        *,
        valid_frame: pl.DataFrame,
        valid_target: np.ndarray,
        prequantize: bool,
    ) -> PreparedFitData:
        """Convert frames into the representation the trials will be fitted on."""
        ...

    def fit_prepared(self, prepared: PreparedFitData) -> None:
        """Fit this backend on already-prepared data."""
        ...

    def predict_prepared_score(self, features: Any) -> np.ndarray:
        """Score already-prepared features, preserving row order."""
        ...
