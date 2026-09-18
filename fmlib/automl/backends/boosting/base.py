"""Shared native-model lifecycle for boosting adapters."""

from __future__ import annotations

import json
import tempfile
from abc import abstractmethod
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl

from fmlib.automl.backends.interface import ModelBackend
from fmlib.automl.config.boosting import default_model_params
from fmlib.automl.data.schema import FeatureSchema
from fmlib.automl.exceptions import (
    ArtifactIntegrityError,
    MissingDependencyError,
    UnsupportedBackendError,
)

_CATBOOST_QUANTIZATION_PARAMETERS = {
    "border_count",
    "max_bin",
    "feature_border_type",
    "per_float_feature_quantization",
    "nan_mode",
}
_XGBOOST_NULL_CATEGORY = "__FMLIB_NULL__"
_XGBOOST_UNKNOWN_CATEGORY = "__FMLIB_UNKNOWN__"


@dataclass
class _PreparedFitData:
    """Engine-ready train and validation data shared across trials."""

    train_features: Any
    valid_features: Any
    valid_prediction_features: Any
    train_target: np.ndarray
    valid_target: np.ndarray
    category_values: dict[str, tuple[str, ...]]
    train_rows: int


@dataclass
class BaseBoostingBackend(ModelBackend):
    """Common preparation and persistence for independent boosting adapters.

    The adapter keeps feature ordering and categorical encoding stable across
    training, inference and artifact loading. CatBoost receives Polars data via
    ``Pool``; XGBoost receives Polars frames with native Enum columns and a
    fitted category vocabulary shared by training and inference.

    Args:
        engine: Native engine name, ``catboost`` or ``xgboost``.
        params: Parameters passed to the native estimator.
        random_state: Reproducibility seed.
        device: Runtime device, ``cpu`` or ``gpu``.
        verbose: Native estimator logging switch or period.
        model: Optional already-created native estimator.
        category_values: Stored categorical encoding for XGBoost inference.
        train_rows: Number of rows used to derive dataset-dependent defaults.
    """

    model: Any = None
    category_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    train_rows: int | None = None
    task_name: ClassVar[str | None] = None

    def can_prequantize(self, search_space: Mapping[str, Any] | None) -> bool:
        """Return whether one quantized CatBoost Pool is valid for every trial.

        Shared prequantization is disabled when the search space changes a
        quantization parameter because each trial then requires different bins.

        Args:
            search_space: Effective hyperparameter search space.

        Returns:
            ``True`` when reusable CatBoost pools can be quantized before trials.
        """
        return self.engine == "catboost" and not (
            search_space
            and _CATBOOST_QUANTIZATION_PARAMETERS.intersection(search_space)
        )

    def _model_params_with_defaults(
        self, *, excluded: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        """Merge packaged engine defaults with explicit native parameters."""
        defaults = default_model_params(
            self.engine, task=self.task_name, train_rows=self.train_rows
        )
        for name in excluded:
            defaults.pop(name, None)
        if self.engine == "catboost":
            if {
                "iterations",
                "num_trees",
                "num_boost_round",
                "n_estimators",
            }.intersection(self.params):
                defaults.pop("num_trees", None)
            if {"depth", "max_depth"}.intersection(self.params):
                defaults.pop("max_depth", None)
        return defaults | dict(self.params)

    @abstractmethod
    def _make_model(self):
        """Construct a task-specific native estimator."""

    def _artifact_state(self) -> Mapping[str, Any]:
        """Return optional JSON-safe estimator-output metadata."""
        return {}

    def _restore_artifact_state(self, state: Mapping[str, Any]) -> None:
        """Restore optional estimator-output metadata before native loading."""
        return None

    def _after_fit(self) -> None:
        """Capture optional estimator state after fitting."""
        return None

    def _prepare_features(
        self, frame: pl.DataFrame, schema: FeatureSchema, *, fit: bool
    ) -> Any:
        """Select ordered features and build the engine-native tabular input."""
        result = frame.select(schema.feature_order)
        for column in schema.categorical:
            if self.engine == "catboost":
                values = result[column].cast(pl.String).fill_null("__NULL__")
                result = result.with_columns(values.alias(column))
                continue
            values = result[column].cast(pl.String).fill_null(_XGBOOST_NULL_CATEGORY)
            if fit:
                categories = tuple(sorted(values.unique().to_list()))
                self.category_values[column] = categories
            categories = self.category_values[column]
            normalized = (
                pl.when(values.is_in(categories))
                .then(values)
                .otherwise(pl.lit(_XGBOOST_UNKNOWN_CATEGORY))
            )
            vocabulary = tuple(dict.fromkeys((*categories, _XGBOOST_UNKNOWN_CATEGORY)))
            result = result.with_columns(
                normalized.cast(pl.Enum(vocabulary)).alias(column)
            )
        return result

    def prepare_fit_data(
        self,
        frame: pl.DataFrame,
        target: np.ndarray,
        schema: FeatureSchema,
        *,
        valid_frame: pl.DataFrame,
        valid_target: np.ndarray,
        prequantize: bool = True,
    ) -> _PreparedFitData:
        """Prepare reusable train and validation inputs for estimator fitting.

        CatBoost pools are optionally quantized once and reused by all Optuna
        trials. Validation keeps a Polars view for score calculation because a
        mixed categorical quantized Pool is not a reliable prediction input.

        Args:
            frame: Normalized training frame.
            target: Training target array.
            schema: Ordered feature schema.
            valid_frame: Normalized validation frame.
            valid_target: Validation target array.
            prequantize: Quantize reusable CatBoost pools before fitting.

        Returns:
            Engine-ready reusable training and validation data.
        """
        self.train_rows = frame.height
        train_features = self._prepare_features(frame, schema, fit=True)
        if self.engine == "catboost":
            try:
                from catboost import Pool
            except ImportError as exc:
                msg = "CatBoost engine requires the 'catboost' package"
                raise MissingDependencyError(msg) from exc
            train_features = Pool(
                train_features, target, cat_features=list(schema.categorical)
            )
            if prequantize:
                quantization_params = {
                    name: value
                    for name, value in self._model_params_with_defaults().items()
                    if name in _CATBOOST_QUANTIZATION_PARAMETERS
                }
                task_type = "GPU" if self.device == "gpu" else "CPU"
                train_features.quantize(task_type=task_type, **quantization_params)
            valid_prediction_features = self._prepare_features(
                valid_frame, schema, fit=False
            )
            valid_features = Pool(
                valid_prediction_features,
                valid_target,
                cat_features=list(schema.categorical),
            )
            if prequantize:
                with tempfile.TemporaryDirectory(
                    prefix="fmlib-catboost-borders-"
                ) as directory:
                    borders_path = Path(directory) / "borders.tsv"
                    train_features.save_quantization_borders(str(borders_path))
                    valid_features.quantize(
                        input_borders=str(borders_path), task_type=task_type
                    )
        else:
            valid_features = self._prepare_features(valid_frame, schema, fit=False)
            valid_prediction_features = valid_features
        return _PreparedFitData(
            train_features=train_features,
            valid_features=valid_features,
            valid_prediction_features=valid_prediction_features,
            train_target=target,
            valid_target=valid_target,
            category_values=dict(self.category_values),
            train_rows=frame.height,
        )

    def fit_prepared(self, prepared: _PreparedFitData) -> None:
        """Fit one estimator using prepared training and validation data.

        Args:
            prepared: Engine-ready data returned by :meth:`prepare_fit_data`.

        Returns:
            ``None``. The fitted native estimator is stored in :attr:`model`.

        Raises:
            UnsupportedBackendError: If ``engine`` is unsupported.
        """
        self.category_values = dict(prepared.category_values)
        self.train_rows = prepared.train_rows
        self.model = self._make_model()
        if self.engine == "catboost":
            self.model.fit(
                prepared.train_features,
                eval_set=prepared.valid_features,
                verbose=self.verbose,
            )
        elif self.engine == "xgboost":
            self.model.fit(
                prepared.train_features,
                prepared.train_target,
                eval_set=[(prepared.valid_features, prepared.valid_target)],
                verbose=self.verbose,
            )
        else:
            msg = f"Unsupported boosting engine: {self.engine!r}"
            raise UnsupportedBackendError(msg)
        self._after_fit()

    def fit(
        self,
        frame: pl.DataFrame,
        target: np.ndarray,
        schema: FeatureSchema,
        *,
        valid_frame: pl.DataFrame,
        valid_target: np.ndarray,
    ) -> None:
        """Prepare data and fit with validation-based early stopping.

        Args:
            frame: Normalized training frame.
            target: Training target array.
            schema: Ordered feature schema.
            valid_frame: Normalized validation frame.
            valid_target: Validation target array.

        Returns:
            ``None``. The fitted native estimator is stored in :attr:`model`.
        """
        prepared = self.prepare_fit_data(
            frame,
            target,
            schema,
            valid_frame=valid_frame,
            valid_target=valid_target,
        )
        self.fit_prepared(prepared)

    def predict_score(self, frame: pl.DataFrame, schema: FeatureSchema) -> np.ndarray:
        """Calculate task-specific continuous scores.

        Args:
            frame: Normalized inference frame.
            schema: Fitted feature schema.

        Returns:
            NumPy array of task-specific scores in input row order.
        """
        features = self._prepare_features(frame, schema, fit=False)
        return self.predict_prepared_score(features)

    @abstractmethod
    def predict_prepared_score(self, features: Any) -> np.ndarray:
        """Return task-specific continuous scores for engine-ready features."""

    def feature_importance(self, schema: FeatureSchema) -> pl.DataFrame | None:
        """Build feature importances sorted from highest to lowest.

        Args:
            schema: Fitted feature schema providing feature names.

        Returns:
            A sorted Polars DataFrame, or ``None`` when unavailable.
        """
        values = self._native_feature_importances(len(schema.feature_order))
        if values is None:
            return None
        return pl.DataFrame({
            "feature": schema.feature_order,
            "importance": values,
        }).sort("importance", descending=True)

    def _native_feature_importances(self, expected_count: int) -> np.ndarray | None:
        """Return a validated one-dimensional native importance vector."""
        if self.model is None:
            return None
        if self.engine == "catboost":
            values = self.model.get_feature_importance(type="FeatureImportance")
        else:
            values = getattr(self.model, "feature_importances_", None)
        if values is None:
            return None
        normalized = np.asarray(values, dtype=float).reshape(-1)
        if normalized.size != expected_count:
            msg = (
                f"Native {self.engine} model returned {normalized.size} feature importances; "
                f"the fitted schema contains {expected_count} features"
            )
            raise ArtifactIntegrityError(msg)
        return normalized

    def save(self, path: Path) -> None:
        """Save the native model and backend metadata.

        Args:
            path: Destination backend directory.

        Returns:
            ``None``.

        Raises:
            ArtifactIntegrityError: If the backend is not fitted.
        """
        path.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            msg = "Cannot save an unfitted boosting backend"
            raise ArtifactIntegrityError(msg)
        suffix = "cbm" if self.engine == "catboost" else "json"
        model_path = path / f"model.{suffix}"
        if self.engine == "xgboost":
            # Persist the native Booster instead of the sklearn wrapper.  The
            # latter calls its private ``_get_type`` compatibility hook, which
            # is absent for some supported XGBoost/scikit-learn combinations.
            self.model.get_booster().save_model(str(model_path))
        else:
            self.model.save_model(str(model_path))
        metadata = {
            "engine": self.engine,
            "params": dict(self.params),
            "random_state": self.random_state,
            "verbose": self.verbose,
            "category_values": {
                name: list(values) for name, values in self.category_values.items()
            },
            "task_state": dict(self._artifact_state()),
        }
        (path / "backend.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> BaseBoostingBackend:
        """Restore a fitted backend using the caller's runtime device.

        Args:
            path: Backend directory written by :meth:`save`.
            device: Runtime ``cpu`` or ``gpu`` device.

        Returns:
            A fitted instance of the concrete backend class.

        Raises:
            ArtifactIntegrityError: If metadata or the native model is missing.
        """
        metadata_path = path / "backend.json"
        if not metadata_path.exists():
            msg = f"Missing backend metadata: {metadata_path}"
            raise ArtifactIntegrityError(msg)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        backend = cls(
            engine=metadata["engine"],
            params=metadata["params"],
            random_state=metadata["random_state"],
            device=device,
            verbose=metadata.get("verbose", False),
            category_values={
                name: tuple(values)
                for name, values in metadata["category_values"].items()
            },
        )
        backend._restore_artifact_state(metadata.get("task_state", {}))
        suffix = "cbm" if backend.engine == "catboost" else "json"
        model_path = path / f"model.{suffix}"
        if not model_path.exists():
            msg = f"Missing native model file: {model_path}"
            raise ArtifactIntegrityError(msg)
        backend.model = backend._make_model()
        backend.model.load_model(str(model_path))
        return backend

    def set_runtime_device(self, device: str) -> None:
        """Select the device used by subsequent prediction calls."""
        super().set_runtime_device(device)
        if self.model is not None and self.engine == "xgboost":
            self.model.set_params(device="cuda" if device == "gpu" else "cpu")

    def for_execution(self, device: str) -> BaseBoostingBackend:
        """Isolate native device changes only when the requested device differs."""
        if device == self.device:
            return self
        backend = copy(self)
        if self.engine == "xgboost" and self.model is not None:
            backend.model = deepcopy(self.model)
        backend.set_runtime_device(device)
        return backend
