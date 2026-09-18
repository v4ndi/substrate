"""Single-estimator training and feature semantics for supervised tasks."""

from abc import abstractmethod
from collections.abc import Mapping
from time import perf_counter
from typing import Any, TypeVar

import numpy as np
import polars as pl

from fmlib.automl.backends.boosting.base import BaseBoostingBackend
from fmlib.automl.backends.boosting.hyperopt import fit_boosting_model
from fmlib.automl.data import normalize_optional_binary_treatment, prepare_data
from fmlib.automl.metrics import MetricInput, resolve_metric
from fmlib.automl.progress import log_progress
from fmlib.automl.types import PredictionResult

from .base import BaseBoostingTask, _ModelEntry

_SingleBackendT = TypeVar("_SingleBackendT", bound=BaseBoostingBackend)


class SupervisedBoostingTask(BaseBoostingTask[_SingleBackendT]):
    """Share single-model fitting between binary, regression and multiclass."""

    def _backend_options(self) -> Mapping[str, Any]:
        """Return task-specific constructor options for a boosting adapter."""
        return {}

    def _normalize_model_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Normalize shared model-role semantics before backend logic.

        Keeping role transformations in the shared data boundary prevents
        train, prediction, calibration and evaluation paths from implementing
        the same convention independently. Tasks that consume a role directly
        may retain stricter presence and value validation.
        """
        return normalize_optional_binary_treatment(
            frame,
            getattr(self._internal_config, "treatment_column", None),
            public_name=getattr(self.config, "treatment_column", None),
            inverse=bool(getattr(self.config, "inverse_treatment", False)),
        )

    def _metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        """Calculate the registered task-specific objective metric."""
        metric = resolve_metric(
            self.config.optimization_metric, self._task_name, "optimization"
        )
        value = metric.compute(
            MetricInput(
                target=target,
                scores=scores,
                class_order=getattr(self, "_class_order", None),
            )
        )
        if value is None:
            msg = f"Optimization metric {metric.name!r} returned no value"
            raise ValueError(msg)
        return float(value)

    def _optimization_metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        raise NotImplementedError

    def _fit_one(
        self,
        train_frame: pl.DataFrame,
        valid_frame: pl.DataFrame,
        *,
        layout: str,
        group_value: Any | None = None,
    ) -> _ModelEntry[_SingleBackendT]:
        """Run one local training workflow and retain the best fitted trial.

        Train and validation inputs are prepared once. Every Optuna trial uses
        validation for early stopping and is ranked by the task-specific
        validation objective. Native booster logs contain the engine-specific
        training and validation metrics; their validation split never refers to
        the path later passed to :meth:`predict` or :meth:`evaluate`.
        """
        preparation_started = perf_counter()
        model_name = self._model_name(layout, group_value)
        log_progress(
            "[model data 1/1] model=%s preparing and validating feature matrices",
            model_name,
        )
        train_frame = self._normalize_model_frame(train_frame)
        valid_frame = self._normalize_model_frame(valid_frame)
        config = self._internal_config
        categorical_roles = (
            config.group_column if layout == "global" else None,
            getattr(config, "treatment_column", None),
        )
        excluded = (
            (config.group_column,)
            if layout == "per_group" and config.group_column
            else ()
        )
        train, dimensions = prepare_data(
            train_frame,
            config,
            require_target=True,
            categorical_role_columns=categorical_roles,
            excluded_feature_columns=excluded,
            public_column_names=self._column_mapper.to_external,
        )
        valid, _ = prepare_data(
            valid_frame,
            config,
            require_target=True,
            fitted_schema=train.schema,
            hidden_dimensions=dimensions,
            categorical_role_columns=categorical_roles,
            excluded_feature_columns=excluded,
            public_column_names=self._column_mapper.to_external,
        )
        train_target = self._target(train.frame)
        valid_target = self._target(valid.frame)
        log_progress(
            "[model data 1/1] model=%s completed duration_seconds=%.3f train_rows=%d valid_rows=%d features=%d",
            model_name,
            perf_counter() - preparation_started,
            train.frame.height,
            valid.frame.height,
            len(train.schema.feature_order),
        )

        metric = resolve_metric(
            self.config.optimization_metric, self._task_name, "optimization"
        )
        fit_result = fit_boosting_model(
            backend_class=self._backend_class,
            engine=self.config.engine,
            model_params=self.config.model_params,
            search_space=self.config.search_space,
            hyperopt=self.config.hyperopt,
            n_trials=self.config.n_trials,
            random_state=self.config.random_state,
            device=config.resolved_device,
            verbose=self.config.verbose,
            train_frame=train.frame,
            train_target=train_target,
            valid_frame=valid.frame,
            valid_target=valid_target,
            schema=train.schema,
            objective_metric=self._optimization_metric,
            direction=metric.optimization_direction,
            backend_options=self._backend_options(),
        )
        schema = train.schema
        del train, valid, train_frame, valid_frame
        return _ModelEntry(
            backend=fit_result.backend,
            schema=schema,
            hidden_dimensions=dimensions,
            best_params=fit_result.best_params,
            validation_metric=fit_result.validation_metric,
            layout=layout,
            group_value=group_value,
        )

    def _predict_entry(
        self, frame: pl.DataFrame, item: _ModelEntry[_SingleBackendT]
    ) -> np.ndarray:
        """Prepare one frame partition and score it with its routed model."""
        config = self._internal_config
        categorical_roles = (
            config.group_column if item.layout == "global" else None,
            getattr(config, "treatment_column", None),
        )
        excluded = (
            (config.group_column,)
            if item.layout == "per_group" and config.group_column
            else ()
        )
        prepared, _ = prepare_data(
            frame,
            config,
            require_target=False,
            fitted_schema=item.schema,
            hidden_dimensions=item.hidden_dimensions,
            categorical_role_columns=categorical_roles,
            excluded_feature_columns=excluded,
            public_column_names=self._column_mapper.to_external,
        )
        return item.backend.predict_score(prepared.frame, item.schema)

    def _normalize_prediction_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Apply supervised feature-role semantics before prediction."""
        return self._normalize_model_frame(frame)

    def _score_prediction_branch(
        self,
        frame: pl.DataFrame,
    ) -> np.ndarray:
        """Score one raw prediction branch."""
        return self._predict_frame(frame)

    @abstractmethod
    def _prediction_result(
        self, frame: pl.DataFrame, raw: np.ndarray
    ) -> PredictionResult:
        """Construct scalar or multiclass raw-score outputs."""
