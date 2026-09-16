"""Path-based boosting uplift task with S-, T- and X-learners."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl

from avatar.automl.backends.boosting.uplift import (
    UPLIFT_SCORE_COLUMNS,
    UpliftBoostingBackend,
)
from avatar.automl.config import UpliftTaskConfig
from avatar.automl.data import ParquetSource, normalize_date, prepare_data
from avatar.automl.exceptions import SchemaError
from avatar.automl.metrics import (
    MetricInput,
    qini_curve,
    resolve_evaluation_metrics,
    resolve_metric,
    uplift_curve,
)
from avatar.automl.progress import log_progress
from avatar.automl.reporting import CurveData, EvaluationData
from avatar.automl.types import (
    CalibrationResult,
    EvaluationKind,
    EvaluationResult,
    ParquetPath,
    PredictionResult,
    TrainingResult,
)

from .base import BaseBoostingTask, _ModelEntry
from .calibration import CalibratableTask
from .evaluation import align_prediction_scores, combine_metric_slices, metric_slices

logger = logging.getLogger(__name__)

_LEARNER_COLUMNS = {
    "s": ("score_s", "score_s_control", "score_s_treatment"),
    "t": ("score_t", "score_t_control", "score_t_treatment"),
    "x": ("score_x", "score_x_control", "score_x_treatment"),
}


class UpliftTask(CalibratableTask, BaseBoostingTask[UpliftBoostingBackend]):
    """Estimate heterogeneous treatment effects with S-, T-, and X-learners.

    All three learners are fitted for every selected model scope. Channel and
    Combined layouts require a configured physical group column; the treatment
    column remains mandatory.

    Args:
        config: Validated uplift-task configuration.
    """

    _backend_class = UpliftBoostingBackend
    _config_class = UpliftTaskConfig
    _task_name = "uplift"
    _artifact_directory = "uplift_model"

    def __init__(self, config: UpliftTaskConfig, *, _entity_path: Path | None = None):
        """Initialize uplift-task orchestration.

        Args:
            config: Validated uplift-task configuration.
        """
        super().__init__(config, _entity_path=_entity_path)

    @staticmethod
    def _binary_values(series: pl.Series, public_name: str) -> np.ndarray:
        if series.null_count():
            msg = f"Column {public_name!r} contains null values"
            raise SchemaError(msg)
        try:
            values = series.cast(pl.Float64, strict=True).to_numpy()
        except Exception as exc:
            msg = f"Column {public_name!r} must contain scalar binary values 0 and 1"
            raise SchemaError(msg) from exc
        if not np.isfinite(values).all():
            msg = f"Column {public_name!r} contains NaN or infinite values"
            raise SchemaError(msg)
        unique = set(np.unique(values).tolist())
        if not unique <= {0.0, 1.0}:
            msg = f"Column {public_name!r} must contain only 0 and 1; got {sorted(unique)}"
            raise SchemaError(msg)
        return values.astype(np.int8)

    def _target(self, frame: pl.DataFrame) -> np.ndarray:
        column = self._internal_config.target_column
        if column not in frame.columns:
            msg = f"Missing target column: {self.config.target_column!r}"
            raise SchemaError(msg)
        return self._binary_values(frame[column], self.config.target_column)

    def _treatment(
        self, frame: pl.DataFrame, *, require_both: bool = False
    ) -> np.ndarray:
        column = self._internal_config.treatment_column
        if column is None or column not in frame.columns:
            msg = f"Missing treatment column: {self.config.treatment_column!r}"
            raise SchemaError(msg)
        values = self._binary_values(
            frame[column], self.config.treatment_column or "treatment"
        )
        if require_both and set(np.unique(values).tolist()) != {0, 1}:
            msg = "Uplift training/evaluation requires both treatment arms 0 and 1"
            raise SchemaError(msg)
        return values

    def _normalize_treatment(
        self, frame: pl.DataFrame, *, required: bool
    ) -> pl.DataFrame:
        column = self._internal_config.treatment_column
        if column is None or column not in frame.columns:
            if required:
                msg = f"Missing treatment column: {self.config.treatment_column!r}"
                raise SchemaError(msg)
            return frame
        values = self._binary_values(
            frame[column], self.config.treatment_column or "treatment"
        )
        if self.config.inverse_treatment:
            values = 1 - values
        return frame.with_columns(pl.Series(column, values, dtype=pl.Int8))

    def _fit_one(
        self,
        train_frame: pl.DataFrame,
        valid_frame: pl.DataFrame,
        *,
        layout: str,
        group_value: Any | None = None,
    ) -> _ModelEntry[UpliftBoostingBackend]:
        preparation_started = perf_counter()
        part_name = self._model_name(layout, group_value)
        log_progress(
            "[model data 1/1] model=%s preparing and validating uplift matrices",
            part_name,
        )
        train_frame = self._normalize_treatment(train_frame, required=True)
        valid_frame = self._normalize_treatment(valid_frame, required=True)
        config = self._internal_config
        excluded = [config.treatment_column]
        if layout == "per_group" and config.group_column:
            excluded.append(config.group_column)
        categorical_roles = (config.group_column if layout == "global" else None,)
        train, dimensions = prepare_data(
            train_frame,
            config,
            require_target=True,
            categorical_role_columns=categorical_roles,
            public_column_names=self._column_mapper.to_external,
            excluded_feature_columns=tuple(value for value in excluded if value),
        )
        valid, _ = prepare_data(
            valid_frame,
            config,
            require_target=True,
            fitted_schema=train.schema,
            hidden_dimensions=dimensions,
            categorical_role_columns=categorical_roles,
            public_column_names=self._column_mapper.to_external,
            excluded_feature_columns=tuple(value for value in excluded if value),
        )
        y_train, y_valid = self._target(train.frame), self._target(valid.frame)
        t_train, t_valid = (
            self._treatment(train.frame, require_both=True),
            self._treatment(valid.frame, require_both=True),
        )
        log_progress(
            "[model data 1/1] model=%s completed duration_seconds=%.3f train_rows=%d valid_rows=%d features=%d",
            part_name,
            perf_counter() - preparation_started,
            train.frame.height,
            valid.frame.height,
            len(train.schema.feature_order),
        )
        backend = UpliftBoostingBackend(
            engine=self.config.engine,
            params=self.config.model_params,
            random_state=self.config.random_state,
            device=config.resolved_device,
            verbose=self.config.verbose,
            estimate_propensity=self.config.estimate_propensity,
            treatment_column=config.treatment_column or "treatment",
        )
        backend.fit_composite(
            train.frame,
            y_train,
            t_train,
            valid.frame,
            y_valid,
            t_valid,
            train.schema,
            hyperopt=self.config.hyperopt,
            n_trials=self.config.n_trials,
            model_params=self.config.model_params,
            search_space=self.config.search_space,
            metric=resolve_metric(
                self.config.optimization_metric, self._task_name, "optimization"
            ),
            part_name=part_name,
        )
        return _ModelEntry(
            backend=backend,
            schema=train.schema,
            hidden_dimensions=dimensions,
            best_params={
                name: dict(value) for name, value in backend.learner_params.items()
            },
            validation_metric=float(np.mean(list(backend.learner_metrics.values()))),
            layout=layout,
            group_value=group_value,
        )

    def _training_summary(self, result: TrainingResult) -> TrainingResult:
        best_params: dict[str, Any] = {}
        validation: dict[str, float] = {}
        for item in self._models:
            part = self._model_name(item.layout, item.group_value)
            for learner, params in item.backend.learner_params.items():
                best_params[f"{part}:{learner}"] = params
                validation[f"{part}:{learner}"] = item.backend.learner_metrics[learner]
        return TrainingResult(
            best_params=best_params,
            validation_metrics=validation,
            feature_names=result.feature_names,
            backend_name=result.backend_name,
            engine_name=result.engine_name,
            task_name=result.task_name,
        )

    def _execute_train(
        self,
        train_path: ParquetPath,
        valid_path: ParquetPath,
        *,
        remote_layout: str | None = None,
        remote_group_value: Any | None = None,
    ) -> TrainingResult:
        result = super()._execute_train(
            train_path,
            valid_path,
            remote_layout=remote_layout,
            remote_group_value=remote_group_value,
        )
        return self._training_summary(result)

    def _remote_training_parts(
        self, train_path: ParquetPath
    ) -> list[tuple[str, Any | None]]:
        return super()._remote_training_parts(train_path)

    def _predict_entry(
        self, frame: pl.DataFrame, item: _ModelEntry[UpliftBoostingBackend]
    ) -> np.ndarray:
        config = self._internal_config
        excluded = [config.treatment_column]
        if item.layout == "per_group" and config.group_column:
            excluded.append(config.group_column)
        prepared, _ = prepare_data(
            frame,
            config,
            require_target=False,
            fitted_schema=item.schema,
            hidden_dimensions=item.hidden_dimensions,
            categorical_role_columns=(
                config.group_column if item.layout == "global" else None,
            ),
            public_column_names=self._column_mapper.to_external,
            excluded_feature_columns=tuple(value for value in excluded if value),
        )
        return item.backend.predict_score(prepared.frame, item.schema)

    def _score_base(self, frame: pl.DataFrame) -> pl.DataFrame:
        config = self._internal_config
        columns = [config.client_id_column]
        if config.date_column is not None:
            columns.append(config.date_column)
        for column in (config.group_column, config.treatment_column):
            if column and column in frame.columns and column not in columns:
                columns.append(column)
        return frame.select(columns)

    @staticmethod
    def _validate_score_matrix(
        values: np.ndarray, *, require_x_effect: bool = False
    ) -> None:
        if values.ndim != 2 or values.shape[1] != len(UPLIFT_SCORE_COLUMNS):
            msg = f"Uplift score matrix must have shape (N, {len(UPLIFT_SCORE_COLUMNS)}); got {values.shape}"
            raise SchemaError(msg)
        if not np.isfinite(values).all():
            msg = "Uplift scores contain NaN or infinite values"
            raise SchemaError(msg)
        probability_indices = (1, 2, 4, 5, 7, 8, 9)
        if np.any(values[:, probability_indices] < 0) or np.any(
            values[:, probability_indices] > 1
        ):
            msg = "Uplift outcome/propensity probabilities must lie in [0, 1]"
            raise SchemaError(msg)
        consistent_learners = [(0, 1, 2), (3, 4, 5)]
        if require_x_effect:
            consistent_learners.append((6, 7, 8))
        for effect, control, treated in consistent_learners:
            if not np.allclose(
                values[:, effect], values[:, treated] - values[:, control], atol=1e-10
            ):
                msg = "Uplift effects are inconsistent with treatment-control score differences"
                raise SchemaError(msg)

    def _normalize_prediction_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Normalize optional factual treatment for production prediction."""
        return self._normalize_treatment(frame, required=False)

    def _score_prediction_branch(
        self,
        frame: pl.DataFrame,
    ) -> np.ndarray:
        """Validate raw S/T/X scores."""
        raw = self._predict_frame(frame)
        self._validate_score_matrix(raw)
        return raw

    def _prediction_result(
        self, frame: pl.DataFrame, raw: np.ndarray
    ) -> PredictionResult:
        """Assemble the raw composite result shape."""
        public_frame = frame.drop("__row_id") if "__row_id" in frame.columns else frame
        base = self._score_base(public_frame)
        scores = base.with_columns([
            pl.Series(name, raw[:, i]) for i, name in enumerate(UPLIFT_SCORE_COLUMNS)
        ])
        treatment = self._internal_config.treatment_column
        if treatment and treatment in scores.columns and self.config.inverse_treatment:
            scores = scores.with_columns((1 - pl.col(treatment)).alias(treatment))
        return PredictionResult(scores=self._column_mapper.restore_frame(scores))

    @staticmethod
    def _safe_metric(function, *args) -> float:
        try:
            return float(function(*args))
        except (ValueError, ZeroDivisionError):
            return float("nan")

    def _learner_metrics(
        self,
        target: np.ndarray,
        treatment: np.ndarray,
        effect: np.ndarray,
        control: np.ndarray,
        treated: np.ndarray,
        *,
        strict: bool,
        metric_names: tuple[str, ...],
    ) -> dict[str, float | int]:
        if strict and set(np.unique(treatment).tolist()) != {0, 1}:
            msg = "Overall uplift evaluation requires both treatment arms"
            raise SchemaError(msg)
        result: dict[str, float | int] = {
            "n_samples": len(target),
            "n_treatment": int(treatment.sum()),
            "n_control": int((1 - treatment).sum()),
            "n_positive_treatment": int(target[treatment == 1].sum()),
            "n_positive_control": int(target[treatment == 0].sum()),
        }
        for metric in resolve_evaluation_metrics(metric_names, self._task_name):
            if metric.name in {"treatment_roc_auc", "control_roc_auc"}:
                arm = 1 if metric.name == "treatment_roc_auc" else 0
                probabilities = treated if arm == 1 else control
                mask = treatment == arm
                value = (
                    self._safe_metric(
                        metric.compute, MetricInput(target[mask], probabilities[mask])
                    )
                    if mask.any() and np.unique(target[mask]).size == 2
                    else float("nan")
                )
            else:
                value = self._safe_metric(
                    metric.compute, MetricInput(target, effect, treatment=treatment)
                )
            if strict and not np.isfinite(value):
                msg = f"Overall uplift metric {metric.name!r} is undefined for the evaluation split"
                raise SchemaError(msg)
            result[metric.name] = value
        return result

    def _evaluate_table(
        self,
        truth: pl.DataFrame,
        score_frame: pl.DataFrame,
        metric_names: tuple[str, ...],
    ) -> tuple[dict[str, float | int], pl.DataFrame | None]:
        if "model_layout" in score_frame.columns:
            overall: dict[str, float | int] = {}
            grouped: list[pl.DataFrame] = []
            group_column = self._internal_config.group_column
            for key, branch in score_frame.partition_by(
                "model_layout", as_dict=True, maintain_order=True
            ).items():
                scope = key[0] if isinstance(key, tuple) else key
                branch_truth = truth
                if (
                    scope == "per_group"
                    and group_column
                    and group_column in branch.columns
                ):
                    branch_truth = truth.filter(
                        pl.col(group_column).is_in(
                            branch[group_column].unique().to_list()
                        )
                    )
                branch_metrics, branch_grouped = self._evaluate_table(
                    branch_truth,
                    branch.drop("model_layout"),
                    metric_names,
                )
                overall.update({
                    f"{scope}_{name}": value for name, value in branch_metrics.items()
                })
                if branch_grouped is not None:
                    grouped.append(
                        branch_grouped.with_columns(pl.lit(scope).alias("model_layout"))
                    )
            return overall, combine_metric_slices(grouped, self._internal_config)
        target, treatment = (
            self._target(truth),
            self._treatment(truth, require_both=True),
        )
        overall: dict[str, float | int] = {}
        for learner, (effect, control, treated) in _LEARNER_COLUMNS.items():
            values = self._learner_metrics(
                target,
                treatment,
                score_frame[effect].to_numpy(),
                score_frame[control].to_numpy(),
                score_frame[treated].to_numpy(),
                strict=True,
                metric_names=metric_names,
            )
            overall.update({
                f"{learner}_{name}": value for name, value in values.items()
            })
        combined = truth.with_columns([
            score_frame[name] for name in UPLIFT_SCORE_COLUMNS
        ])
        config = self._internal_config
        tables: list[pl.DataFrame] = []
        for slice_name, group_columns in metric_slices(config):
            rows: list[dict[str, Any]] = []
            for keys, part in combined.group_by(group_columns, maintain_order=True):
                key_values = keys if isinstance(keys, tuple) else (keys,)
                part_target, part_treatment = self._target(part), self._treatment(part)
                for learner, (effect, control, treated) in _LEARNER_COLUMNS.items():
                    metrics = self._learner_metrics(
                        part_target,
                        part_treatment,
                        part[effect].to_numpy(),
                        part[control].to_numpy(),
                        part[treated].to_numpy(),
                        strict=False,
                        metric_names=metric_names,
                    )
                    if any(
                        isinstance(value, float) and np.isnan(value)
                        for value in metrics.values()
                    ):
                        logger.warning(
                            "Some uplift metrics are undefined for slice %s, learner=%s; null is returned",
                            dict(zip(group_columns, key_values, strict=True)),
                            learner,
                        )
                    rows.append(
                        {
                            "scope": slice_name,
                            **dict(zip(group_columns, key_values, strict=True)),
                        }
                        | {"learner": learner}
                        | metrics
                    )
            table = pl.DataFrame(rows)
            floating_columns = [
                name
                for name, dtype in table.schema.items()
                if dtype in {pl.Float32, pl.Float64}
            ]
            expressions = [pl.col(floating_columns).fill_nan(None)]
            if config.date_column in group_columns:
                expressions.append(pl.col(config.date_column).dt.strftime("%Y-%m-%d"))
            tables.append(
                table.with_columns(*expressions).sort([*group_columns, "learner"])
            )
        return overall, combine_metric_slices(tables, config)

    def _score_input(
        self, scores: PredictionResult | CalibrationResult | ParquetPath
    ) -> pl.DataFrame:
        selected = (
            scores.scores
            if isinstance(scores, PredictionResult | CalibrationResult)
            else ParquetSource.resolve(scores).read()
        )
        normalized = self._column_mapper.normalize_frame(selected)
        date_column = self._internal_config.date_column
        return (
            normalized
            if date_column is None
            else normalize_date(normalized, date_column)
        )

    def _curve_data(
        self,
        truth: pl.DataFrame,
        raw_frame: pl.DataFrame,
    ) -> dict[str, CurveData]:
        """Compute learner curve coordinates before rendering."""
        curves = {}
        layout_tables = (
            list(
                raw_frame.partition_by(
                    "model_layout", as_dict=True, maintain_order=True
                ).items()
            )
            if "model_layout" in raw_frame.columns
            else [(None, raw_frame)]
        )
        for layout_key, layout_table in layout_tables:
            layout = layout_key[0] if isinstance(layout_key, tuple) else layout_key
            layout_truth = truth
            group = self._internal_config.group_column
            if layout == "per_group" and group and group in layout_table.columns:
                layout_truth = truth.filter(
                    pl.col(group).is_in(layout_table[group].unique().to_list())
                )
            target, treatment = (
                self._target(layout_truth),
                self._treatment(layout_truth),
            )
            layout_suffix = f"_{layout}" if layout is not None else ""
            for name, function in (("qini", qini_curve), ("uplift", uplift_curve)):
                learners = {
                    learner: function(
                        target, layout_table[effect].to_numpy(), treatment
                    )
                    for learner, (effect, _, _) in _LEARNER_COLUMNS.items()
                }
                curves[f"{name}_curves{layout_suffix}"] = CurveData(
                    name, layout_suffix, learners
                )
        return curves

    def _evaluate_data(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | CalibrationResult | ParquetPath,
        evaluation_kind: EvaluationKind,
        metric_names: tuple[str, ...],
    ) -> EvaluationData:
        truth = self._normalize_treatment(self._read(test_path), required=True)
        self._target(truth)
        config = self._internal_config

        def aligned_score_frame(
            frame: pl.DataFrame, *, warn_duplicates: bool = True
        ) -> pl.DataFrame:
            return align_prediction_scores(
                frame,
                truth,
                config,
                self._column_mapper,
                score_columns=UPLIFT_SCORE_COLUMNS,
                warn_duplicates=warn_duplicates,
            )

        raw_frame = aligned_score_frame(self._score_input(scores))
        self._validate_score_matrix(raw_frame.select(UPLIFT_SCORE_COLUMNS).to_numpy())
        metrics, grouped = self._evaluate_table(truth, raw_frame, metric_names)

        importance: list[pl.DataFrame] = []
        for item in self._models:
            table = item.backend.feature_importance(item.schema)
            if table is not None:
                importance.append(
                    table.with_columns(
                        pl.lit(item.layout).alias("model_layout"),
                        pl.lit(
                            None if item.group_value is None else str(item.group_value),
                            dtype=pl.String,
                        ).alias("group"),
                        pl.col("component").str.slice(0, 1).alias("learner"),
                        pl.col("feature")
                        .replace(self._column_mapper.to_external)
                        .alias("feature"),
                    ).select(
                        "model_layout",
                        "group",
                        "learner",
                        "component",
                        "feature",
                        "importance",
                    )
                )
        importance_table = (
            pl.concat(importance).sort(
                ["model_layout", "group", "learner", "component", "importance"],
                descending=[False, False, False, False, True],
            )
            if importance
            else None
        )
        result = EvaluationResult.for_kind(
            evaluation_kind,
            metrics=metrics,
            metrics_by_group=None
            if grouped is None
            else self._column_mapper.restore_frame(grouped),
        )

        return EvaluationData(
            result=result,
            task_name=self._task_name,
            n_samples=raw_frame.height,
            grouped=grouped,
            date_column=self._internal_config.date_column,
            date_label=self.config.date_column,
            group_column=self._internal_config.group_column,
            feature_importance=importance_table,
            curves=self._curve_data(truth, raw_frame),
        )

    def _task_manifest(self) -> dict[str, Any]:
        return {
            "learner_metadata": {
                "learners": ["s", "t", "x"],
                "estimate_propensity": self.config.estimate_propensity,
            },
        }

    def _restore_task_manifest(self, manifest: Mapping[str, Any]) -> None:
        return None

    def _reset_task_state(self) -> None:
        """Clear task-specific transient state after failed training."""
