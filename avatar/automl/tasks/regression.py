"""Path-based orchestration for boosting regression tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from avatar.automl.backends.boosting import RegressionBoostingBackend
from avatar.automl.config import RegressionTaskConfig
from avatar.automl.exceptions import SchemaError
from avatar.automl.metrics import MetricInput, resolve_evaluation_metrics
from avatar.automl.reporting import EvaluationData
from avatar.automl.tasks.evaluation import (
    align_scalar_scores,
    combine_metric_slices,
    feature_importance_table,
    metric_slices,
    prepare_evaluation_truth,
    scalar_prediction_result,
    scalar_score_inputs,
)
from avatar.automl.tasks.supervised import SupervisedBoostingTask
from avatar.automl.types import EvaluationKind, EvaluationResult, ParquetPath, PredictionResult

logger = logging.getLogger(__name__)


class RegressionTask(SupervisedBoostingTask[RegressionBoostingBackend]):
    """Train, score, evaluate, and persist continuous-target boosting models.

    Args:
        config: Validated regression-task configuration.
    """

    _backend_class = RegressionBoostingBackend
    _config_class = RegressionTaskConfig
    _task_name = "regression"
    _artifact_directory = "regression_model"

    def __init__(self, config: RegressionTaskConfig, *, _entity_path: Path | None = None):
        """Initialize regression-task orchestration.

        Args:
            config: Validated regression-task configuration.
        """
        super().__init__(config, _entity_path=_entity_path)

    def _target(self, frame: pl.DataFrame) -> np.ndarray:
        """Return a finite, non-null numerical target without lossy coercion."""
        config = self._internal_config
        if config.target_column not in frame.columns:
            msg = f"Missing target column: {self.config.target_column!r}"
            raise SchemaError(msg)
        target = frame[config.target_column]
        if not target.dtype.is_numeric():
            msg = f"Regression target must be numerical; got {target.dtype}"
            raise SchemaError(msg)
        if target.null_count():
            msg = f"Target column {self.config.target_column!r} contains null values"
            raise SchemaError(msg)
        if target.cast(pl.Float64).is_infinite().any():
            msg = f"Target column {self.config.target_column!r} contains infinite values"
            raise SchemaError(msg)
        values = target.to_numpy()
        if not np.isfinite(values).all():
            msg = f"Target column {self.config.target_column!r} contains non-finite values"
            raise SchemaError(msg)
        return values

    def _metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        """Calculate the configured registered regression metric."""
        return super()._metric(target, scores)

    def _optimization_metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        """Calculate the explicitly directed MSE/MAE Optuna objective."""
        return self._metric(target, scores)

    def _prediction_result(self, frame: pl.DataFrame, raw: np.ndarray) -> PredictionResult:
        """Assemble scalar raw scores with the configured roles."""
        return scalar_prediction_result(
            frame,
            raw,
            self._internal_config,
            self._column_mapper,
        )

    def _metric_table(
        self,
        evaluated: pl.DataFrame,
        metric_names: tuple[str, ...],
    ) -> tuple[dict[str, float], pl.DataFrame | None]:
        """Compute overall and configured date/group-slice MSE, MAE and MAPE."""
        if "model_layout" in evaluated.columns:
            overall: dict[str, float] = {}
            grouped: list[pl.DataFrame] = []
            for key, branch in evaluated.partition_by("model_layout", as_dict=True, maintain_order=True).items():
                scope = key[0] if isinstance(key, tuple) else key
                branch_metrics, branch_grouped = self._metric_table(branch.drop("model_layout"), metric_names)
                overall.update({f"{scope}_{name}": value for name, value in branch_metrics.items()})
                if branch_grouped is not None:
                    grouped.append(branch_grouped.with_columns(pl.lit(scope).alias("model_layout")))
            return overall, combine_metric_slices(grouped, self._internal_config)
        config = self._internal_config
        selected = resolve_evaluation_metrics(metric_names, self._task_name)
        target = self._target(evaluated)
        scores = evaluated["score"].to_numpy()
        overall = {metric.name: float(metric.compute(MetricInput(target, scores))) for metric in selected}
        tables: list[pl.DataFrame] = []
        for slice_name, group_columns in metric_slices(config):
            rows: list[dict[str, Any]] = []
            for keys, group in evaluated.group_by(group_columns, maintain_order=True):
                key_values = keys if isinstance(keys, tuple) else (keys,)
                rows.append(
                    {"scope": slice_name, **dict(zip(group_columns, key_values, strict=True))}
                    | {"n_samples": group.height}
                    | {
                        metric.name: float(metric.compute(MetricInput(self._target(group), group["score"].to_numpy())))
                        for metric in selected
                    }
                )
            table = pl.DataFrame(rows)
            if config.date_column in group_columns:
                table = table.with_columns(pl.col(config.date_column).dt.strftime("%Y-%m-%d"))
            tables.append(table.sort(group_columns))
        return overall, combine_metric_slices(tables, config)

    def _evaluate_data(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | ParquetPath,
        evaluation_kind: EvaluationKind,
        metric_names: tuple[str, ...],
    ) -> EvaluationData:
        """Compute continuous-score metrics and report data without writing files."""
        self._require_fitted()
        config = self._internal_config
        truth = prepare_evaluation_truth(self._normalize_model_frame(self._read(test_path)), config, self._models[0])
        raw_external = scalar_score_inputs(scores)
        evaluated = align_scalar_scores(raw_external, truth, config, self._column_mapper)
        metrics, metrics_by_group = self._metric_table(evaluated, metric_names)

        importance_table = feature_importance_table(self._models, self._column_mapper, self._model_name)
        result = EvaluationResult.for_kind(
            evaluation_kind,
            metrics=metrics,
            metrics_by_group=None if metrics_by_group is None else self._column_mapper.restore_frame(metrics_by_group),
        )

        return EvaluationData(
            result=result,
            task_name=self._task_name,
            n_samples=evaluated.height,
            grouped=metrics_by_group,
            date_column=self._internal_config.date_column,
            date_label=self.config.date_column,
            group_column=self._internal_config.group_column,
            feature_importance=importance_table,
        )
