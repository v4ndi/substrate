"""Binary-classification boosting task facade."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl

from fmlib.automl.backends.boosting import BinaryBoostingBackend
from fmlib.automl.config import BinaryTaskConfig
from fmlib.automl.exceptions import SchemaError
from fmlib.automl.metrics import MetricInput, resolve_evaluation_metrics
from fmlib.automl.reporting import EvaluationData
from fmlib.automl.tasks.evaluation import (
    align_scalar_scores,
    combine_metric_slices,
    feature_importance_table,
    metric_slices,
    prepare_evaluation_truth,
    scalar_prediction_result,
    scalar_score_inputs,
)
from fmlib.automl.tasks.supervised import SupervisedTask
from fmlib.automl.types import (
    CalibrationResult,
    EvaluationKind,
    EvaluationResult,
    ParquetPath,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class BinaryTask(SupervisedTask[BinaryBoostingBackend]):
    """Train, score, evaluate, and persist binary boosting models.

    Global, per-group, or combined layouts are selected through the configuration.
    Prediction returns probabilities rather than thresholded labels. Evaluation
    includes ROC AUC, precision@k, recall@k, plots, and Excel reports.

    Args:
        config: Validated binary-task configuration.
    """

    _backend_loaders: ClassVar[Mapping[str, Callable[[], type]]] = {
        "boosting": lambda: BinaryBoostingBackend,
    }
    _config_class = BinaryTaskConfig
    _task_name = "binary"
    _artifact_directory = "binary_model"

    def __init__(self, config: BinaryTaskConfig, *, _entity_path: Path | None = None):
        """Initialize binary-task orchestration.

        Args:
            config: Validated binary-task configuration.
        """
        super().__init__(config, _entity_path=_entity_path)

    def _target(self, frame: pl.DataFrame) -> np.ndarray:
        """Validate and return the binary target."""
        config = self._internal_config
        if config.target_column not in frame.columns:
            msg = f"Missing target column: {self.config.target_column!r}"
            raise SchemaError(msg)
        target = frame[config.target_column]
        if target.null_count():
            msg = f"Target column {self.config.target_column!r} contains null values"
            raise SchemaError(msg)
        values = set(target.unique().to_list())
        if values != {0, 1}:
            msg = f"Binary target must contain exactly 0 and 1; got {sorted(values, key=str)}"
            raise SchemaError(msg)
        return target.cast(pl.Int8).to_numpy()

    def _metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        """Calculate the configured binary metric."""
        return super()._metric(target, scores)

    def _optimization_metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        """Calculate the registered Binary/Response Optuna metric."""
        return self._metric(target, scores)

    def _prediction_result(
        self, frame: pl.DataFrame, raw: np.ndarray
    ) -> PredictionResult:
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
        """Calculate binary metrics independently for every returned model branch."""
        if "model_layout" in evaluated.columns:
            overall: dict[str, float] = {}
            grouped: list[pl.DataFrame] = []
            for key, branch in evaluated.partition_by(
                "model_layout", as_dict=True, maintain_order=True
            ).items():
                scope = key[0] if isinstance(key, tuple) else key
                branch_metrics, branch_grouped = self._metric_table(
                    branch.drop("model_layout"), metric_names
                )
                overall.update({
                    f"{scope}_{name}": value for name, value in branch_metrics.items()
                })
                if branch_grouped is not None:
                    grouped.append(
                        branch_grouped.with_columns(pl.lit(scope).alias("model_layout"))
                    )
            return overall, combine_metric_slices(grouped, self._internal_config)

        config = self._internal_config
        target = self._target(evaluated)
        values = evaluated["score"].to_numpy()
        selected = resolve_evaluation_metrics(metric_names, self._task_name)
        overall = {
            metric.name: float(metric.compute(MetricInput(target, values)))
            for metric in selected
        }
        tables: list[pl.DataFrame] = []
        for slice_name, group_columns in metric_slices(config):
            rows: list[dict[str, Any]] = []
            for keys, group in evaluated.group_by(group_columns, maintain_order=True):
                key_values = keys if isinstance(keys, tuple) else (keys,)
                group_target = group[config.target_column].cast(pl.Int8).to_numpy()
                group_scores = group["score"].to_numpy()
                metrics = {}
                for metric in selected:
                    metrics[metric.name] = (
                        float("nan")
                        if metric.name == "roc_auc"
                        and np.unique(group_target).size != 2
                        else float(
                            metric.compute(MetricInput(group_target, group_scores))
                        )
                    )
                rows.append(
                    {
                        "scope": slice_name,
                        **dict(zip(group_columns, key_values, strict=True)),
                    }
                    | {"n_samples": group.height}
                    | metrics
                )
            table = pl.DataFrame(rows)
            if config.date_column in group_columns:
                table = table.with_columns(
                    pl.col(config.date_column).dt.strftime("%Y-%m-%d")
                )
            tables.append(table.sort(group_columns))
        return overall, combine_metric_slices(tables, config)

    def _evaluate_data(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | CalibrationResult | ParquetPath,
        evaluation_kind: EvaluationKind,
        metric_names: tuple[str, ...],
    ) -> EvaluationData:
        """Compute metrics and report data without writing files.

        Targets are always read from the configured column in ``test_path``.
        Returned metrics include ROC AUC and precision/recall within configured
        top-score percentages. Slice metrics follow the configured optional date
        and group roles. Dates are returned and written to Excel as
        ``YYYY-MM-DD`` strings.
        """
        self._require_fitted()
        config = self._internal_config
        truth = prepare_evaluation_truth(
            self._normalize_model_frame(self._read(test_path)), config, self._models[0]
        )
        external_scores = scalar_score_inputs(scores)
        evaluated = align_scalar_scores(
            external_scores, truth, config, self._column_mapper
        )
        overall_metrics, metrics_by_group = self._metric_table(evaluated, metric_names)

        importance = feature_importance_table(
            self._models, self._column_mapper, self._model_name
        )
        result = EvaluationResult.for_kind(
            evaluation_kind,
            metrics=overall_metrics,
            metrics_by_group=None
            if metrics_by_group is None
            else self._column_mapper.restore_frame(metrics_by_group),
        )
        return EvaluationData(
            result=result,
            task_name=self._task_name,
            n_samples=evaluated.height,
            grouped=metrics_by_group,
            date_column=config.date_column,
            date_label=self.config.date_column,
            group_column=config.group_column,
            feature_importance=importance,
        )
