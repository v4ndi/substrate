"""Path-based multiclass boosting task with stable probability columns."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from avatar.automl.backends.boosting import MulticlassBoostingBackend
from avatar.automl.config import MulticlassTaskConfig
from avatar.automl.data import ParquetSource
from avatar.automl.exceptions import SchemaError
from avatar.automl.metrics import MetricInput, resolve_evaluation_metrics
from avatar.automl.reporting import ConfusionData, EvaluationData
from avatar.automl.tasks.base import BaseBoostingTask, _ModelEntry
from avatar.automl.tasks.evaluation import (
    align_prediction_scores,
    combine_metric_slices,
    feature_importance_table,
    metric_slices,
    prepare_evaluation_truth,
)
from avatar.automl.tasks.supervised import SupervisedBoostingTask
from avatar.automl.types import EvaluationKind, EvaluationResult, ParquetPath, PredictionResult

logger = logging.getLogger(__name__)


class MulticlassTask(SupervisedBoostingTask[MulticlassBoostingBackend]):
    """Train, score, evaluate, and persist multiclass boosting models.

    Labels may be strings, integers, or finite floats. The fitted global label
    order determines the stable ``score_<index>`` probability columns.

    Args:
        config: Validated multiclass-task configuration.
    """

    _backend_class = MulticlassBoostingBackend
    _config_class = MulticlassTaskConfig
    _task_name = "multiclass"
    _artifact_directory = "multiclass_model"

    def __init__(self, config: MulticlassTaskConfig, *, _entity_path: Path | None = None):
        """Initialize multiclass-task orchestration.

        Args:
            config: Validated multiclass-task configuration.
        """
        super().__init__(config, _entity_path=_entity_path)
        self._class_order: tuple[Any, ...] | None = None
        self._target_dtype: str | None = None
        self._target_encoding: list[tuple[Any, int]] = []

    @property
    def class_order(self) -> tuple[Any, ...] | None:
        """Return the label order used by ``score_<index>`` columns.

        Returns:
            Fitted labels in probability-column order, or ``None`` before training.
        """
        return self._class_order

    @staticmethod
    def _validate_scalar_target(series: pl.Series, public_name: str) -> None:
        supported_dtype = series.dtype == pl.String or series.dtype.is_integer() or series.dtype.is_float()
        if not supported_dtype:
            msg = (
                f"Multiclass target {public_name!r} must contain scalar string, integer or finite float labels; "
                f"got {series.dtype}"
            )
            raise SchemaError(msg)
        if series.null_count():
            msg = f"Multiclass target {public_name!r} contains null values"
            raise SchemaError(msg)
        if series.dtype.is_float():
            values = series.cast(pl.Float64)
            if values.is_nan().any() or values.is_infinite().any():
                msg = f"Multiclass target {public_name!r} contains NaN or infinite values"
                raise SchemaError(msg)

    def _label_index(self, label: Any) -> int | None:
        for known, encoded in self._target_encoding:
            if type(label) is type(known) and label == known:
                return encoded
        return None

    def _prepare_training_state(self, train_frame: pl.DataFrame, valid_frame: pl.DataFrame) -> None:
        column = self._internal_config.target_column
        if column not in train_frame.columns:
            msg = f"Missing target column: {self.config.target_column!r}"
            raise SchemaError(msg)
        if column not in valid_frame.columns:
            msg = f"Validation data is missing target column: {self.config.target_column!r}"
            raise SchemaError(msg)
        train_target = train_frame[column]
        self._validate_scalar_target(train_target, self.config.target_column)
        try:
            labels = tuple(train_target.unique().sort().to_list())
        except Exception as exc:
            msg = f"Multiclass target labels must have a deterministic sortable order: {exc}"
            raise SchemaError(msg) from exc
        if len(labels) < 3:
            msg = f"Multiclass training target requires at least three classes; got {list(labels)!r}"
            raise SchemaError(msg)
        self._class_order = labels
        self._target_dtype = str(train_target.dtype)
        self._target_encoding = [(label, index) for index, label in enumerate(labels)]
        self._validate_known_target(valid_frame, split="validation")

    def _validate_known_target(self, frame: pl.DataFrame, *, split: str) -> None:
        column = self._internal_config.target_column
        if column not in frame.columns:
            msg = f"{split.capitalize()} data is missing target column: {self.config.target_column!r}"
            raise SchemaError(msg)
        target = frame[column]
        self._validate_scalar_target(target, self.config.target_column)
        unknown = [value for value in target.unique().to_list() if self._label_index(value) is None]
        if unknown:
            msg = f"{split.capitalize()} target contains labels absent from training class_order: {unknown!r}"
            raise SchemaError(msg)

    def _target(self, frame: pl.DataFrame) -> np.ndarray:
        if self._class_order is None:
            msg = "Multiclass target schema is not fitted"
            raise SchemaError(msg)
        self._validate_known_target(frame, split="input")
        encoded = [self._label_index(value) for value in frame[self._internal_config.target_column].to_list()]
        return np.asarray(encoded, dtype=np.int64)

    def _fit_one(
        self,
        train_frame: pl.DataFrame,
        valid_frame: pl.DataFrame,
        *,
        layout: str,
        group_value: Any | None = None,
    ) -> _ModelEntry:
        part = self._model_name(layout, group_value)
        train_codes = self._target(train_frame)
        present = set(np.unique(train_codes).tolist())
        expected = set(range(len(self._class_order or ())))
        if present != expected:
            missing = [self._class_order[index] for index in sorted(expected - present)]
            msg = f"Multiclass model part {part!r} cannot be trained because target classes are missing: {missing!r}"
            raise SchemaError(msg)
        valid_codes = self._target(valid_frame)
        if np.unique(valid_codes).size < 2:
            msg = f"Multiclass validation for model part {part!r} must contain at least two classes"
            raise SchemaError(msg)
        if self.config.optimization_metric == "roc_auc_ovr_macro" and set(np.unique(valid_codes).tolist()) != expected:
            missing = [self._class_order[index] for index in sorted(expected - set(np.unique(valid_codes).tolist()))]
            msg = f"roc_auc_ovr_macro is undefined for validation model part {part!r}; missing classes: {missing!r}"
            raise SchemaError(msg)
        return super()._fit_one(train_frame, valid_frame, layout=layout, group_value=group_value)

    def _backend_options(self) -> Mapping[str, Any]:
        return {"num_classes": len(self._class_order or ())}

    def _metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        return super()._metric(target, scores)

    def _optimization_metric(self, target: np.ndarray, scores: np.ndarray) -> float:
        return self._metric(target, scores)

    @staticmethod
    def _score_columns(class_count: int) -> list[str]:
        return [f"score_{index}" for index in range(class_count)]

    def _probability_frame(self, base: pl.DataFrame, probabilities: np.ndarray) -> pl.DataFrame:
        if probabilities.shape != (base.height, len(self._class_order or ())):
            msg = (
                "Multiclass score matrix shape mismatch: "
                f"expected {(base.height, len(self._class_order or ()))}, got {probabilities.shape}"
            )
            raise SchemaError(msg)
        columns = [
            pl.Series(name, probabilities[:, index]) for index, name in enumerate(self._score_columns(probabilities.shape[1]))
        ]
        return base.with_columns(columns)

    @staticmethod
    def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(values, dtype=float)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            msg = "Multiclass probabilities must be finite and lie in [0, 1]"
            raise SchemaError(msg)
        totals = probabilities.sum(axis=1)
        if np.any(totals <= 0):
            msg = "Multiclass probability rows must have a positive sum"
            raise SchemaError(msg)
        probabilities = probabilities / totals[:, None]
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
            msg = "Multiclass probability rows could not be normalized to one"
            raise SchemaError(msg)
        return probabilities

    def _score_prediction_branch(
        self,
        frame: pl.DataFrame,
    ) -> np.ndarray:
        """Normalize raw probabilities using the fitted class order."""
        return self._normalize_probabilities(self._predict_frame(frame))

    def _prediction_result(self, frame: pl.DataFrame, raw: np.ndarray) -> PredictionResult:
        """Assemble class probabilities and class-order metadata."""
        config = self._internal_config
        columns = ["__row_id", config.client_id_column]
        if config.date_column is not None:
            columns.append(config.date_column)
        base = frame.select(columns).sort("__row_id").drop("__row_id")
        scores = self._probability_frame(base, raw)
        return PredictionResult(
            scores=self._column_mapper.restore_frame(scores),
            class_order=self._class_order,
        )

    def _validate_score_frame(self, frame: pl.DataFrame) -> np.ndarray:
        expected = self._score_columns(len(self._class_order or ()))
        missing = sorted(set(expected) - set(frame.columns))
        if missing:
            msg = f"Multiclass scores are missing required columns: {missing}"
            raise SchemaError(msg)
        return self._normalize_probabilities(frame.select(expected).to_numpy())

    def _metric_values(
        self,
        evaluated: pl.DataFrame,
        metric_names: tuple[str, ...],
        *,
        grouped: bool,
    ) -> dict[str, float | int | None]:
        target = self._target(evaluated)
        probabilities = self._validate_score_frame(evaluated)
        class_order = self._class_order or ()
        selected = resolve_evaluation_metrics(metric_names, self._task_name)
        values: dict[str, float | int | None] = {}
        class_wise_names = {
            "class_wise_roc_auc",
            *(f"precision@{k}" for k in (5, 10, 20, 25, 50)),
            *(f"recall@{k}" for k in (5, 10, 20, 25, 50)),
        }
        for metric in selected:
            if metric.name in class_wise_names:
                output_name = "roc_auc" if metric.name == "class_wise_roc_auc" else metric.name
                for index in range(probabilities.shape[1]):
                    binary_target = (target == index).astype(np.int8)
                    values[f"n_positives_class_{index}"] = int(binary_target.sum())
                    if metric.name == "class_wise_roc_auc" and np.unique(binary_target).size != 2:
                        value = None
                        if grouped:
                            logger.warning(
                                "Grouped class-wise ROC AUC is undefined and will be null; class_index=%d, class_label=%r",
                                index,
                                class_order[index],
                            )
                    else:
                        value = metric.compute(MetricInput(binary_target, probabilities[:, index]))
                    values[f"{output_name}_class_{index}"] = value
                continue
            if metric.name == "roc_auc_ovr_macro" and set(np.unique(target)) != set(range(len(class_order))):
                if not grouped:
                    missing = [class_order[index] for index in sorted(set(range(len(class_order))) - set(np.unique(target)))]
                    msg = f"Overall multiclass ROC AUC is undefined because target classes are missing: {missing!r}"
                    raise SchemaError(msg)
                value = None
                logger.warning("Grouped multiclass ROC AUC is undefined and will be null because target classes are missing")
            else:
                value = metric.compute(MetricInput(target, probabilities, class_order=class_order))
            values[metric.name] = value
        return values

    def _metric_table(
        self,
        evaluated: pl.DataFrame,
        metric_names: tuple[str, ...],
    ) -> tuple[dict[str, float | int], pl.DataFrame | None]:
        if "model_layout" in evaluated.columns:
            overall: dict[str, float | int] = {}
            grouped: list[pl.DataFrame] = []
            for key, branch in evaluated.partition_by("model_layout", as_dict=True, maintain_order=True).items():
                scope = key[0] if isinstance(key, tuple) else key
                branch_metrics, branch_grouped = self._metric_table(branch.drop("model_layout"), metric_names)
                overall.update({f"{scope}_{name}": value for name, value in branch_metrics.items()})
                if branch_grouped is not None:
                    grouped.append(branch_grouped.with_columns(pl.lit(scope).alias("model_layout")))
            return overall, combine_metric_slices(grouped, self._internal_config)
        config = self._internal_config
        overall_values = self._metric_values(evaluated, metric_names, grouped=False)
        overall = {
            name: value if isinstance(value, int) else float(value)
            for name, value in overall_values.items()
            if value is not None
        }
        tables: list[pl.DataFrame] = []
        for slice_name, group_columns in metric_slices(config):
            rows = []
            for keys, group in evaluated.group_by(group_columns, maintain_order=True):
                values = keys if isinstance(keys, tuple) else (keys,)
                rows.append(
                    {"scope": slice_name, **dict(zip(group_columns, values, strict=True))}
                    | {"n_samples": group.height}
                    | self._metric_values(group, metric_names, grouped=True)
                )
            table = pl.DataFrame(rows)
            if config.date_column in group_columns:
                table = table.with_columns(pl.col(config.date_column).dt.strftime("%Y-%m-%d"))
            tables.append(table.sort(group_columns))
        return overall, combine_metric_slices(tables, config)

    def _evaluation_frames(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | ParquetPath,
    ) -> pl.DataFrame:
        config = self._internal_config
        truth = prepare_evaluation_truth(self._normalize_model_frame(self._read(test_path)), config, self._models[0])
        if isinstance(scores, PredictionResult):
            if scores.class_order is not None and tuple(scores.class_order) != self._class_order:
                msg = f"PredictionResult class_order mismatch: expected {self._class_order!r}, got {scores.class_order!r}"
                raise SchemaError(msg)
            external = scores.scores
        else:
            external = ParquetSource.resolve(scores).read()
        self._validate_known_target(truth, split="evaluation")

        def joined(external: pl.DataFrame, *, warn_duplicates: bool = True) -> pl.DataFrame:
            return align_prediction_scores(
                external,
                truth,
                config,
                self._column_mapper,
                score_columns=self._score_columns(len(self._class_order or ())),
                warn_duplicates=warn_duplicates,
            )

        return joined(external)

    def _metrics_by_class_table(
        self,
        metrics: Mapping[str, float | int],
        metric_names: tuple[str, ...],
    ) -> pl.DataFrame | None:
        """Build a readable per-class table while retaining the fitted label dtype."""
        scopes = [scope for scope in ("global", "per_group") if any(name.startswith(f"{scope}_") for name in metrics)]
        if scopes:
            return pl.concat(
                [
                    self._metrics_by_class_table(
                        {
                            name.removeprefix(f"{scope}_"): value
                            for name, value in metrics.items()
                            if name.startswith(f"{scope}_")
                        },
                        metric_names,
                    ).with_columns(pl.lit(scope).alias("model_layout"))
                    for scope in scopes
                ],
                how="diagonal_relaxed",
            )
        class_metrics = [name for name in metric_names if name == "class_wise_roc_auc" or "@" in name]
        if not class_metrics:
            return None
        class_count = len(self._class_order or ())
        label_dtype = getattr(pl, self._target_dtype or "", None)
        labels = pl.Series("class_label", list(self._class_order or ()), dtype=label_dtype)
        values: dict[str, Any] = {"class_index": range(class_count)}
        if "class_wise_roc_auc" in class_metrics:
            values["roc_auc"] = [metrics[f"roc_auc_class_{index}"] for index in range(class_count)]
        values["n_positives"] = [metrics[f"n_positives_class_{index}"] for index in range(class_count)]
        for name in class_metrics:
            if name == "class_wise_roc_auc":
                continue
            output_name = "roc_auc" if name == "class_wise_roc_auc" else name
            values[output_name] = [metrics[f"{output_name}_class_{index}"] for index in range(class_count)]
        return pl.DataFrame(values).insert_column(1, labels)

    def _confusion_data(self, evaluated: pl.DataFrame, *, suffix: str = "") -> dict[str, ConfusionData]:
        """Compute normalized matrices without importing a plotting backend."""
        if "model_layout" in evaluated.columns:
            matrices: dict[str, ConfusionData] = {}
            for key, branch in evaluated.partition_by("model_layout", as_dict=True, maintain_order=True).items():
                scope = key[0] if isinstance(key, tuple) else key
                matrices.update(self._confusion_data(branch.drop("model_layout"), suffix=f"_{scope}"))
            return matrices
        from sklearn.metrics import confusion_matrix

        target = self._target(evaluated)
        predicted = self._validate_score_frame(evaluated).argmax(axis=1)
        labels = tuple(str(value) for value in self._class_order or ())
        matrix = confusion_matrix(target, predicted, labels=np.arange(len(labels)), normalize="true")
        return {f"multiclass_confusion_matrix{suffix}": ConfusionData(matrix, labels, suffix)}

    def _evaluate_data(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | ParquetPath,
        evaluation_kind: EvaluationKind,
        metric_names: tuple[str, ...],
    ) -> EvaluationData:
        self._require_fitted()
        evaluated = self._evaluation_frames(test_path, scores)
        metrics, grouped = self._metric_table(evaluated, metric_names)
        metrics_by_class = self._metrics_by_class_table(metrics, metric_names)
        importance_table = feature_importance_table(self._models, self._column_mapper, self._model_name)
        result = EvaluationResult.for_kind(
            evaluation_kind,
            metrics=metrics,
            metrics_by_group=None if grouped is None else self._column_mapper.restore_frame(grouped),
            metrics_by_class=metrics_by_class,
        )

        return EvaluationData(
            result=result,
            task_name=self._task_name,
            n_samples=evaluated.height,
            grouped=grouped,
            date_column=self._internal_config.date_column,
            date_label=self.config.date_column,
            group_column=self._internal_config.group_column,
            feature_importance=importance_table,
            confusions=self._confusion_data(evaluated),
        )

    def _copy_task_state(self, other: BaseBoostingTask) -> None:
        if self._class_order is not None and other._class_order is not None and self._class_order != other._class_order:
            msg = "Remote multiclass model parts have inconsistent class_order values"
            raise SchemaError(msg)
        self._class_order = other._class_order
        self._target_dtype = other._target_dtype
        self._target_encoding = list(other._target_encoding)

    def _merge_task_state(self, other: BaseBoostingTask) -> None:
        if self._class_order is None:
            self._class_order = other._class_order
            self._target_dtype = other._target_dtype
            self._target_encoding = list(other._target_encoding)
        elif self._class_order != other._class_order:
            msg = "Remote multiclass model parts have inconsistent class_order values"
            raise SchemaError(msg)

    def _task_manifest(self) -> dict[str, Any]:
        return {
            "target_schema": {"dtype": self._target_dtype, "class_order": list(self._class_order or ())},
            "class_order": list(self._class_order or ()),
            "target_label_encoding": [{"label": label, "encoded": encoded} for label, encoded in self._target_encoding],
        }

    def _restore_task_manifest(self, manifest: Mapping[str, Any]) -> None:
        target_schema = manifest.get("target_schema")
        if not isinstance(target_schema, Mapping) or len(target_schema.get("class_order", ())) < 3:
            msg = "Multiclass artifact is missing a valid fitted target schema"
            raise SchemaError(msg)
        self._target_dtype = str(target_schema["dtype"])
        self._class_order = tuple(target_schema["class_order"])
        encoding = manifest.get("target_label_encoding", ())
        self._target_encoding = [(item["label"], int(item["encoded"])) for item in encoding]
        if len(self._target_encoding) != len(self._class_order):
            msg = "Multiclass artifact target label encoding is inconsistent with class_order"
            raise SchemaError(msg)

    def _reset_task_state(self) -> None:
        """Clear learned target encoding state."""
        self._class_order = None
        self._target_dtype = None
        self._target_encoding = []
