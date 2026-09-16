"""Public result containers and path types for the AutoML API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

import polars as pl

ParquetPath: TypeAlias = str | Path
EvaluationKind: TypeAlias = Literal["raw", "calibrated"]


@dataclass(frozen=True)
class TrainingResult:
    """Describe a completed training or hyperparameter-search run.

    Attributes:
        best_params: Parameters of the selected estimator.
        validation_metrics: Metrics calculated on the validation split.
        feature_names: Ordered feature names consumed by the estimator.
        backend_name: Backend family used for training, for example ``boosting``.
        engine_name: Concrete estimator engine, for example ``catboost``.
        task_name: Task kind such as ``binary`` or ``uplift``.
    """

    best_params: Mapping[str, Any]
    validation_metrics: Mapping[str, float]
    feature_names: tuple[str, ...]
    backend_name: str
    engine_name: str
    task_name: str


@dataclass(frozen=True)
class PredictionResult:
    """Contain row-level model scores and their identifying columns.

    Attributes:
        scores: Raw scores returned to the caller. For
            ``model_layout="global_and_per_group"``, global and
            per-group rows are returned independently and identified by the
            ``model_layout`` column.
        class_order: Labels corresponding to multiclass score columns, or ``None``
            for tasks with a scalar score.
    """

    scores: pl.DataFrame
    class_order: tuple[Any, ...] | None = None

    def to_pandas(self):
        """Convert public scores to a pandas DataFrame.

        Returns:
            A pandas DataFrame with the same rows and columns as :attr:`scores`.
        """
        return self.scores.to_pandas()


@dataclass(frozen=True)
class CalibrationResult:
    """Contain calibrated row-level scores produced from parquet inputs."""

    scores: pl.DataFrame
    calibration_strategy: str

    def to_pandas(self):
        """Convert calibrated scores to a pandas DataFrame."""
        return self.scores.to_pandas()


@dataclass(frozen=True)
class EvaluationResult:
    """Contain metrics and reports produced by task evaluation.

    Attributes:
        Fields ending in ``_raw`` describe a PredictionResult evaluation.
        Fields ending in ``_calibrated`` describe a CalibrationResult
        evaluation. Both variants may coexist after two evaluate calls for the
        same test path.
    """

    metrics_raw: Mapping[str, float] | None = None
    metrics_calibrated: Mapping[str, float] | None = None
    metrics_by_group_raw: pl.DataFrame | None = None
    metrics_by_group_calibrated: pl.DataFrame | None = None
    metrics_by_class_raw: pl.DataFrame | None = None
    metrics_by_class_calibrated: pl.DataFrame | None = None
    figures_raw: Mapping[str, Any] = field(default_factory=dict)
    figures_calibrated: Mapping[str, Any] = field(default_factory=dict)
    excel_paths_raw: Mapping[str, Path] = field(default_factory=dict)
    excel_paths_calibrated: Mapping[str, Path] = field(default_factory=dict)

    @classmethod
    def for_kind(
        cls,
        kind: EvaluationKind,
        *,
        metrics: Mapping[str, float],
        metrics_by_group: pl.DataFrame | None = None,
        metrics_by_class: pl.DataFrame | None = None,
    ) -> "EvaluationResult":
        """Create the half of a result produced by one evaluate call."""
        return cls(
            **{
                f"metrics_{kind}": metrics,
                f"metrics_by_group_{kind}": metrics_by_group,
                f"metrics_by_class_{kind}": metrics_by_class,
            }
        )

    def merge(self, other: "EvaluationResult") -> "EvaluationResult":
        """Replace only the raw/calibrated halves present in ``other``."""
        values: dict[str, Any] = {}
        for kind in ("raw", "calibrated"):
            present = getattr(other, f"metrics_{kind}") is not None
            source = other if present else self
            for prefix in (
                "metrics",
                "metrics_by_group",
                "metrics_by_class",
                "figures",
                "excel_paths",
            ):
                name = f"{prefix}_{kind}"
                values[name] = getattr(source, name)
        return EvaluationResult(**values)
