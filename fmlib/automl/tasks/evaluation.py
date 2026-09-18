"""Shared evaluation mechanics for supervised AutoML tasks."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import polars as pl

from fmlib.automl.config.base import BaseTaskConfig
from fmlib.automl.data import (
    CanonicalColumnMapper,
    ParquetSource,
    normalize_date,
    prepare_data,
)
from fmlib.automl.exceptions import SchemaError
from fmlib.automl.types import CalibrationResult, ParquetPath, PredictionResult

logger = logging.getLogger(__name__)


def metric_slices(config: BaseTaskConfig) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the required non-overall evaluation slices for a task config."""
    slices: list[tuple[str, tuple[str, ...]]] = []
    if config.group_column is not None:
        slices.append(("group", (config.group_column,)))
    if config.date_column is not None:
        columns = (
            (config.date_column, config.group_column)
            if config.group_column is not None
            else (config.date_column,)
        )
        slices.append((
            "date_group" if config.group_column is not None else "date",
            columns,
        ))
    return tuple(slices)


def combine_metric_slices(
    tables: Sequence[pl.DataFrame], config: BaseTaskConfig
) -> pl.DataFrame | None:
    """Combine heterogeneous slice tables with a stable role-first column order."""
    if not tables:
        return None
    combined = pl.concat(tables, how="diagonal_relaxed")
    leading = [
        name
        for name in ("scope", config.date_column, config.group_column, "model_layout")
        if name is not None and name in combined.columns
    ]
    return combined.select(*leading, pl.exclude(leading))


def prepare_evaluation_truth(
    frame: pl.DataFrame,
    config: BaseTaskConfig,
    model: Any,
) -> pl.DataFrame:
    """Validate evaluation truth against the schema of the routed artifact."""
    categorical_roles = (
        config.group_column if model.layout == "global" else None,
        getattr(config, "treatment_column", None),
    )
    excluded = (
        (config.group_column,)
        if model.layout == "per_group" and config.group_column
        else ()
    )
    prepared, _ = prepare_data(
        frame,
        config,
        require_target=True,
        fitted_schema=model.schema,
        hidden_dimensions=model.hidden_dimensions,
        categorical_role_columns=categorical_roles,
        excluded_feature_columns=excluded,
    )
    return prepared.frame


def scalar_score_inputs(
    scores: PredictionResult | CalibrationResult | ParquetPath,
) -> pl.DataFrame:
    """Return the explicitly selected scalar score frame."""
    if isinstance(scores, PredictionResult | CalibrationResult):
        return scores.scores
    return ParquetSource.resolve(scores).read()


def scalar_prediction_result(
    frame: pl.DataFrame,
    scores: Any,
    config: BaseTaskConfig,
    mapper: CanonicalColumnMapper,
) -> PredictionResult:
    """Build the common ordered scalar prediction result."""
    columns = ["__row_id", config.client_id_column]
    if config.date_column is not None:
        columns.append(config.date_column)
    result = (
        frame.select(columns)
        .with_columns(pl.Series("score", scores))
        .sort("__row_id")
        .drop("__row_id")
    )
    return PredictionResult(scores=mapper.restore_frame(result))


def align_scalar_scores(
    external_scores: pl.DataFrame,
    truth: pl.DataFrame,
    config: BaseTaskConfig,
    mapper: CanonicalColumnMapper,
    *,
    warn_duplicates: bool = True,
    operation_name: str = "Evaluation",
) -> pl.DataFrame:
    """Normalize and align scalar scores to uniquely keyed evaluation truth."""
    return align_prediction_scores(
        external_scores,
        truth,
        config,
        mapper,
        score_columns=("score",),
        warn_duplicates=warn_duplicates,
        operation_name=operation_name,
    )


def align_prediction_scores(
    external_scores: pl.DataFrame,
    truth: pl.DataFrame,
    config: BaseTaskConfig,
    mapper: CanonicalColumnMapper,
    *,
    score_columns: Sequence[str],
    truth_columns: Sequence[str] = (),
    warn_duplicates: bool = True,
    operation_name: str = "Evaluation",
) -> pl.DataFrame:
    """Align predictions to truth by identity key and occurrence order.

    Duplicate keys produce one warning per evaluation and are paired by
    occurrence order. Calibrated scores retain validation without repeating it.
    """
    if "model_scope" in external_scores.columns:
        msg = (
            "Scores use incompatible legacy model_scope metadata; expected model_layout"
        )
        raise SchemaError(msg)
    scores = mapper.normalize_frame(external_scores)
    if config.date_column is not None:
        scores = normalize_date(scores, config.date_column)
    truth_duplicates = _check_global_keys(
        truth,
        config,
        source=f"{operation_name.lower()} data",
        public_column_names=mapper.to_external,
    )
    score_duplicates = _check_global_keys(
        scores,
        config,
        source="scores",
        extra_keys=("model_layout",) if "model_layout" in scores.columns else (),
        public_column_names=mapper.to_external,
    )
    if warn_duplicates and (truth_duplicates or score_duplicates):
        identity_columns = [config.client_id_column]
        if config.date_column is not None:
            identity_columns.append(config.date_column)
        keys = [mapper.to_external.get(name, name) for name in identity_columns]
        logger.warning(
            "%s contains duplicate rows by %s; truth and selected scores are aligned by occurrence order. "
            "Duplicate keys include %s",
            operation_name,
            keys,
            truth_duplicates or score_duplicates,
        )
    if "model_layout" in scores.columns:
        branches = []
        for key, branch in scores.partition_by(
            "model_layout", as_dict=True, maintain_order=True
        ).items():
            scope = key[0] if isinstance(key, tuple) else key
            branch_truth = truth
            if (
                scope == "per_group"
                and config.group_column
                and config.group_column in branch.columns
            ):
                branch_truth = truth.filter(
                    pl.col(config.group_column).is_in(
                        branch[config.group_column].unique().to_list()
                    )
                )
            aligned = _align_prediction_branch(
                branch.drop("model_layout"),
                branch_truth,
                config,
                score_columns=score_columns,
                truth_columns=truth_columns,
            )
            branches.append(aligned.with_columns(pl.lit(scope).alias("model_layout")))
        return pl.concat(branches, how="diagonal_relaxed")
    return _align_prediction_branch(
        scores,
        truth,
        config,
        score_columns=score_columns,
        truth_columns=truth_columns,
    )


def _check_global_keys(
    frame: pl.DataFrame,
    config: BaseTaskConfig,
    *,
    source: str,
    extra_keys: Sequence[str] = (),
    public_column_names: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    join_columns = [config.client_id_column]
    if config.date_column is not None:
        join_columns.append(config.date_column)
    join_columns.extend(extra_keys)
    missing = sorted(set(join_columns) - set(frame.columns))
    if missing:
        visible = (
            [public_column_names.get(name, name) for name in missing]
            if public_column_names
            else missing
        )
        msg = (
            f"Scores are missing required columns: {visible}"
            if source == "scores"
            else f"{source.capitalize()} is missing identity key columns: {visible}"
        )
        raise SchemaError(msg)
    duplicates = (
        frame.group_by(join_columns)
        .len()
        .filter(pl.col("len") > 1)
        .select(join_columns)
        .head(3)
    )
    return duplicates.to_dicts()


def _align_prediction_branch(
    scores: pl.DataFrame,
    truth: pl.DataFrame,
    config: BaseTaskConfig,
    *,
    score_columns: Sequence[str],
    truth_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Align one branch after the complete truth and score tables were validated."""
    join_columns = [config.client_id_column]
    if config.date_column is not None:
        join_columns.append(config.date_column)
    missing = sorted({*join_columns, *score_columns} - set(scores.columns))
    if missing:
        msg = f"Scores are missing required columns: {missing}"
        raise SchemaError(msg)
    metric_columns = [
        column
        for column in (config.group_column,)
        if column and column in truth.columns and column not in join_columns
    ]
    occurrence_column = "__fmlib_key_occurrence"
    scores = scores.with_columns(
        pl.int_range(pl.len()).over(join_columns).alias(occurrence_column)
    )
    truth = truth.with_columns(
        pl.int_range(pl.len()).over(join_columns).alias(occurrence_column)
    )
    occurrence_keys = [*join_columns, occurrence_column]
    score_keys = scores.select(occurrence_keys)
    truth_keys = truth.select(occurrence_keys)
    missing_scores = truth_keys.join(score_keys, on=occurrence_keys, how="anti")
    extra_scores = score_keys.join(truth_keys, on=occurrence_keys, how="anti")
    if (
        scores.height != truth.height
        or not missing_scores.is_empty()
        or not extra_scores.is_empty()
    ):
        msg = f"Scores must match evaluation rows one-to-one by {join_columns}; expected {truth.height} rows, got {scores.height}"
        raise SchemaError(msg)
    selected_truth_columns = list(
        dict.fromkeys([
            *occurrence_keys,
            *metric_columns,
            config.target_column,
            *truth_columns,
        ])
    )
    truth_values = truth.select(selected_truth_columns).with_row_index(
        "__fmlib_truth_row"
    )
    return (
        truth_values.join(
            scores.select([*occurrence_keys, *score_columns]),
            on=occurrence_keys,
            how="left",
        )
        .sort("__fmlib_truth_row")
        .drop("__fmlib_truth_row", occurrence_column)
    )


def feature_importance_table(
    models: list[Any],
    mapper: CanonicalColumnMapper,
    model_name: Callable[[str, Any | None], str],
) -> pl.DataFrame | None:
    """Collect feature importance without serializing it."""
    frames: list[pl.DataFrame] = []
    for item in models:
        importance = item.backend.feature_importance(item.schema)
        if importance is not None:
            frames.append(
                importance.with_columns(
                    pl.col("feature").replace(mapper.to_external).alias("feature"),
                    pl.lit(model_name(item.layout, item.group_value)).alias("model"),
                )
            )
    if not frames:
        return None
    return pl.concat(frames).sort(["model", "importance"], descending=[False, True])
