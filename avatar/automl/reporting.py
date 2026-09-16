"""Render and export computed evaluation data without accessing tasks or models."""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from avatar.automl.types import EvaluationKind, EvaluationResult


@dataclass(frozen=True)
class ConfusionData:
    """A normalized confusion matrix with fitted class labels."""

    matrix: np.ndarray
    labels: tuple[str, ...]
    suffix: str = ""


@dataclass(frozen=True)
class CurveData:
    """Precomputed learner curves for one model layout."""

    name: str
    layout_suffix: str
    learners: Mapping[str, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class EvaluationData:
    """Computed public results and compact inputs needed to reproduce reports.

    Grouped plot tables retain canonical names for existing plot labels; the
    public result already contains configured physical column names.
    """

    result: EvaluationResult
    task_name: str
    n_samples: int
    grouped: pl.DataFrame | None
    date_column: str | None
    date_label: str | None
    group_column: str | None
    feature_importance: pl.DataFrame | None = None
    confusions: Mapping[str, ConfusionData] = field(default_factory=dict)
    curves: Mapping[str, CurveData] = field(default_factory=dict)


def export_evaluation(data: EvaluationData, output: Path) -> EvaluationResult:
    """Write existing Excel/PNG reports and attach their paths and figures."""
    output.mkdir(parents=True, exist_ok=True)
    result = data.result
    kinds = [kind for kind in ("raw", "calibrated") if getattr(result, f"metrics_{kind}") is not None]
    if len(kinds) != 1:
        msg = f"One evaluate call must contain exactly one score kind, got {kinds}"
        raise ValueError(msg)
    kind: EvaluationKind = kinds[0]
    metrics = getattr(result, f"metrics_{kind}")
    grouped = getattr(result, f"metrics_by_group_{kind}")
    by_class = getattr(result, f"metrics_by_class_{kind}")
    tables: dict[str, pl.DataFrame] = {}
    counts = {} if data.task_name == "uplift" else {"n_samples": data.n_samples}
    frames = [pl.DataFrame([{"scope": "overall", **counts, **metrics}])]
    if grouped is not None:
        frames.append(grouped)
    tables[f"metrics_{kind}"] = pl.concat(frames, how="diagonal_relaxed")
    for name, table in ((f"metrics_by_class_{kind}", by_class), (f"feature_importance_{kind}", data.feature_importance)):
        if table is not None:
            tables[name] = table
    paths = {}
    for name, table in tables.items():
        path = output / f"{name}.xlsx"
        table.write_excel(path, autofilter=True, autofit=True, float_precision=6)
        paths[name] = path
    return replace(
        result,
        **{
            f"excel_paths_{kind}": paths,
            f"figures_{kind}": render_evaluation(data, output, kind),
        },
    )


def render_evaluation(data: EvaluationData, output: Path, kind: EvaluationKind) -> dict[str, Any]:
    """Render prepared tables, matrices and curves using existing layouts."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    figures = {}
    if data.task_name in {"binary", "response"} and data.grouped is not None:
        figures.update(_plot_binary(data, data.grouped, getattr(data.result, f"metrics_{kind}"), output, plt))
    elif data.task_name == "regression" and data.grouped is not None:
        figures.update(_plot_regression(data, data.grouped, getattr(data.result, f"metrics_{kind}"), output))
    elif data.task_name == "multiclass" and data.grouped is not None:
        figures.update(_plot_multiclass(data, data.grouped, output))
    if data.grouped is not None and data.group_column is not None:
        figures.update(_plot_group_metrics(data, data.grouped, output, plt))
    for key, confusion in data.confusions.items():
        from sklearn.metrics import ConfusionMatrixDisplay

        size = max(6.0, min(12.0, len(confusion.labels) * 0.8))
        figure, axis = plt.subplots(figsize=(size, size))
        try:
            ConfusionMatrixDisplay(confusion.matrix, display_labels=list(confusion.labels)).plot(
                ax=axis, cmap="Blues", values_format=".2f", colorbar=False
            )
            axis.set_title(f"Normalized multiclass confusion matrix{confusion.suffix.replace('_', ' ')}")
            figure.tight_layout()
            figure.savefig(output / f"{key}.png")
            figures[key] = figure
        finally:
            plt.close(figure)
    for key, curve in data.curves.items():
        figure, axis = plt.subplots(figsize=(7, 5))
        try:
            for learner, (x, y) in curve.learners.items():
                axis.plot(x, y, label=learner.upper())
            axis.set_title(f"{curve.name.capitalize()} curves{curve.layout_suffix}")
            axis.set_xlabel("Targeted observations")
            axis.set_ylabel(curve.name.capitalize())
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f"{key}.png")
            figures[key] = figure
        finally:
            plt.close(figure)
    suffixed = {}
    for name, figure in figures.items():
        suffixed_name = f"{name}_{kind}"
        source = output / f"{name}.png"
        if source.is_file():
            os.replace(source, output / f"{suffixed_name}.png")
        suffixed[suffixed_name] = figure
    return suffixed


def _plot_group_metrics(
    data: EvaluationData, grouped: pl.DataFrame, output_dir: Path, plt: Any
) -> dict[str, Any]:
    """Render the required group-only metric slice independently of date."""
    table = grouped.filter(pl.col("scope") == "group")
    if table.is_empty() or data.group_column not in table.columns:
        return {}
    partition_columns = [name for name in ("model_layout", "learner") if name in table.columns]
    partitions = (
        list(table.partition_by(partition_columns, as_dict=True, maintain_order=True).items())
        if partition_columns
        else [("all", table)]
    )
    excluded = {"scope", data.group_column, "model_layout", "learner", "n_samples"}
    preferred = {
        "binary": {"roc_auc"},
        "response": {"roc_auc"},
        "regression": {"mse", "mae", "mape"},
        "multiclass": {"roc_auc_ovr_macro", "log_loss", "accuracy", "f1_macro", "f1_weighted"},
        "uplift": {"qini_auc", "uplift_auc", "uplift_at_20"},
    }[data.task_name]
    metric_columns = [
        name
        for name, dtype in table.schema.items()
        if name not in excluded and name in preferred and dtype.is_numeric()
    ]
    if not metric_columns:
        return {}
    figure, axes = plt.subplots(len(partitions), 1, figsize=(10, 4.5 * len(partitions)), squeeze=False)
    for axis, (key, values) in zip(axes.ravel(), partitions, strict=True):
        key_values = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(partition_columns, key_values, strict=True)) if partition_columns else {}
        x = values[data.group_column].cast(pl.String).to_list()
        for metric in metric_columns:
            axis.plot(x, values[metric].to_list(), marker="o", label=metric)
        axis.set_title(f"{data.task_name.capitalize()} metrics by group — {labels or 'all'}")
        axis.set_xlabel(data.group_column)
        axis.set_ylabel("Metric value")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    name = f"{data.task_name}_metrics_by_group"
    try:
        figure.savefig(output_dir / f"{name}.png")
        return {name: figure}
    finally:
        plt.close(figure)


def _plot_binary(
    data: EvaluationData, table: pl.DataFrame, baselines: Mapping[str, float], output_dir: Path, plt: Any
) -> dict[str, Any]:
    """Render the selected task metric by date and group."""
    config = data
    if config.date_column is None:
        return {}
    if "roc_auc" not in table.columns:
        return {}
    table = table.filter(pl.col("scope").is_in(["date", "date_group"]))
    group_column = config.group_column
    partition_columns = [name for name in ("model_layout", group_column) if name and name in table.columns]
    partitions = (
        list(table.partition_by(partition_columns, as_dict=True, maintain_order=True).items())
        if partition_columns
        else [("all", table)]
    )
    column_count = min(2, len(partitions))
    row_count = int(np.ceil(len(partitions) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7 * column_count, 4 * row_count),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, (group_key, group_metrics) in zip(flat_axes, partitions, strict=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        labels = dict(zip(partition_columns, key_values, strict=True)) if partition_columns else {}
        metric_name = f"{labels['model_layout']}_roc_auc" if "model_layout" in labels else "roc_auc"
        baseline = baselines[metric_name]
        axis.plot(
            group_metrics[config.date_column].to_list(),
            group_metrics["roc_auc"].to_list(),
            marker="o",
            label="score",
        )
        axis.axhline(baseline, color="gray", linestyle="--", label=f"overall={baseline:.4f}")
        axis.set_title(f"ROC AUC by date — {labels or 'all'}")
        axis.set_xlabel(data.date_label)
        axis.set_ylabel("ROC AUC")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in flat_axes[len(partitions) :]:
        axis.set_visible(False)
    figure.tight_layout()
    name = "roc_auc_by_date"
    try:
        figure.savefig(output_dir / f"{name}.png")
        return {name: figure}
    finally:
        plt.close(figure)


def _plot_regression(
    data: EvaluationData,
    table: pl.DataFrame,
    overall: Mapping[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    """Render MSE/MAE/MAPE trends by date and group."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    config = data
    if config.date_column is None:
        return {}
    table = table.filter(pl.col("scope").is_in(["date", "date_group"]))
    group_column = config.group_column
    partition_columns = [name for name in ("model_layout", group_column) if name and name in table.columns]
    partitions = (
        list(table.partition_by(partition_columns, as_dict=True, maintain_order=True).items())
        if partition_columns
        else [("all", table)]
    )
    figure, axes = plt.subplots(len(partitions), 1, figsize=(9, 4 * len(partitions)), squeeze=False)
    for axis, (key, values) in zip(axes.ravel(), partitions, strict=True):
        key_values = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(partition_columns, key_values, strict=True)) if partition_columns else {}
        metric_names = [name for name in ("mse", "mae", "mape") if name in values.columns]
        for metric in metric_names:
            axis.plot(
                values[config.date_column].to_list(),
                values[metric].to_list(),
                marker="o",
                label=metric.upper(),
            )
        axis.set_title(f"Regression metrics by date — {labels or 'all'}")
        axis.set_xlabel(data.date_label)
        axis.set_ylabel("Metric value")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
        scope_prefix = f"{labels['model_layout']}_" if "model_layout" in labels else ""
        axis.legend(
            title=", ".join(
                f"{name.upper()}={overall[f'{scope_prefix}{name}']:.4g}" for name in metric_names
            )
        )
    figure.tight_layout()
    name = "regression_metrics_by_date"
    try:
        figure.savefig(output_dir / f"{name}.png")
        return {name: figure}
    finally:
        plt.close(figure)


def _plot_multiclass(
    data: EvaluationData,
    grouped: pl.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    config = data
    if config.date_column is None:
        return {}
    grouped = grouped.filter(pl.col("scope").is_in(["date", "date_group"]))
    metric_names = [
        name
        for name in ("roc_auc_ovr_macro", "log_loss", "accuracy", "f1_macro", "f1_weighted")
        if name in grouped.columns
    ]
    if not metric_names:
        return {}
    partition_columns = [name for name in ("model_layout", config.group_column) if name and name in grouped.columns]
    partitions = (
        list(grouped.partition_by(partition_columns, as_dict=True, maintain_order=True).items())
        if partition_columns
        else [("all", grouped)]
    )
    figure, axes = plt.subplots(len(partitions), 1, figsize=(10, 4.5 * len(partitions)), squeeze=False)
    for axis, (key, table) in zip(axes.ravel(), partitions, strict=True):
        key_values = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(partition_columns, key_values, strict=True)) if partition_columns else {}
        for metric in metric_names:
            axis.plot(table[config.date_column].to_list(), table[metric].to_list(), marker="o", label=metric)
        axis.set_title(f"Multiclass metrics by date — {labels or 'all'}")
        axis.set_xlabel(data.date_label)
        axis.set_ylabel("Metric value")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    name = "multiclass_metrics_by_date"
    try:
        figure.savefig(output_dir / f"{name}.png")
        return {name: figure}
    finally:
        plt.close(figure)
