"""Persist evaluation tables in result metadata and scores under predictions."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from avatar.automl.types import EvaluationResult

_TABLE_FIELDS = (
    "metrics_by_group_raw",
    "metrics_by_group_calibrated",
    "metrics_by_class_raw",
    "metrics_by_class_calibrated",
)
_DTYPES = {
    str(dtype): dtype
    for dtype in (
        pl.Null,
        pl.Boolean,
        pl.String,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Date,
    )
}


def evaluation_payload(result: EvaluationResult) -> dict[str, Any]:
    """Encode small metric tables without extra files, preserving scalar dtypes."""
    tables = {}
    for name in _TABLE_FIELDS:
        frame = getattr(result, name)
        tables[name] = (
            None
            if frame is None
            else {
                column: {
                    "dtype": str(_DTYPES[str(dtype)]),
                    "values": frame[column].cast(pl.String).to_list()
                    if dtype == pl.Date
                    else frame[column].to_list(),
                }
                for column, dtype in frame.schema.items()
            }
        )
    return {
        "metrics_raw": None if result.metrics_raw is None else dict(result.metrics_raw),
        "metrics_calibrated": None
        if result.metrics_calibrated is None
        else dict(result.metrics_calibrated),
        "metric_tables": tables,
        "excel_paths_raw": {
            name: str(path) for name, path in result.excel_paths_raw.items()
        },
        "excel_paths_calibrated": {
            name: str(path) for name, path in result.excel_paths_calibrated.items()
        },
    }


def evaluation_from_payload(value: Mapping[str, Any]) -> EvaluationResult:
    """Restore public tables from JSON, reading only client scores from parquet."""
    tables = {}
    for name in _TABLE_FIELDS:
        table = value.get("metric_tables", {}).get(name)
        if table is None:
            tables[name] = None
        else:
            tables[name] = pl.DataFrame([
                pl.Series(column, data["values"], dtype=pl.String).cast(pl.Date)
                if data["dtype"] == "Date"
                else pl.Series(column, data["values"], dtype=_DTYPES[data["dtype"]])
                for column, data in table.items()
            ])
    return EvaluationResult(
        metrics_raw=value.get("metrics_raw"),
        metrics_calibrated=value.get("metrics_calibrated"),
        figures_raw={
            name: Path(path) for name, path in value.get("figure_paths_raw", {}).items()
        },
        figures_calibrated={
            name: Path(path)
            for name, path in value.get("figure_paths_calibrated", {}).items()
        },
        excel_paths_raw={
            name: Path(path) for name, path in value.get("excel_paths_raw", {}).items()
        },
        excel_paths_calibrated={
            name: Path(path)
            for name, path in value.get("excel_paths_calibrated", {}).items()
        },
        **tables,
    )
