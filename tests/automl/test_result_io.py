"""Metric metadata preserves public scalar types without extra table files."""

import json
from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from fmlib.automl.result_io import evaluation_from_payload, evaluation_payload
from fmlib.automl.types import EvaluationResult


@pytest.mark.parametrize("empty", [False, True])
def test_metric_json_round_trip_preserves_types_nulls_and_large_labels(tmp_path, empty):
    table = pl.DataFrame({
        "month": pl.Series([date(2026, 3, 1), None], dtype=pl.Date),
        "class_label": pl.Series([2**64 - 1, 2**64 - 2], dtype=pl.UInt64),
        "metric": pl.Series([float("nan"), None], dtype=pl.Float32),
        "count": pl.Series([1, 2], dtype=pl.UInt32),
        "all_null": pl.Series([None, None], dtype=pl.Null),
        "group": ["a", "b"],
    })
    if empty:
        table = table.clear()
    result = EvaluationResult(
        metrics_raw={},
        metrics_calibrated={"roc_auc": 0.9},
        metrics_by_group_raw=table,
        metrics_by_group_calibrated=table,
        metrics_by_class_raw=table,
        metrics_by_class_calibrated=table,
    )
    payload = json.loads(json.dumps(evaluation_payload(result)))
    restored = evaluation_from_payload(payload)
    assert restored.metrics_raw == {}
    assert restored.metrics_calibrated == {"roc_auc": 0.9}
    assert not hasattr(restored, "metrics")
    assert not hasattr(restored, "metrics_by_group")
    assert not hasattr(restored, "metrics_by_class")
    assert not hasattr(restored, "figures")
    assert not hasattr(restored, "excel_paths")
    assert_frame_equal(restored.metrics_by_group_raw, table)
    assert_frame_equal(restored.metrics_by_group_calibrated, table)
    assert_frame_equal(restored.metrics_by_class_raw, table)
    assert_frame_equal(restored.metrics_by_class_calibrated, table)
    assert not list(tmp_path.rglob("*.parquet"))
