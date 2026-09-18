from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from fmlib.automl.config import BinaryTaskConfig
from fmlib.automl.data import FeatureSchema, ParquetSource, prepare_data
from fmlib.automl.exceptions import SchemaError


def _config(**overrides):
    values = {
        "env_type": "local",
        "backend": "boosting",
        "engine": "catboost",
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": None,
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ["feature"],
        "hidden_state_columns": (),
        "model_layout": None,
        "hyperopt": False,
        "output_dir": "outputs",
        "environment": {},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


def _frame(**overrides):
    values = {
        "epk_id": [1, 2],
        "target": [0, 1],
        "feature": [0.1, 0.2],
        "report_month": ["2026-01-01", "2026-02-01"],
    }
    values.update(overrides)
    return pl.DataFrame(values)


@pytest.mark.parametrize(
    "values",
    [
        ["2026-01-01", "2026-02-01"],
        ["20260101", "20260201"],
        ["2026-01", "2026-02"],
        ["202601", "202602"],
        [date(2026, 1, 1), date(2026, 2, 1)],
        [datetime(2026, 1, 1, 12), datetime(2026, 2, 1, 12)],
    ],
)
def test_report_month_is_normalized_from_supported_types(values):
    prepared, _ = prepare_data(
        _frame(report_month=values), _config(), require_target=True
    )

    assert prepared.frame.schema["report_month"] == pl.Date
    assert prepared.frame["report_month"].to_list() == [
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]


def test_custom_report_month_alias_from_hive_partition_is_normalized_to_date(tmp_path):
    split_root = tmp_path / "global_name=pr" / "split_type=train"
    month = split_root / "month_part=2025-10-31"
    month.mkdir(parents=True)
    pl.DataFrame({
        "epk_id": [1, 2],
        "target": [0, 1],
        "feature": [0.1, 0.2],
    }).write_parquet(month / "part-00000.parquet")
    frame = ParquetSource.resolve(split_root).read()
    config = _config(date_column="month_part")

    prepared, _ = prepare_data(frame, config, require_target=True)

    assert frame.schema["month_part"] == pl.String
    assert prepared.frame.schema["month_part"] == pl.Date


def test_invalid_report_month_has_diagnostic_error():
    with pytest.raises(SchemaError, match="not-a-date"):
        prepare_data(
            _frame(report_month=["not-a-date", "2026-02-01"]),
            _config(),
            require_target=True,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Array(pl.Float32, 2),
        pl.Array(pl.Float64, 2),
        pl.List(pl.Float32),
        pl.List(pl.Float64),
    ],
)
def test_hidden_states_are_normalized_and_expanded_in_stable_order(dtype):
    frame = _frame(
        hidden=pl.Series(
            "hidden",
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=dtype,
        )
    )
    prepared, dimensions = prepare_data(
        frame,
        _config(hidden_state_columns=["hidden"]),
        require_target=True,
    )

    assert dimensions == {"hidden": 2}
    assert prepared.schema.feature_order == ("feature", "hidden__0", "hidden__1")
    assert "hidden" not in prepared.frame.columns
    assert prepared.frame.schema["hidden__0"] == pl.Float32
    assert prepared.frame.schema["hidden__1"] == pl.Float32
    assert np.allclose(
        prepared.frame.select("hidden__0", "hidden__1").to_numpy(),
        [[0.1, 0.2], [0.3, 0.4]],
    )


@pytest.mark.parametrize(
    ("hidden", "expected_dimensions", "message"),
    [
        (
            pl.Series(
                "hidden", [[0.1, 0.2], [0.3, 0.4]], dtype=pl.Array(pl.Float32, 2)
            ),
            {"hidden": 3},
            "dimension mismatch",
        ),
        (
            pl.Series("hidden", [[1, 2], [3, 4]], dtype=pl.List(pl.Int64)),
            None,
            "Float32/Float64",
        ),
        (
            pl.Series("hidden", [[0.1, 0.2], [0.3]], dtype=pl.List(pl.Float64)),
            None,
            "fixed dimension",
        ),
        (
            pl.Series("hidden", [[0.1, None], [0.3, 0.4]], dtype=pl.List(pl.Float64)),
            None,
            "null elements",
        ),
        (
            pl.Series("hidden", [None, [0.3, 0.4]], dtype=pl.Array(pl.Float32, 2)),
            None,
            "null embeddings",
        ),
    ],
)
def test_invalid_hidden_states_are_rejected(hidden, expected_dimensions, message):
    with pytest.raises(SchemaError, match=message):
        prepare_data(
            _frame(hidden=hidden),
            _config(hidden_state_columns=["hidden"]),
            require_target=True,
            hidden_dimensions=expected_dimensions,
        )


def test_fitted_schema_ignores_extra_columns_and_restores_feature_order():
    config = _config(categorical_columns=["segment"])
    train, dimensions = prepare_data(
        _frame(segment=["a", "b"]),
        config,
        require_target=True,
    )
    inference, _ = prepare_data(
        _frame(segment=["b", "a"], extra=[10, 20]).select(
            "extra", "feature", "segment", "epk_id", "report_month"
        ),
        config,
        require_target=False,
        fitted_schema=train.schema,
        hidden_dimensions=dimensions,
    )

    assert inference.schema.feature_order == train.schema.feature_order
    assert inference.frame["extra"].to_list() == [10, 20]


def test_fitted_schema_rejects_dtype_drift():
    config = _config()
    train, dimensions = prepare_data(_frame(), config, require_target=True)

    with pytest.raises(SchemaError, match="expected Float64, got Float32"):
        prepare_data(
            _frame(feature=pl.Series("feature", [0.1, 0.2], dtype=pl.Float32)),
            config,
            require_target=True,
            fitted_schema=train.schema,
            hidden_dimensions=dimensions,
        )


def test_missing_report_month_is_rejected_early():
    with pytest.raises(SchemaError, match="date_column='report_month'"):
        prepare_data(_frame().drop("report_month"), _config(), require_target=True)


def test_null_tabular_values_are_allowed_but_infinite_numbers_are_rejected():
    prepared, _ = prepare_data(
        _frame(feature=[None, 0.2]), _config(), require_target=True
    )

    assert prepared.frame["feature"].null_count() == 1
    with pytest.raises(SchemaError, match="Infinite values"):
        prepare_data(
            _frame(feature=[float("inf"), 0.2]), _config(), require_target=True
        )


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Array(pl.Float32, 2),
        pl.Array(pl.Float64, 2),
        pl.List(pl.Float32),
        pl.List(pl.Float64),
    ],
)
def test_hidden_states_stay_vectors_when_expansion_is_turned_off(dtype):
    """The TabNN shape: one column, one width, not a feature."""
    frame = _frame(hidden=pl.Series("hidden", [[0.1, 0.2], [0.3, 0.4]], dtype=dtype))
    prepared, dimensions = prepare_data(
        frame,
        _config(hidden_state_columns=["hidden"]),
        require_target=True,
        expand_hidden_states=False,
    )

    assert dimensions == {"hidden": 2}
    assert prepared.schema.hidden_states == {"hidden": 2}
    assert prepared.schema.feature_order == ("feature",)
    assert "hidden__0" not in prepared.frame.columns
    assert prepared.frame.schema["hidden"] == pl.List(pl.Float32)
    assert np.allclose(
        np.stack(prepared.frame["hidden"].to_list()), [[0.1, 0.2], [0.3, 0.4]]
    )


def test_expanded_hidden_states_leave_the_width_field_empty():
    """Boosting keeps its widths in `dimensions`; nothing about it changes."""
    frame = _frame(
        hidden=pl.Series("hidden", [[0.1, 0.2], [0.3, 0.4]], dtype=pl.List(pl.Float32))
    )
    prepared, _ = prepare_data(
        frame, _config(hidden_state_columns=["hidden"]), require_target=True
    )
    assert prepared.schema.hidden_states == {}


@pytest.mark.parametrize(
    ("hidden", "expected_dimensions", "message"),
    [
        (
            pl.Series("hidden", [[1, 2], [3, 4]], dtype=pl.List(pl.Int64)),
            None,
            "Float32/Float64",
        ),
        (
            pl.Series("hidden", [[0.1, 0.2], [0.3]], dtype=pl.List(pl.Float64)),
            None,
            "fixed dimension",
        ),
        (
            pl.Series(
                "hidden", [[0.1, 0.2], [0.3, 0.4]], dtype=pl.Array(pl.Float32, 2)
            ),
            {"hidden": 3},
            "dimension mismatch",
        ),
    ],
)
def test_unexpanded_hidden_states_are_validated_just_as_strictly(
    hidden, expected_dimensions, message
):
    with pytest.raises(SchemaError, match=message):
        prepare_data(
            _frame(hidden=hidden),
            _config(hidden_state_columns=["hidden"]),
            require_target=True,
            hidden_dimensions=expected_dimensions,
            expand_hidden_states=False,
        )


def test_schema_widths_survive_a_manifest_round_trip():
    frame = _frame(
        hidden=pl.Series("hidden", [[0.1, 0.2], [0.3, 0.4]], dtype=pl.List(pl.Float32))
    )
    prepared, _ = prepare_data(
        frame,
        _config(hidden_state_columns=["hidden"]),
        require_target=True,
        expand_hidden_states=False,
    )
    restored = FeatureSchema.from_dict(prepared.schema.to_dict())
    assert restored == prepared.schema


def test_a_schema_written_before_widths_existed_still_loads():
    payload = {
        "categorical": ["segment"],
        "numerical": ["feature"],
        "feature_order": ["segment", "feature"],
        "dtypes": {"segment": "String", "feature": "Float64"},
        "target_column": "target",
        "client_id_column": "epk_id",
        "treatment_column": None,
        "group_column": None,
    }
    assert FeatureSchema.from_dict(payload).hidden_states == {}
