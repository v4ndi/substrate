"""Deterministic structural checks for dynamic default Optuna dimensions."""

import numpy as np
import polars as pl
import pytest

from fmlib.automl.backends.boosting import resolve_default_search_space, suggest_params
from fmlib.automl.data import FeatureSchema


class RecordingTrial:
    def __init__(self):
        self.calls = {}

    def suggest_categorical(self, name, choices):
        self.calls[name] = ("categorical", tuple(choices))
        return choices[0]

    def suggest_int(self, name, low, high, **kwargs):
        self.calls[name] = ("int", low, high, kwargs)
        return low

    def suggest_float(self, name, low, high, **kwargs):
        self.calls[name] = ("float", low, high, kwargs)
        return low


def _data(*, has_nan: bool, has_categorical: bool):
    numerical = pl.Series("number", [1.0, np.nan if has_nan else 2.0])
    columns = [numerical]
    categorical = ("category",) if has_categorical else ()
    if has_categorical:
        columns.append(pl.Series("category", ["a", "b"]))
    frame = pl.DataFrame(columns)
    schema = FeatureSchema(
        categorical=categorical,
        numerical=("number",),
        feature_order=(*categorical, "number"),
        dtypes={name: str(frame.schema[name]) for name in (*categorical, "number")},
        target_column="target",
        client_id_column="epk_id",
        treatment_column=None,
        group_column=None,
    )
    return frame, schema


@pytest.mark.parametrize(
    ("n_trials", "has_nan", "has_categorical", "expected"),
    [
        (3, False, False, {"max_depth"}),
        (3, True, False, {"max_depth", "nan_mode"}),
        (20, True, False, {"max_depth", "nan_mode"}),
        (21, True, False, {"max_depth", "nan_mode", "l2_leaf_reg"}),
        (50, True, True, {"max_depth", "nan_mode", "l2_leaf_reg"}),
        (51, True, False, {"max_depth", "nan_mode", "l2_leaf_reg", "min_data_in_leaf"}),
        (
            51,
            True,
            True,
            {
                "max_depth",
                "nan_mode",
                "l2_leaf_reg",
                "min_data_in_leaf",
                "one_hot_max_size",
            },
        ),
    ],
)
def test_catboost_dynamic_default_space(n_trials, has_nan, has_categorical, expected):
    frame, schema = _data(has_nan=has_nan, has_categorical=has_categorical)
    space = resolve_default_search_space(
        "catboost", n_trials=n_trials, train_frame=frame, schema=schema
    )
    trial = RecordingTrial()
    suggest_params(trial, engine="catboost", model_params={}, search_space=space)

    assert set(trial.calls) == expected
    assert trial.calls["max_depth"] == ("int", 3, 7, {"step": 1, "log": False})
    if "nan_mode" in expected:
        assert trial.calls["nan_mode"] == ("categorical", ("Max", "Min"))
    if "l2_leaf_reg" in expected:
        assert trial.calls["l2_leaf_reg"] == (
            "float",
            1e-8,
            10.0,
            {"step": None, "log": True},
        )
    if "min_data_in_leaf" in expected:
        assert trial.calls["min_data_in_leaf"] == (
            "int",
            1,
            20,
            {"step": 1, "log": False},
        )
    if "one_hot_max_size" in expected:
        assert trial.calls["one_hot_max_size"] == (
            "int",
            3,
            10,
            {"step": 1, "log": False},
        )


@pytest.mark.parametrize(
    ("n_trials", "conditional"),
    [(3, False), (30, False), (31, True)],
)
def test_xgboost_dynamic_default_space(n_trials, conditional):
    frame, schema = _data(has_nan=False, has_categorical=False)
    space = resolve_default_search_space(
        "xgboost", n_trials=n_trials, train_frame=frame, schema=schema
    )
    trial = RecordingTrial()
    suggest_params(trial, engine="xgboost", model_params={}, search_space=space)

    base = {"colsample_bytree", "subsample", "max_depth", "learning_rate"}
    extra = {"min_child_weight", "reg_alpha", "reg_lambda"}
    assert set(trial.calls) == base | (extra if conditional else set())
    assert trial.calls["colsample_bytree"] == (
        "categorical",
        (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    assert trial.calls["subsample"] == ("categorical", (0.4, 0.5, 0.6, 0.7, 0.8, 1.0))
    assert trial.calls["max_depth"] == ("categorical", (5, 7, 9, 11, 13, 15, 17))
    assert trial.calls["learning_rate"] == (
        "categorical",
        (0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02),
    )
    if conditional:
        assert trial.calls["min_child_weight"] == (
            "int",
            1,
            300,
            {"step": 1, "log": False},
        )
        for name in ("reg_alpha", "reg_lambda"):
            assert trial.calls[name] == (
                "float",
                1e-3,
                10.0,
                {"step": None, "log": True},
            )


def test_resolver_uses_only_actual_model_feature_columns():
    frame, schema = _data(has_nan=False, has_categorical=False)
    frame = frame.with_columns(pl.Series("unused", [np.nan, np.nan]))

    space = resolve_default_search_space(
        "catboost", n_trials=3, train_frame=frame, schema=schema
    )

    assert set(space) == {"max_depth"}


def test_catboost_treats_numerical_null_as_missing_but_not_categorical_null():
    frame, schema = _data(has_nan=False, has_categorical=True)
    categorical_null = frame.with_columns(
        pl.lit(None, dtype=pl.String).alias("category")
    )
    numerical_null = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("number"))

    categorical_space = resolve_default_search_space(
        "catboost", n_trials=3, train_frame=categorical_null, schema=schema
    )
    numerical_space = resolve_default_search_space(
        "catboost", n_trials=3, train_frame=numerical_null, schema=schema
    )

    assert set(categorical_space) == {"max_depth"}
    assert set(numerical_space) == {"max_depth", "nan_mode"}
