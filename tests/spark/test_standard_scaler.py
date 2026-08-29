import math

import pytest
from pyspark.sql.types import DoubleType, StructField, StructType
from pyspark.testing.utils import assertDataFrameEqual

from avatar.preprocessing.spark import StandardScaler


@pytest.fixture
def sample_df(spark_session):
    schema = StructType([
        StructField("col1", DoubleType(), True),
        StructField("col2", DoubleType(), True),
        StructField("col3", DoubleType(), True),
    ])
    data = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (10.0, 11.0, 12.0)]
    return spark_session.createDataFrame(data, schema)


def test_standard_scaler_initialization():
    columns = ["col1", "col2"]
    scaler = StandardScaler(columns)
    assert scaler.columns == columns
    assert all(col in scaler.mean_std for col in columns)
    assert all(
        key in scaler.mean_std[col] for col in columns for key in ["mean", "std"]
    )


def test_standard_scaler_fit(sample_df):
    columns = ["col1", "col2"]
    scaler = StandardScaler(columns)
    scaler.fit(sample_df)

    expected_mean = {"col1": 5.5, "col2": 6.5}
    expected_std = {
        "col1": pytest.approx(3.872983, rel=1e-6),
        "col2": pytest.approx(3.872983, rel=1e-6),
    }

    for col in columns:
        assert scaler.mean_std[col]["mean"] == pytest.approx(
            expected_mean[col], rel=1e-6
        )
        assert scaler.mean_std[col]["std"] == expected_std[col]


def test_standard_scaler_transform(sample_df):
    columns = ["col1", "col2"]
    scaler = StandardScaler(columns)
    scaler.fit(sample_df)

    transformed_df = scaler.transform(sample_df)

    expected_data = [
        (-1.161895, -1.161895, 3.0),
        (-0.387298, -0.387298, 6.0),
        (0.387298, 0.387298, 9.0),
        (1.161895, 1.161895, 12.0),
    ]

    actual_data = transformed_df.collect()

    for expected, actual in zip(expected_data, actual_data, strict=False):
        assert actual[0] == pytest.approx(expected[0], rel=1e-6)
        assert actual[1] == pytest.approx(expected[1], rel=1e-6)
        assert actual[2] == expected[2]


def test_standard_scaler_fit_transform(sample_df):
    columns = ["col1", "col2"]
    scaler = StandardScaler(columns)

    transformed_df = scaler.fit_transform(sample_df)

    expected_data = [
        (-1.161895, -1.161895, 3.0),
        (-0.387298, -0.387298, 6.0),
        (0.387298, 0.387298, 9.0),
        (1.161895, 1.161895, 12.0),
    ]

    actual_data = transformed_df.collect()

    for expected, actual in zip(expected_data, actual_data, strict=False):
        assert actual[0] == pytest.approx(expected[0], rel=1e-6)
        assert actual[1] == pytest.approx(expected[1], rel=1e-6)
        assert actual[2] == expected[2]


def test_standard_scaler_non_existent_column(sample_df):
    from pyspark.errors import AnalysisException

    columns = ["col1", "non_existent_col"]
    scaler = StandardScaler(columns)

    with pytest.raises(AnalysisException):
        scaler.fit(sample_df)


def test_dump_and_load(sample_df):
    columns = ["col1", "col2"]

    scaler = StandardScaler(columns)
    df1 = scaler.fit_transform(sample_df)

    loaded_scaler = StandardScaler.load(scaler.dump())
    df2 = loaded_scaler.transform(sample_df)

    assert df1.collect() == df2.collect(), (
        "Transformed data should be identical after dump and load"
    )
    assert loaded_scaler.__dict__ == scaler.__dict__, "Columns should match"


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_fit_uses_logged_values_for_stats(spark_session):
    # x will be logged, y not logged; only x in scaler columns for this test
    df = spark_session.createDataFrame(
        [(0.0, 10.0), (1.0, 20.0), (3.0, 30.0)], ["x", "y"]
    )
    scaler = StandardScaler(columns=["x"], to_log_columns=["x"], fillna=False)
    scaler.fit(df)

    # stats computed on log1p(|x|)*sign(x)
    xs = [0.0, 1.0, 3.0]
    xs_logged = [
        math.log1p(abs(v)) * (0 if v == 0 else (1 if v > 0 else -1)) for v in xs
    ]
    mean = sum(xs_logged) / len(xs_logged)
    var = sum((v - mean) ** 2 for v in xs_logged) / (len(xs_logged) - 1)
    std = math.sqrt(var)
    assert approx(scaler.mean_std["x"]["mean"], mean, 1e-9)
    assert approx(scaler.mean_std["x"]["std"], std, 1e-9)


def test_transform_applies_log_then_scales(spark_session):
    df = spark_session.createDataFrame([(0.0, 2.0), (1.0, 3.0), (3.0, 5.0)], ["x", "z"])
    scaler = StandardScaler(columns=["x"], to_log_columns=["x"], fillna=False)
    scaler.fit(df)

    out = scaler.transform(df).collect()
    # validate that only column x is scaled and z is untouched
    xs_scaled = [r.x for r in out]
    zs = [r.z for r in out]
    assert zs == [2.0, 3.0, 5.0]

    # recompute using stored mean/std over logged x
    def signed_log1p(v):
        return math.log1p(abs(v)) * (0 if v == 0 else (1 if v > 0 else -1))

    xs_logged = [signed_log1p(v) for v in [0.0, 1.0, 3.0]]
    mean = scaler.mean_std["x"]["mean"]
    std = scaler.mean_std["x"]["std"] + 1e-8
    xs_scaled_expected = [(v - mean) / std for v in xs_logged]
    for a, b in zip(xs_scaled, xs_scaled_expected, strict=False):
        assert approx(a, b, 1e-6)


def test_update_adds_log_columns_and_new_numeric_stats(spark_session):
    df1 = spark_session.createDataFrame([(0.0, 2.0), (2.0, 2.0)], ["x", "y"])
    scaler = StandardScaler(columns=["x"], to_log_columns=["x"], fillna=False)
    scaler.fit(df1)

    # update: add y as a new numeric column and also mark it as to_log
    df2 = spark_session.createDataFrame(
        [(0.0, 2.0), (0.0, 4.0), (0.0, 8.0)], ["x", "y"]
    )
    scaler.update(df2, new_columns=["y"], new_to_log_columns=["y"])

    assert "y" in scaler.columns
    assert "y" in scaler.to_log_columns

    # stats for y computed on log1p(|y|)
    ys = [2.0, 4.0, 8.0]
    ys_logged = [math.log1p(abs(v)) for v in ys]
    mean = sum(ys_logged) / len(ys_logged)
    var = sum((v - mean) ** 2 for v in ys_logged) / (len(ys_logged) - 1)
    std = math.sqrt(var)
    assert approx(scaler.mean_std["y"]["mean"], mean, 1e-9)
    assert approx(scaler.mean_std["y"]["std"], std, 1e-9)


def test_update_only_log_config_no_new_numeric(spark_session):
    df = spark_session.createDataFrame([(1.0,), (0.0,)], ["x"])
    scaler = StandardScaler(columns=["x"], to_log_columns=None, fillna=False)
    scaler.fit(df)

    # only update log config (no recompute for existing stats here)
    scaler.update(df, new_columns=[], new_to_log_columns=["x"])
    assert "x" in scaler.columns
    assert "x" in scaler.to_log_columns


def test_fillna_behavior_after_log(spark_session):
    # check that fillna still applies after log-transform + scale
    df = spark_session.createDataFrame([(None,), (0.0,), (1.0,)], ["x"])
    scaler = StandardScaler(columns=["x"], to_log_columns=["x"], fillna=True)
    scaler.fit(df.fillna({"x": 0.0}))  # fit on filled data or upstream-cleaned dataset

    out = scaler.transform(df).select("x").collect()
    xs = [r.x for r in out]
    # None should be filled to 0 after scaling
    assert not any(v is None for v in xs)


def test_dataframe_equivalence_helpers_with_log_scaled(spark_session):
    # Integration-style check using assertDataFrameEqual with tolerance
    df = spark_session.createDataFrame([(0.0,), (1.0,), (3.0,)], ["x"])
    scaler = StandardScaler(columns=["x"], to_log_columns=["x"], fillna=False)
    pred = scaler.fit_transform(df)

    xs = [0.0, 1.0, 3.0]
    xs_logged = [math.log1p(abs(v)) * (0 if v == 0 else 1) for v in xs]
    mean = sum(xs_logged) / len(xs_logged)
    var = sum((v - mean) ** 2 for v in xs_logged) / (len(xs_logged) - 1)
    std = math.sqrt(var) + 1e-8
    exp_vals = [((v - mean) / std,) for v in xs_logged]
    expected = spark_session.createDataFrame(exp_vals, ["x"])

    assertDataFrameEqual(pred, expected, rtol=1e-6, atol=1e-8)
