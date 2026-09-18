import math

import numpy as np
import pytest
from pyspark.sql import functions as F

from fmlib.preprocessing.spark import LabelEncoder, StandardScaler
from fmlib.preprocessing.spark.pipeline.base_pipe import NumCatPipeline


@pytest.fixture
def sample_data(spark_session):
    return spark_session.createDataFrame(
        [("A", 10.0, "X"), ("B", 20.0, "Y"), ("C", 30.0, "Z")], ["cat1", "num1", "cat2"]
    )


def test_initialization_valid_columns():
    # Test valid initialization
    processor = NumCatPipeline(
        categorical_columns=["cat1", "cat2"], numeric_columns=["num1"]
    )
    assert processor.cat_cols == ["cat1", "cat2"]
    assert processor.num_cols == ["num1"]


def test_fit_with_new_components(sample_data):
    # Test fitting with new encoder/scaler
    processor = NumCatPipeline(
        categorical_columns=["cat1", "cat2"], numeric_columns=["num1"]
    )
    processor.fit(sample_data)

    assert isinstance(processor.label_encoder, LabelEncoder)
    assert sorted(processor.label_encoder.columns) == ["cat1", "cat2"]

    assert isinstance(processor.standard_scaler, StandardScaler)
    assert processor.standard_scaler.columns == ["num1"]


def test_fit_with_existing_components(sample_data):
    # Test fitting with pre-configured components
    le = LabelEncoder(["cat1", "cat2"], spec_tokens={"unk": 0})
    ss = StandardScaler(["num1"])

    processor = NumCatPipeline(
        categorical_columns=["cat1", "cat2"],
        numeric_columns=["num1"],
        label_encoder=le,
        standard_scaler=ss,
    )
    processor.fit(sample_data)

    assert processor.label_encoder is le
    assert processor.standard_scaler is ss


def test_transform(sample_data):
    # Test transformation pipeline
    processor = NumCatPipeline(
        categorical_columns=["cat1", "cat2"], numeric_columns=["num1"]
    )
    transformed = processor.fit_transform(sample_data)
    # Check scaling results
    scaled_values = [row[0] for row in transformed.select("num1").collect()]
    assert abs(processor.standard_scaler.mean_std["num1"]["mean"] - 20.0) < 1e-6
    assert abs(processor.standard_scaler.mean_std["num1"]["std"] - 10.0) < 1e-6
    np.testing.assert_allclose(scaled_values, [-1.0, 0.0, 1.0], rtol=1e-3)


def test_dump_load(sample_data):
    # Test serialization/deserialization
    original = NumCatPipeline(
        categorical_columns=["cat1", "cat2"], numeric_columns=["num1"]
    )
    original.fit(sample_data)

    dumped = original.dump()
    reloaded = NumCatPipeline.load(dumped)

    assert reloaded.cat_cols == original.cat_cols
    assert reloaded.num_cols == original.num_cols
    assert reloaded.label_encoder.values_to_id == original.label_encoder.values_to_id
    assert reloaded.standard_scaler.mean_std == original.standard_scaler.mean_std

    df1 = original.transform(sample_data).toPandas()
    df2 = reloaded.transform(sample_data).toPandas()

    assert all(df1 == df2)


def test_error_handling(sample_data):
    # Test component mismatch errors
    with pytest.raises(AssertionError):
        wrong_le = LabelEncoder(["wrong_cat"], spec_tokens={"unk": 0})
        NumCatPipeline(
            categorical_columns=["cat1", "cat2"],
            numeric_columns=["num1"],
            label_encoder=wrong_le,
        ).fit(sample_data)

    with pytest.raises(AssertionError):
        wrong_ss = StandardScaler(["wrong_num"])
        NumCatPipeline(
            categorical_columns=["cat1", "cat2"],
            numeric_columns=["num1"],
            standard_scaler=wrong_ss,
        ).fit(sample_data)


def test_null_handling(spark_session):
    # Test handling of unknown/null values
    data = spark_session.createDataFrame(
        [("A", 10.0, "X"), (None, 20.0, None), ("C", 30.0, "Z")],
        ["cat1", "num1", "cat2"],
    )

    processor = NumCatPipeline(
        categorical_columns=["cat1", "cat2"], numeric_columns=["num1"]
    )
    transformed = processor.fit_transform(data)

    # Check UNK handling
    unk_counts = processor.label_encoder.count_unk(transformed)
    assert unk_counts["total"] == 2  # 2 null values in categorical columns


def collect2list(df, cols):
    return [tuple(getattr(r, c) for c in cols) for r in df.select(*cols).collect()]


def signed_log1p_py(x):
    if x is None:
        return None
    if x == 0:
        return 0.0
    return math.log1p(abs(x)) * (1.0 if x > 0 else -1.0)


def test_construct_with_kwargs_builds_components(spark_session):
    pipe = NumCatPipeline(
        categorical_columns=["c1", "c2"],
        numeric_columns=["n1", "n2"],
        label_encoder_kwargs={"frequency_encoder": False},
        standard_scaler_kwargs={"to_log_columns": ["n2"], "fillna": True},
    )
    assert isinstance(pipe.label_encoder, LabelEncoder)
    assert isinstance(pipe.standard_scaler, StandardScaler)
    assert pipe.label_encoder.columns == ["c1", "c2"]
    assert pipe.standard_scaler.columns == ["n1", "n2"]
    assert "n2" in (pipe.standard_scaler.to_log_columns or [])


def test_fit_and_transform_with_kwargs(spark_session):
    df = spark_session.createDataFrame(
        [
            ("a", "x", 0.0, 10.0),
            ("b", "y", 1.0, 20.0),
            ("a", "y", 3.0, 40.0),
        ],
        ["c1", "c2", "n1", "n2"],
    )
    pipe = NumCatPipeline(
        categorical_columns=["c1", "c2"],
        numeric_columns=["n1", "n2"],
        label_encoder_kwargs={"frequency_encoder": False},
        standard_scaler_kwargs={"to_log_columns": ["n2"], "fillna": False},
    )
    pipe.fit(df)
    out = pipe.transform(df)

    # Validate categorical encoded to ints with unk token present in mapping
    # Mapping is data-dependent; check types and presence of keys
    assert all(
        isinstance(v, int)
        for _, mapping in pipe.label_encoder.values_to_id.items()
        for v in mapping.values()
    )
    assert all("unk" in mapping for mapping in pipe.label_encoder.values_to_id.values())

    # Validate numeric scaling: n1 scaled linearly, n2 logged then scaled
    rows = collect2list(out, ["n1", "n2"])
    # recompute expected using stored stats
    mean1 = pipe.standard_scaler.mean_std["n1"]["mean"]
    std1 = pipe.standard_scaler.mean_std["n1"]["std"] + 1e-8
    mean2 = pipe.standard_scaler.mean_std["n2"]["mean"]
    std2 = pipe.standard_scaler.mean_std["n2"]["std"] + 1e-8
    xs1 = [0.0, 1.0, 3.0]
    xs2_logged = [signed_log1p_py(v) for v in [10.0, 20.0, 40.0]]
    exp = [
        ((x1 - mean1) / std1, (z - mean2) / std2)
        for x1, z in zip(xs1, xs2_logged, strict=False)
    ]
    for (a1, a2), (e1, e2) in zip(rows, exp, strict=False):
        assert abs(a1 - e1) < 1e-6
        assert abs(a2 - e2) < 1e-6


def test_unknown_categories_mapped_to_unk_zero(spark_session):
    train = spark_session.createDataFrame(
        [("a", "x"), ("b", "y")],
        ["c1", "c2"],
    )
    test = spark_session.createDataFrame(
        [("a", "x"), ("c", "z"), (None, "x")],
        ["c1", "c2"],
    )
    pipe = NumCatPipeline(
        categorical_columns=["c1", "c2"],
        numeric_columns=None,
        label_encoder_kwargs={"frequency_encoder": True},
    )
    pipe.fit(train)
    out = pipe.transform(test)

    # All unknowns should map to 0
    mapping_c1 = pipe.label_encoder.values_to_id["c1"]
    mapping_c2 = pipe.label_encoder.values_to_id["c2"]
    unk1 = mapping_c1["unk"]
    unk2 = mapping_c2["unk"]
    vals = collect2list(out, ["c1", "c2"])

    assert vals[1][0] == unk1
    assert vals[2][0] == unk1
    assert vals[1][1] == unk2


def test_use_preconfigured_components(spark_session):
    # Prebuild encoder/scaler and pass them through kwargs off
    enc = LabelEncoder(columns=["c"])
    sca = StandardScaler(columns=["n"], to_log_columns=["n"])
    df = spark_session.createDataFrame([("a", 1.0), ("b", 2.0)], ["c", "n"])

    pipe = NumCatPipeline(
        categorical_columns=["c"],
        numeric_columns=["n"],
        label_encoder=enc,
        standard_scaler=sca,
    )
    # Asserts inside fit check column alignment
    pipe.fit(df)
    out = pipe.transform(df)
    assert "c" in out.columns and "n" in out.columns


def test_fit_transform_roundtrip_and_dump_load(spark_session, tmp_path):
    df = spark_session.createDataFrame(
        [
            ("a", 0.0),
            ("b", 1.0),
            ("a", 3.0),
        ],
        ["c", "n"],
    )
    pipe = NumCatPipeline(
        categorical_columns=["c"],
        numeric_columns=["n"],
        label_encoder_kwargs={"frequency_encoder": True},
        standard_scaler_kwargs={"to_log_columns": ["n"]},
    )
    out1 = pipe.fit_transform(df)
    # dump and load back
    dumped = pipe.dump()
    loaded = NumCatPipeline.load(dumped)
    out2 = loaded.transform(df)

    # Compare outputs numerically
    r1 = collect2list(out1, ["c", "n"])
    r2 = collect2list(out2, ["c", "n"])
    assert pipe.label_encoder.values_to_id["c"]["a"] == 1
    assert len(r1) == len(r2)
    for (c1, n1), (c2, n2) in zip(r1, r2, strict=False):
        assert c1 == c2
        assert abs(n1 - n2) < 1e-8


def test_transform_order_cat_then_num(spark_session):
    df = spark_session.createDataFrame([("a", 1.0), ("a", 2.0)], ["c", "n"])
    pipe = NumCatPipeline(
        categorical_columns=["c"],
        numeric_columns=["n"],
        label_encoder_kwargs={"frequency_encoder": True},
        standard_scaler_kwargs={"to_log_columns": []},
    )
    pipe.fit(df)
    out = pipe.transform(df)
    # Ensure label encoding applied before scaling doesn't break numeric column
    assert set(out.columns) == {"c", "n"}
    assert isinstance(out.select("c").collect()[0][0], int)
    assert isinstance(out.select("n").collect()[0][0], float)


def test_standard_scaler_kwargs_fillna_after_log(spark_session):
    df = spark_session.createDataFrame(
        [(None, 0.0), ("a", None)],
        ["c", "n"],
    )
    pipe = NumCatPipeline(
        categorical_columns=["c"],
        numeric_columns=["n"],
        label_encoder_kwargs={"frequency_encoder": True},
        standard_scaler_kwargs={"to_log_columns": ["n"], "fillna": True},
    )
    pipe.fit(df.fillna({"n": 0.0}))
    out = pipe.transform(df)
    # n should be filled post-transform
    assert out.filter(F.col("n").isNull()).count() == 0
