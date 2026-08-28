import random
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from avatar.preprocessing.spark.pipeline import EventSequencePreprocessor


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder.appName("SequencePreprocessorTests").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_data(spark):
    schema = T.StructType([
        T.StructField("epk_id", T.IntegerType(), nullable=False),
        T.StructField("timestamps", T.TimestampType(), nullable=False),
        T.StructField("mcc_code", T.StringType(), nullable=False),
        T.StructField("price", T.DoubleType(), nullable=False),
        T.StructField("trx_dir", T.StringType(), nullable=False),
        T.StructField("event_ids", T.IntegerType(), nullable=False),
    ])

    data = []
    user_ids = list(range(1, 101))
    mcc_codes = [
        "5411",
        "5812",
        "5912",
        "5310",
        "5412",
        "5814",
        "5499",
        "5300",
        "5311",
        "5921",
    ]
    directions = ["in", "out"]

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    for _ in range(1000):  # Smaller dataset for testing
        days_diff = random.randint(0, (end_date - start_date).days)
        random_date = start_date + timedelta(days=days_diff)

        data.append((
            random.choice(user_ids),
            random_date,
            random.choice(mcc_codes),
            round(random.uniform(1, 1000), 2),
            random.choice(directions),
            random.choice([0, 1]),
        ))

    return spark.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_data_2(spark):
    df = spark.createDataFrame([
        Row(
            epk_id=1,
            mcc=1,
            comm_type="SMS",
            report_month=date(2024, 1, 31),
            timestamps=datetime(2024, 1, 1, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=1,
            comm_type="PUSH",
            report_month=date(2024, 2, 29),
            timestamps=datetime(2024, 1, 1, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=2,
            comm_type="SMS",
            report_month=date(2024, 1, 31),
            timestamps=datetime(2024, 1, 2, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=2,
            comm_type="PUSH",
            report_month=date(2024, 2, 29),
            timestamps=datetime(2024, 1, 2, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=3,
            comm_type="SMS",
            report_month=date(2024, 1, 31),
            timestamps=datetime(2024, 1, 3, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=3,
            comm_type="PUSH",
            report_month=date(2024, 2, 29),
            timestamps=datetime(2024, 1, 3, 12, 0),
            event_ids=0,
        ),
    ])
    return df


@pytest.fixture
def sample_data_3(spark):
    df = spark.createDataFrame([
        Row(
            epk_id=1,
            mcc=None,
            comm_type="SMS",
            report_month=date(2024, 1, 31),
            timestamps=datetime(2024, 1, 1, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=1,
            comm_type="PUSH",
            report_month=date(2024, 2, 29),
            timestamps=datetime(2024, 1, 1, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=2,
            comm_type="SMS",
            report_month=date(2024, 1, 31),
            timestamps=datetime(2024, 1, 2, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=2,
            comm_type="PUSH",
            report_month=date(2024, 2, 29),
            timestamps=datetime(2024, 1, 2, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=3,
            comm_type="SMS",
            report_month=date(2024, 1, 31),
            timestamps=datetime(2024, 1, 3, 12, 0),
            event_ids=0,
        ),
        Row(
            epk_id=1,
            mcc=3,
            comm_type="PUSH",
            report_month=date(2024, 2, 29),
            timestamps=datetime(2024, 1, 3, 12, 0),
            event_ids=0,
        ),
    ])
    return df


def test_initialization():
    """Test that the preprocessor initializes correctly"""
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["cat1", "cat2"],
        numeric_columns=["num1"],
        event_time_column="event_time",
        id_column="user_id",
    )

    assert preprocessor.cat_cols == ["cat1", "cat2"]
    assert preprocessor.num_cols == ["num1"]
    assert preprocessor.event_time_column == "event_time"
    assert preprocessor.id_column == "user_id"
    assert preprocessor.groupby_columns == []


def test_initialization_with_groupby():
    """Test initialization with groupby columns"""
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["cat1"],
        numeric_columns=["num1"],
        event_time_column="event_time",
        id_column="user_id",
        groupby_columns=["group1", "group2"],
    )

    assert preprocessor.groupby_columns == ["group1", "group2"]


def test_columns_cannot_be_both_categorical_and_numeric():
    """Test that columns can't be both categorical and numeric"""
    with pytest.raises(AssertionError):
        EventSequencePreprocessor(
            categorical_columns=["col1"],
            numeric_columns=["col1"],
            event_time_column="event_time",
            id_column="user_id",
        )


def test_transform_columns_meta(sample_data):
    """Test columns_meta property"""
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc_code", "trx_dir"],
        numeric_columns=["price"],
        event_time_column="timestamps",
        id_column="epk_id",
    )

    preprocessor.fit(sample_data)
    meta = preprocessor.columns_meta

    assert isinstance(meta, dict)
    assert "mcc_code" in meta
    assert meta["mcc_code"]["type"] == "categorical"
    n_unique_mcc = sample_data.select("mcc_code").dropDuplicates().count()
    assert meta["mcc_code"]["n_classes"] == n_unique_mcc + 1
    assert meta["price"]["type"] == "numeric"
    assert meta["price"]["n_classes"] == 1


def test_transform_output_structure(sample_data):
    """Test the structure of the transformed output"""
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc_code", "trx_dir"],
        numeric_columns=["price"],
        event_time_column="timestamps",
        id_column="epk_id",
    )

    processed = preprocessor.fit_transform(sample_data)

    # Check output columns
    assert "epk_id" in processed.columns
    assert "mcc_code" in processed.columns
    assert "trx_dir" in processed.columns
    assert "price" in processed.columns

    # Check that the columns are arrays
    for col in ["mcc_code", "trx_dir", "price"]:
        assert "array" in str(processed.schema[col].dataType).lower()


def test_groupby_behavior(sample_data):
    """Test groupby functionality with additional columns"""
    # Add a groupby column to test data
    sample_data = sample_data.withColumn("group_col", F.lit("test_group"))

    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc_code", "trx_dir"],
        numeric_columns=["price"],
        event_time_column="timestamps",
        id_column="epk_id",
        groupby_columns=["group_col"],
    )

    processed = preprocessor.fit_transform(sample_data)

    # Should have both id and groupby columns
    assert "epk_id" in processed.columns
    assert "group_col" in processed.columns


def test_output_dtypes(sample_data):
    """Test output dtypes"""
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc_code", "trx_dir"],
        numeric_columns=["price"],
        event_time_column="timestamps",
        id_column="epk_id",
    )

    processed = preprocessor.fit_transform(sample_data)
    assert isinstance(processed.schema["mcc_code"].dataType.elementType, T.LongType)
    assert isinstance(processed.schema["trx_dir"].dataType.elementType, T.LongType)
    assert isinstance(processed.schema["price"].dataType.elementType, T.FloatType)


def test_sorting_behavior(sample_data):
    """Test that events are properly sorted by time"""
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc_code", "trx_dir"],
        numeric_columns=["price"],
        event_time_column="timestamps",
        id_column="epk_id",
    )

    processed = preprocessor.fit_transform(sample_data)

    # Get one user's transactions
    user_data = processed.filter(F.col("epk_id") == 1).collect()[0]

    # Check that arrays are in chronological order
    timestamps = user_data["timestamps"]
    if len(timestamps) > 1:
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]


def test_dump_load_consistency(sample_data):
    """Test that dump/load preserves the processor state"""
    preprocessor1 = EventSequencePreprocessor(
        categorical_columns=["mcc_code", "trx_dir"],
        numeric_columns=["price"],
        event_time_column="timestamps",
        id_column="epk_id",
    )

    preprocessor1.fit(sample_data)
    state = preprocessor1.dump()

    preprocessor2 = EventSequencePreprocessor.load(state)

    # Check that loaded processor has same attributes
    assert preprocessor1.cat_cols == preprocessor2.cat_cols
    assert preprocessor1.num_cols == preprocessor2.num_cols
    assert preprocessor1.event_time_column == preprocessor2.event_time_column
    assert preprocessor1.id_column == preprocessor2.id_column

    # Check that transformations produce same results
    result1 = preprocessor1.transform(sample_data)
    result2 = preprocessor2.transform(sample_data)

    assert result1.collect() == result2.collect()


def test_groupby(sample_data_2):
    """Test groupby by with duplicates in rows"""
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc"],
        numeric_columns=None,
        event_time_column="timestamps",
        id_column="epk_id",
        groupby_columns=["report_month", "comm_type"],
    )

    processed = preprocessor.fit_transform(sample_data_2)
    assert processed.count() == 2
    assert "report_month" in processed.columns
    assert "comm_type" in processed.columns
    pd_df = processed.toPandas()
    timestamps = pd_df["timestamps"]
    assert sorted(timestamps[0]) == timestamps[0]
    assert sorted(timestamps[1]) == timestamps[1]
    assert isinstance(timestamps[0][0], float)
    assert ((np.array(timestamps[0])[1:] - np.array(timestamps[0])[:-1]) > 0).all()


def test_null_cat_values(sample_data_2, sample_data_3):
    # 1 case - fit using df without null values and transform null_values
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc"],
        numeric_columns=None,
        event_time_column="timestamps",
        id_column="epk_id",
        groupby_columns=["report_month", "comm_type"],
    )

    preprocessor.fit(sample_data_2)
    processed = preprocessor.transform(sample_data_3).toPandas()
    # First mcc for row with comm_type should be equal to 0
    assert processed[processed["comm_type"] == "SMS"]["mcc"].iloc[0][0] == 0

    # 2 case - fit_transform df with null values
    preprocessor = EventSequencePreprocessor(
        categorical_columns=["mcc"],
        numeric_columns=None,
        event_time_column="timestamps",
        id_column="epk_id",
        groupby_columns=["report_month", "comm_type"],
    )
    preprocessor.fit_transform(sample_data_3)
    assert processed[processed["comm_type"] == "SMS"]["mcc"].iloc[0][0] == 0
