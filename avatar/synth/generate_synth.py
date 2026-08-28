import os
import random
import tempfile
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pyspark.sql.types as T


def generate_tabular_df(
    spark,
    num_records: int = 1000,
    num_categorical_features: int = 10,
    num_numeric_features: int = 10,
):
    """Generate processed tabular features
    Args:
        spark: SparkSession
        num_records: int = 1000 - number of rows
        num_categorical_features: int = 10 - number of categorical features
        num_numeric_features: int = 10 - number of numeric features
    Return:
        pandas.DataFrame
            epk_id: int64
            cat_features: np.ndarray(int64)
            num_features: np.ndarray(float32)

        NUMBER OF UNIQUE CAT_FEATURES = num_numeric_features * 5
    """
    schema = T.StructType([
        T.StructField("epk_id", T.LongType(), nullable=False),
        T.StructField("cat_features", T.ArrayType(T.LongType()), nullable=False),
        T.StructField("num_features", T.ArrayType(T.FloatType()), nullable=False),
    ])

    data = []
    for i in range(num_records):
        epk_id = i + 1
        cat_features = [
            random.randint((i + 1) * 5 - 4, (i + 1) * 5)
            for i in range(num_categorical_features)
        ]
        num_features = [
            round(random.uniform(1.0, 1000.0), 2) for _ in range(num_numeric_features)
        ]
        data.append((epk_id, cat_features, num_features))

    df = spark.createDataFrame(data, schema=schema)
    return df.toPandas()


def generate_sequence_dataset(
    spark,
    num_records: int = 1000,
    num_output_partitions: int = 8,
    num_events_range: tuple = (0, 100),
    target_column: Optional[str] = None,
    tabular_features: bool = False,
) -> str:
    """Generate parquet files with sequence features
    Args:
        spark: SparkSession
        num_records: int = 1000 - number of synth records
        num_output_partitions:int = 8 - number of output parquet files
        num_events_range: tuple(int, int) = (0, 100) - min max events in sequence
        target_column: Optional[str] Default None
            Possible values:
                - "classification" - generate binary target
        tabular_features: bool - if True add categorical and numeric features
    Return:
        output_path - path to folder with parquet files
        dataframe.schema:
        if target_column is None:
            schema = T.StructType([
                T.StructField("epk_id", T.LongType(), nullable=False),
                T.StructField("evt_dttm", T.ArrayType(T.TimestampType()), nullable=False),
                T.StructField("mcc", T.ArrayType(T.LongType()), nullable=False),
                T.StructField("price", T.ArrayType(T.FloatType()), nullable=False),
                T.StructField("event_ids", T.ArrayType(T.LongType()), nullable=False),
            ])
        elif target_column == "classification":
              schema = T.StructType([
                T.StructField("epk_id", T.LongType(), nullable=False),
                T.StructField("evt_dttm", T.ArrayType(T.TimestampType()), nullable=False),
                T.StructField("mcc", T.ArrayType(T.LongType()), nullable=False),
                T.StructField("price", T.ArrayType(T.FloatType()), nullable=False),
                T.StructField("target", T.LongType(), nullable=False),
                T.StructField("event_ids", T.ArrayType(T.LongType()), nullable=False),
            ])

        mcc n_classes = 6
        event_ids in range(3) # 0, 1
    """

    # assert num_records % num_output_partitions != 0

    if target_column is None:
        schema = T.StructType([
            T.StructField("epk_id", T.LongType(), nullable=False),
            T.StructField("evt_dttm", T.ArrayType(T.TimestampType()), nullable=False),
            T.StructField("mcc", T.ArrayType(T.LongType()), nullable=False),
            T.StructField("price", T.ArrayType(T.FloatType()), nullable=False),
            T.StructField("event_ids", T.ArrayType(T.LongType()), nullable=False),
        ])
    elif target_column == "classification":
        schema = T.StructType([
            T.StructField("epk_id", T.LongType(), nullable=False),
            T.StructField("evt_dttm", T.ArrayType(T.TimestampType()), nullable=False),
            T.StructField("mcc", T.ArrayType(T.LongType()), nullable=False),
            T.StructField("price", T.ArrayType(T.FloatType()), nullable=False),
            T.StructField("target", T.LongType(), nullable=False),
            T.StructField("event_ids", T.ArrayType(T.LongType()), nullable=False),
        ])
    else:
        raise ValueError(f"Invalid target_column: {target_column}")

    data = []
    for i in range(num_records):
        epk_id = i + 1
        num_events = random.randint(num_events_range[0], num_events_range[1])
        base_time = datetime.now() - timedelta(days=random.randint(1, 365))
        evt_dttm = sorted([base_time + timedelta(hours=j) for j in range(num_events)])
        mcc = [random.choice([0, 1, 2, 3, 4, 5]) for _ in range(num_events)]
        price = [round(random.uniform(1.0, 1000.0), 2) for _ in range(num_events)]
        event_ids = [random.choice([0, 1]) for _ in range(num_events)]
        if target_column is None:
            data.append((epk_id, evt_dttm, mcc, price, event_ids))
        elif target_column == "classification":
            targets = random.randint(0, 1)
            data.append((epk_id, evt_dttm, mcc, price, targets, event_ids))

    df = spark.createDataFrame(data, schema=schema)
    temp_dir = tempfile.mkdtemp()
    temp_dir = os.path.join(temp_dir, "synthetic_sequence_data")

    df = df.toPandas()
    df["partition"] = df["epk_id"] % num_output_partitions

    if tabular_features:
        tab_features_df = generate_tabular_df(
            spark=spark,
            num_records=num_records,
            num_categorical_features=10,
            num_numeric_features=10,
        )
        df = pd.merge(df, tab_features_df, how="inner", on=["epk_id"])
        assert df.shape[0] == tab_features_df.shape[0]

    df.to_parquet(temp_dir, partition_cols=["partition"], engine="pyarrow")

    return temp_dir
