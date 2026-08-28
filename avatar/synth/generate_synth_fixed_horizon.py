import os
import random
import tempfile
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pyspark.sql.types as T

from avatar.synth import generate_tabular_df


def generate_fixed_horizon_dataset(
    spark,
    num_records: int = 1000,
    num_output_partitions: int = 8,
    num_events_range: tuple = (0, 100),
    num_ctx_events_range: tuple = (0, 400),
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

    assert num_records % num_output_partitions != 0

    schema = T.StructType([
        T.StructField("epk_id", T.LongType(), nullable=False),
        T.StructField("evt_dttm", T.ArrayType(T.TimestampType()), nullable=False),
        T.StructField("mcc", T.ArrayType(T.LongType()), nullable=False),
        T.StructField("price", T.ArrayType(T.FloatType()), nullable=False),
        T.StructField("event_ids", T.ArrayType(T.LongType()), nullable=False),
        T.StructField("ctx_events", T.ArrayType(T.LongType()), nullable=False),
        T.StructField("targets_events", T.ArrayType(T.LongType()), nullable=False),
        T.StructField("ctx_timestamps", T.ArrayType(T.FloatType()), nullable=False),
        T.StructField("targets_timestamps", T.ArrayType(T.FloatType()), nullable=False),
    ])
    data = []
    for i in range(num_records):
        epk_id = i + 1
        num_events = random.randint(num_events_range[0], num_events_range[1])
        num_events_ctx = random.randint(
            num_ctx_events_range[0], num_ctx_events_range[1]
        )
        num_events_targets = random.randint(
            num_ctx_events_range[0], num_ctx_events_range[1]
        )
        base_time = datetime.now() - timedelta(days=random.randint(1, 365))
        evt_dttm = sorted([base_time + timedelta(hours=j) for j in range(num_events)])
        mcc = [random.choice([0, 1, 2, 3, 4, 5]) for _ in range(num_events)]
        price = [round(random.uniform(1.0, 1000.0), 2) for _ in range(num_events)]
        event_ids = [random.choice([0, 1]) for _ in range(num_events)]
        ctx_events = [random.choice([0, 1, 2, 3, 4, 5]) for _ in range(num_events_ctx)]
        targets_events = [
            random.choice([0, 1, 2, 3, 4, 5]) for _ in range(num_events_ctx)
        ]
        ctx_timestamps = [
            round(random.uniform(1.0, 1000.0), 2) for _ in range(num_events_targets)
        ]
        targets_timestamps = [
            round(random.uniform(1.0, 1000.0), 2) for _ in range(num_events_targets)
        ]
        data.append((
            epk_id,
            evt_dttm,
            mcc,
            price,
            event_ids,
            ctx_events,
            targets_events,
            ctx_timestamps,
            targets_timestamps,
        ))
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
