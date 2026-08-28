from __future__ import annotations

import yaml
from pyspark.sql import Column
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .pipeline.tabular_pipe import TabularPreprocessor


def dump_tabular_meta(
    tabular_preprocessor: TabularPreprocessor, save_meta: bool = False
):
    """
    Args:
        tabular_preprocessor: TabularPreprocessor
        save_meta: bool - if True meta will dump in yaml file
    Return:
        column_meta: {column_name: {
                "type": "categorical/numerical",
                "n_classes": int - only for categorical
                "offset": int - only for categorical
                "position": int - position in list of values
                "values_to_id": Mapping oriding values to id - only for categorical
                "offset_values_to_id": {val: idx + offset for val, idx in values_to_id.itmes()} -categorical only
                "std": float - standard deviation - only for numerical
                "mean": float - mean - only for numerical
            }
        }
        meta: {
            "vocab_size": int - the sum of unique values by categorical columns
            "num_numerical_features": int - len(numerical_columns)
        }
    """
    meta = {}
    if tabular_preprocessor.cat_cols is not None:
        meta["vocab_size"] = tabular_preprocessor.vocab_size
    else:
        meta["vocab_size"] = None

    if tabular_preprocessor.num_cols is not None:
        meta["num_numerical_features"] = len(tabular_preprocessor.num_cols)

    cat_cols_meta = {}
    if tabular_preprocessor.cat_cols is not None:
        for column in tabular_preprocessor.cat_cols:
            offset_value = tabular_preprocessor.offset_map[column]
            position = [
                i
                for i, col in enumerate(tabular_preprocessor.cat_cols)
                if col == column
            ]
            assert len(position) == 1
            position = position[0]
            values_to_id = tabular_preprocessor.label_encoder.values_to_id[column]
            offset_values_to_id = {
                value: idx + offset_value for value, idx in values_to_id.items()
            }
            cat_cols_meta[column] = {
                "type": "categorical",
                "n_classes": len(values_to_id),
                "offset": offset_value,
                "position": position,
                "values_to_id": values_to_id,
                "offset_values_to_id": offset_values_to_id,
            }

    num_cols_meta = {}
    if tabular_preprocessor.num_cols is not None:
        for column in tabular_preprocessor.num_cols:
            position = [
                i
                for i, col in enumerate(tabular_preprocessor.num_cols)
                if col == column
            ]
            assert len(position) == 1
            position = position[0]
            mean = tabular_preprocessor.standard_scaler.mean_std[column]["mean"]
            std = tabular_preprocessor.standard_scaler.mean_std[column]["std"]
            num_cols_meta[column] = {
                "type": "numerical",
                "position": position,
                "std": std,
                "mean": mean,
            }

    columns_meta = {**cat_cols_meta, **num_cols_meta}
    assert len(columns_meta) == len(
        tabular_preprocessor.num_cols + tabular_preprocessor.cat_cols
    )

    if save_meta:
        with open("tabular_columns_meta.yaml", "w") as f:
            yaml.dump(columns_meta, f)
        with open("tabular_meta.yaml", "w") as f:
            yaml.dump(meta, f)

    return columns_meta, meta


def date_to_scaled_unix(date_col: Column, time_unit: str = "days") -> Column:
    """
    Convert a date column to Unix timestamp and scale it by the specified time unit.

    Args:
        date_col: A PySpark Column containing date or timestamp values
        time_unit: The time unit to scale by ('days', 'weeks', or 'months')

    Returns:
        A Column with the scaled Unix timestamp value

    Example:
        df.withColumn("scaled_date", date_to_scaled_unix(F.col("date_col"), "weeks"))
    """
    # Convert date to Unix timestamp (seconds since 1970-01-01)
    unix_time = F.unix_timestamp(date_col)

    # Define conversion factors (in seconds)
    conversion_factors = {
        "days": 86400,  # 60 * 60 * 24
        "weeks": 604800,  # 60 * 60 * 24 * 7
        "months": 2629800,  # Approximate: 60 * 60 * 24 * 30.4375 (avg days/month)
    }

    # Get the appropriate conversion factor
    factor = conversion_factors.get(time_unit.lower())
    if factor is None:
        raise ValueError(
            f"Invalid time_unit: {time_unit}. Must be 'days', 'weeks', or 'months'"
        )

    return (unix_time / factor).cast(T.FloatType())
