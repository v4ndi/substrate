from __future__ import annotations

from copy import deepcopy

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base_preprocessor import BasePreprocessor


def signed_log1p(col: str):
    return F.log1p(F.abs(col)) * F.when(F.col(col) == 0, 0).otherwise(F.signum(col))


def _apply_log1p(df: DataFrame, cols: list[str] | None) -> DataFrame:
    """
    Apply signed log1p to selected columns in-place (same names).
    No-op if cols is None or empty.
    """
    if not cols:
        return df
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, signed_log1p(c))
    return df


class StandardScaler(BasePreprocessor):
    """Standard Scaler
    Args:
        columns (List[str]): list of numerical columns to scale;
        fillna (bool): whether to fill null values in transform.
            Null values will be replaced by 0;
        to_log_columns (list[str]): list of columns to apply log(x + 1);
    """

    def __init__(
        self,
        columns: list[str],
        fillna: bool = True,
        to_log_columns: list[str] | None = None,
    ) -> None:
        assert len(columns) > 0, "Expected list of columns"
        self.columns = columns
        self.fillna = fillna
        self.to_log_columns = list(to_log_columns) if to_log_columns is not None else []
        self.mean_std = {col: {"mean": 0.0, "std": 0.0} for col in columns}

    def fit(self, df: DataFrame) -> None:
        """Compute the mean and standard deviation for later scaling.
        Args:
            df (DataFrame): DataFrame containing the data
        """

        df_logged = _apply_log1p(df=df, cols=self.to_log_columns)

        mean_std_df = df_logged.agg(
            *[F.mean(col).alias(f"{col}_mean") for col in self.columns]
            + [F.stddev(col).alias(f"{col}_stddev") for col in self.columns]
        )
        mean_std_values = mean_std_df.collect()[0]

        self.mean_std = {
            col: {
                "mean": float(mean_std_values[f"{col}_mean"]),
                "std": float(mean_std_values[f"{col}_stddev"]),
            }
            for col in self.columns
        }

    def update(
        self,
        df: DataFrame,
        new_columns: list[str],
        new_to_log_columns: list[str] | None = None,
    ) -> None:
        """Add new columns to the scaler and compute their statistics.

        Args:
            df: DataFrame containing the new columns
            new_columns: List of new column names to add to the scaler

        Returns:
            None (updates the scaler in-place)
        """
        if new_to_log_columns:
            if self.to_log_columns is None:
                self.to_log_columns = []
            self.to_log_columns = list({*self.to_log_columns, *new_to_log_columns})

        if not new_columns:
            return

        duplicates = set(new_columns) & set(self.columns)
        if duplicates:
            raise ValueError(f"Columns already exist in scaler: {duplicates}")

        df_logged = _apply_log1p(df, self.to_log_columns)

        new_mean_std_df = df_logged.agg(
            *[F.mean(col).alias(f"{col}_mean") for col in new_columns]
            + [F.stddev(col).alias(f"{col}_stddev") for col in new_columns]
        )
        new_mean_std_values = new_mean_std_df.collect()[0]

        for col in new_columns:
            self.mean_std[col] = {
                "mean": float(new_mean_std_values[f"{col}_mean"]),
                "std": float(new_mean_std_values[f"{col}_stddev"]),
            }

        self.columns.extend(new_columns)

    def transform(self, df: DataFrame) -> DataFrame:
        """Scale the data using the computed mean and standard deviation."""
        df_logged = _apply_log1p(df=df, cols=self.to_log_columns)

        scaled_df = df_logged.select(*[
            (
                (F.col(col) - self.mean_std[col]["mean"])
                / (self.mean_std[col]["std"] + 1e-8)
            ).alias(f"{col}")
            if col in self.columns
            else F.col(col)
            for col in df.columns
        ])

        if self.fillna:
            scaled_df = scaled_df.fillna({col: 0.0 for col in self.columns})

        return scaled_df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Fit the scaler to the data and then transform the data."""
        self.fit(df)
        df = self.transform(df)
        return df

    def dump(self) -> dict[str, any]:
        """
        Creates a dictionary representation of the standard scaler.

        Returns:
            dict[str, any]:
                dictionary containing the attributes of the standard scaler
        """
        return deepcopy(self.__dict__)

    @classmethod
    def load(cls, attr_dict: dict[str, any]):
        """Creates a new instance of the standard scaler from a dictionary.

        Args:
            attr_dict (dict[str, any]):
                Dictionary containing the attributes of the label encoder

        Returns:
            clas: A new instance of the standard scaler
        """
        instance = cls.__new__(cls)
        # for supporting older versions
        if "fillna" not in attr_dict:
            attr_dict["fillna"] = True
        if "to_log_columns" not in attr_dict:
            attr_dict["to_log_columns"] = None

        instance.__dict__.update(deepcopy(attr_dict))
        return instance
