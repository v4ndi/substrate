from __future__ import annotations

from copy import deepcopy
from itertools import chain

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .base_preprocessor import BasePreprocessor


class LabelEncoder(BasePreprocessor):
    """Label Encoder
    Args:
        columns: List[str] - list of categorical columns
        spec_tokens: Dict[str, int] - special tokens must be from 0 to k
            where k is the total number of spec_tokens.
            Also spec_tokens dict must contains "unk".
        frequency_encoder: bool - if True, the encoder will use frequency encoding
    """

    def __init__(
        self,
        columns: list[str],
        spec_tokens: dict[str, int] | None = None,
        frequency_encoder=False,
    ):
        if spec_tokens is None:
            spec_tokens = {"unk": 0}
        assert len(columns) > 0, "Expected list of columns"
        assert list(range(len(spec_tokens))) == sorted([
            val for val in spec_tokens.values()
        ]), "spec_tokens must be in range from 0 to k where k is len(spec_tokens)"
        assert "unk" in spec_tokens, "speck token unk not found"
        assert spec_tokens["unk"] == 0

        self.frequency_encoder = frequency_encoder
        self.columns = columns
        self.spec_tokens = spec_tokens
        self.values_to_id = {col: spec_tokens.copy() for col in self.columns}

    def _create_mapping_dict(self, df: DataFrame, columns: list[str]) -> None:
        """create mapping dictionary
        Fill dictionary in the next way:
            self.values_to_id[column_name][source_value] = destination_value

        Args:
            df (DataFrame)
            columns (list[str]): list of categorical columns
        """
        if not self.frequency_encoder:
            unique_values_df = df.select(*[
                F.collect_set(col).alias(col) for col in columns
            ]).collect()
            unique_values_dict = {col: unique_values_df[0][col] for col in columns}
        else:
            unique_values_dict = self.get_ordered_unique_values(
                df=df, categorical_cols=columns
            )
        for col in unique_values_dict:
            for i, value in enumerate(
                unique_values_dict[col], start=len(self.values_to_id[col])
            ):
                assert value not in self.values_to_id[col]
                self.values_to_id[col][value] = i

    def fit(self, df: DataFrame):
        """Fit the label encoder to the data."""
        assert len(self.values_to_id[self.columns[0]]) == len(self.spec_tokens), (
            "Your preprocessor is already fitted"
        )
        self._create_mapping_dict(df, self.columns)

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the labels to normalized encoding."""
        for col, mapping in self.values_to_id.items():
            mapping_expr = F.create_map([F.lit(x) for x in chain(*mapping.items())])
            default_value = mapping["unk"]
            df = df.withColumn(
                col, F.coalesce(mapping_expr[df[col]], F.lit(default_value))
            )

        return df

    def update(self, df: DataFrame, new_columns: list[str]) -> None:
        """Add new columns to the preprocessor and fit them.

        Args:
            df: DataFrame containing the new columns
            new_columns: List of new column names to add to the preprocessor

        Returns:
            None (updates the preprocessor in-place)
        """
        if not new_columns:
            return

        duplicates = set(new_columns) & set(self.columns)
        if duplicates:
            raise ValueError(f"Columns already exist in preprocessor: {duplicates}")

        for col in new_columns:
            self.values_to_id[col] = self.spec_tokens.copy()

        self._create_mapping_dict(df, new_columns)
        self.columns.extend(new_columns)

    def count_unk(self, transformed_df: DataFrame) -> dict[str, int]:
        """
        Count occurrences of UNK values in transformed DataFrame.

        Args:
            transformed_df: DataFrame processed by transform() method

        Returns:
            Dictionary with:
            - 'total': total UNK count across all columns
            - 'per_column': dict of UNK counts per column
        """
        unk_counts = {"total": 0, "per_column": {}}

        for col in self.columns:
            unk_id = self.values_to_id[col].get("unk")
            if unk_id is None:
                raise ValueError(f"Column '{col}' has no UNK token defined")

            count_expr = F.sum(F.when(F.col(col) == unk_id, 1).otherwise(0)).alias(col)
            result = transformed_df.select(count_expr).collect()[0]
            column_unks = result[col]

            unk_counts["per_column"][col] = column_unks
            unk_counts["total"] += column_unks

        return unk_counts

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Fit the label encoder to the data and then transform the data."""
        self.fit(df)
        df = self.transform(df)
        return df

    def get_ordered_unique_values(
        self, df: DataFrame, categorical_cols: list[str]
    ) -> dict[str, list[int]]:
        """Return unique values ordered by frequency
        Args:
            df (DataFrame): _description_
            categorical_cols (list[str]): _description_

        Returns:
            dict[str, list[int]]: unique values for each column ordered by frequency
            example_output = {
                "column_name": [1, 101, 202], # 1 - is the most frequent value
                ...
            }
        """
        final = {}
        for col in categorical_cols:
            res = df.groupBy(col).agg(F.count(F.lit(1)).alias("_cnt"))
            res = res.withColumn(
                "_rn",
                F.row_number().over(Window.partitionBy().orderBy(F.col("_cnt").desc())),
            ).collect()
            res = {row[col]: row["_rn"] for row in res}
            if None in res:
                res.pop(None)
            res = dict(sorted(res.items(), key=lambda item: item[1], reverse=False))
            final[col] = res
        return {key: list(value.keys()) for key, value in final.items()}

    def dump(self) -> dict[str, any]:
        """
        Creates a dictionary representation of the label encoder.

        Returns:
            dict[str, any]:
                dictionary containing the attributes of the label encoder
        """
        return deepcopy(self.__dict__)

    @classmethod
    def load(cls, attr_dict: dict[str, any]):
        """Creates a new instance of the label encoder from a dictionary.

        Args:
            attr_dict (dict[str, any]):
                Dictionary containing the attributes of the label encoder

        Returns:
            clas: A new instance of the label encoder
        """
        instance = cls.__new__(cls)
        instance.__dict__.update(deepcopy(attr_dict))
        return instance
