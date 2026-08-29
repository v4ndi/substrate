from __future__ import annotations

from copy import deepcopy
from typing import Any

from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..label_encoder import LabelEncoder
from ..standard_scaler import StandardScaler
from ..utils import date_to_scaled_unix
from .base_pipe import NumCatPipeline


class EventSequencePreprocessor(NumCatPipeline):
    """Preprocessor for sequence features
    Args:
        event_time_column: str - column with date and time Default: "evt_dttm"
        id_column: str - name of column with id
        event_time_column: str - timestamp column
        time_unit: str - The time unit to scale by ('days', 'weeks' or 'months')
        event_type_ids_column: str - column containing event type identifiers (0, 1, 2, ... n);
        groupby_columns: Optional[List[str]] - list of additional columns for grouby operation
            for aggregation into a sequence of event attributes
            Default: None - df.groupby(id_column)
            if you pass groupby_columns -> df.groupby(id_column, *groupby_columns)
        label_encoder (Union[None, LabelEncoder]): Pre-initialized label encoder.
        standard_scaler (Union[None, StandardScaler]): Pre-initialized standard scaler.
        label_encoder_kwargs (Dict[str, Any]): kwargs for LabelEncoder.
        standard_scaler_kwargs (Dict[str, Any]): kwargs for StandardScaler.
    """

    def __init__(
        self,
        categorical_columns: list[str],
        numeric_columns: list[str] | None,
        event_time_column: str = "evt_dttm",
        time_unit: str = "days",
        event_type_ids_column: str = "event_ids",
        id_column: str = "epk_id",
        groupby_columns: list[str] | None = None,
        label_encoder: LabelEncoder | None = None,
        standard_scaler: StandardScaler | None = None,
        label_encoder_kwargs: dict[str, Any] | None = None,
        standard_scaler_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            label_encoder=label_encoder,
            standard_scaler=standard_scaler,
            label_encoder_kwargs=label_encoder_kwargs,
            standard_scaler_kwargs=standard_scaler_kwargs,
        )
        if numeric_columns is not None:
            assert not set(categorical_columns).intersection(set(numeric_columns)), (
                "Columns cannot be both categorical and numerical"
            )
        self.event_time_column = event_time_column
        self.id_column = id_column
        self.groupby_columns = groupby_columns if groupby_columns is not None else []
        self.event_type_ids_columns = event_type_ids_column
        if time_unit not in ["days", "weeks", "months"]:
            raise ValueError(
                f"Invalid time_unit: {time_unit}. Must be 'days', 'weeks', or 'months'"
            )
        self.time_unit = time_unit

    @property
    def columns(self):
        if self.num_cols is not None:
            return self.cat_cols + self.num_cols
        else:
            return self.cat_cols

    @property
    def sequence_columns(self):
        sequence_columns = (
            [self.event_type_ids_columns]
            + [self.event_time_column]
            + ([] or self.num_cols)
            + self.cat_cols
        )
        return sequence_columns

    @property
    def columns_meta(self):
        """Return columns meta informations.
        Return:
            Dict[str, Dict[str, any]]
            example_output = {
                "column_name_1": {
                    "type": "categorical",
                    "n_classes": 385,
                },
                "column_name_2": {
                    "type": "numeric"
                    "n_classes": 1
                }
            }
        """
        columns_meta = {}
        for column in self.columns:
            column_type = "categorical" if column in self.cat_cols else "numeric"
            if column_type == "numeric":
                n_classes = 1
            else:
                n_classes = len(self.label_encoder.values_to_id[column])

            columns_meta[column] = {
                "type": column_type,
                "n_classes": n_classes,
            }

        return columns_meta

    def transform(self, df):
        """Transform dataframe
        Args:
            df: pyspark.sql.functions.DataFrame
        Return:
            pyspark.sql.functions.DataFrame
        """
        encoded_df = super().transform(df)
        sorted_df = encoded_df.sort(self.id_column, self.event_time_column)
        sorted_df = sorted_df.withColumn(
            self.event_time_column,
            date_to_scaled_unix(F.col(self.event_time_column), self.time_unit),
        )
        agg_cols = []

        if self.num_cols is not None:
            for col in self.num_cols:
                agg_cols.append(
                    F.collect_list(F.col(col).cast(T.FloatType())).alias(col)
                )
            sorted_df = sorted_df.fillna({col: 0.0 for col in self.num_cols})

        for col in self.cat_cols:
            agg_cols.append(F.collect_list(F.col(col).cast(T.LongType())).alias(col))

        agg_cols.append(
            F.collect_list(F.col(self.event_time_column).cast(T.FloatType())).alias(
                self.event_time_column
            )
        )
        agg_cols.append(
            F.collect_list(F.col(self.event_type_ids_columns).cast(T.LongType())).alias(
                self.event_type_ids_columns
            )
        )
        aggregated_df = sorted_df.groupby(self.id_column, *self.groupby_columns).agg(
            *agg_cols
        )

        return aggregated_df

    def fit(self, df):
        """Fit preprocessor
        Args:
            df: pyspark.sql.DataFrame
        Return:
            None
        """
        super().fit(df)

    def fit_transform(self, df):
        """Fit preprocessor and transform dataframe
        Args:
            df: pyspark.sql.functions.DataFrame
        Return:
            pyspark.sql.functions.DataFrame
        """
        self.fit(df)
        df = self.transform(df)

        return df

    def dump(self):
        """Dump the preprocessor state to a dictionary.."""
        dumped_state = deepcopy(super().dump())
        dumped_state.update({
            "groupby_columns": self.groupby_columns,
            "id_column": self.id_column,
            "event_time_column": self.event_time_column,
            "event_type_ids_columns": self.event_type_ids_columns,
        })
        return deepcopy(dumped_state)

    @classmethod
    def load(cls, attr_dict):
        """Load preprocessor state from a dumped dictionary."""
        attr_dict = deepcopy(attr_dict)
        instance = super().load(attr_dict)

        instance.groupby_columns = attr_dict["groupby_columns"]
        instance.id_column = attr_dict["id_column"]
        instance.event_time_column = attr_dict["event_time_column"]
        instance.event_type_ids_columns = attr_dict["event_type_ids_columns"]

        return instance
