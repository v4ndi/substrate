"""Spark tabular preprocessing: label-encode, standardise, pack into two columns."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..label_encoder import LabelEncoder
from ..standard_scaler import StandardScaler
from .base_pipe import NumCatPipeline


class TabularPreprocessor(NumCatPipeline):
    """A preprocessor for tabular data that handles both categorical and numerical features.

    This class extends NumCatPipeline to provide additional functionality for processing
    tabular data, including special token handling, vocabulary size calculation, and offset
    mapping for categorical features. It can transform input data into formats suitable for
    machine learning models while preserving specified identity columns.

    Args:
        categorical_columns (Union[list[str], None]): List of categorical column names.
        numeric_columns (Union[list[str], None]): List of numerical column names.
        spec_tokens (Union[None, dict[str, int]]): Special tokens dictionary (e.g., {"pad": 0}).
        label_encoder (Union[None, LabelEncoder]): Pre-initialized label encoder.
        standard_scaler (Union[None, StandardScaler]): Pre-initialized standard scaler.
        label_encoder_kwargs (Dict[str, Any]): kwargs for LabelEncoder.
        standard_scaler_kwargs (Dict[str, Any]): kwargs for StandardScaler.

    Attributes:
        spec_tokens (dict): Dictionary of special tokens and their encodings.
        offset_map (dict): Mapping of categorical columns to their value offsets.
        vocab_size (int): Total vocabulary size including special tokens and categorical values.

    Example:
        >>> preprocessor = TabularPreprocessor(
        ...     categorical_columns=["category_col"],
        ...     numeric_columns=["numeric_col"],
        ...     spec_tokens={"pad": 0}
        ... )
        >>> preprocessor.fit(df)
        >>> transformed_df = preprocessor.transform(df, identity_cols=["numeric_col"])
    """

    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        numeric_columns: list[str] | None = None,
        spec_tokens: dict[str, int] | None = None,
        label_encoder: LabelEncoder | None = None,
        standard_scaler: StandardScaler | None = None,
        label_encoder_kwargs: dict[str, Any] | None = None,
        standard_scaler_kwargs: dict[str, Any] | None = None,
    ):
        if spec_tokens is None:
            spec_tokens = {"pad": 0}
        super().__init__(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            label_encoder=label_encoder,
            standard_scaler=standard_scaler,
            label_encoder_kwargs=label_encoder_kwargs,
            standard_scaler_kwargs=standard_scaler_kwargs,
        )
        assert categorical_columns is not None or numeric_columns is not None, (
            "You should specify cat_cols or num_cols"
        )
        self.spec_tokens = spec_tokens
        self.offset_map = {}
        if self.cat_cols is not None:
            self.vocab_size = len(self.spec_tokens)
        else:
            self.vocab_size = 0

    def transform(self, df, identity_cols: list[str] | None = None):
        """Transform the input DataFrame into processed features.

        Converts categorical columns to encoded indices and numerical columns to scaled values,
        while optionally preserving specified identity columns in their original form.

        Args:
            df: Input DataFrame to transform.
            identity_cols: Optional list of column names to preserve in original form
                         while still including them in the feature arrays.

        Returns:
            A transformed DataFrame with:
            - 'cat_features' array column (if categorical columns exist)
            - 'num_features' array column (if numerical columns exist)
            - Original identity columns (if specified)

        Raises:
            ValueError: If neither categorical nor numerical columns were specified during initialization.
        """
        if identity_cols is not None:
            for col in identity_cols:
                df = df.withColumn(f"source_{col}", F.col(col))
        out = super().transform(df)

        if self.cat_cols is not None:
            # apply offset_map
            for col in self.cat_cols:
                out = out.withColumn(col, F.col(col) + self.offset_map[col])
            out = out.withColumn(
                "cat_features", F.array(self.cat_cols).cast(T.ArrayType(T.LongType()))
            )
        if self.num_cols is not None:
            out = out.withColumn(
                "num_features",
                F.array([col for col in self.num_cols]).cast(
                    T.ArrayType(T.FloatType())
                ),
            )

        cols_to_drop = (self.cat_cols or []) + (self.num_cols or [])
        out = out.drop(*cols_to_drop)
        if identity_cols is not None:
            for col in identity_cols:
                out = out.withColumnRenamed(f"source_{col}", col)

        return out

    def fit(self, df, create_offset_only: bool = False):
        """Fit the preprocessor to the input data.

        Learns categorical encodings and numerical scaling parameters, and calculates
        value offsets for categorical features.

        Args:
            df: Input DataFrame for fitting the preprocessor.
            create_offset_only: If True, only calculates offsets without fitting
                               the label encoder and scaler (useful when they're pre-initialized).
        """
        if not create_offset_only:
            super().fit(df)
        # create offset_map
        if self.cat_cols is not None:
            for col in self.cat_cols:
                n_unique = len(self.label_encoder.values_to_id[col])
                self.offset_map[col] = self.vocab_size
                self.vocab_size += n_unique

    def fit_transform(self, df, identity_cols=None):
        """Convenience method that combines fit and transform operations.

        Args:
            df: Input DataFrame to fit on and transform.
            identity_cols: Optional list of column names to preserve in original form.

        Returns:
            Transformed DataFrame as described in the transform method.
        """
        self.fit(df)
        df = self.transform(df, identity_cols)
        return df

    def dump(self):
        """Serialize the preprocessor state to a dictionary.

        Returns:
            dict: A dictionary containing all necessary state information
                  to recreate the preprocessor, including:
                  - Parent class state
                  - Special tokens
                  - Offset mapping
                  - Vocabulary size
        """
        dumped_state = deepcopy(super().dump())
        dumped_state.update({
            "spec_tokens": self.spec_tokens,
            "offset_map": self.offset_map,
            "vocab_size": self.vocab_size,
        })
        return deepcopy(dumped_state)

    @classmethod
    def load(cls, attr_dict):
        """Recreate a preprocessor instance from a serialized state.

        Args:
            attr_dict: Dictionary containing serialized preprocessor state.

        Returns:
            TabularPreprocessor: A new instance with the loaded state.
        """
        attr_dict = deepcopy(attr_dict)
        instance = super().load(attr_dict)

        instance.spec_tokens = attr_dict["spec_tokens"]
        instance.offset_map = attr_dict["offset_map"]
        instance.vocab_size = attr_dict["vocab_size"]

        return instance
