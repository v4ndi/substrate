from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Union

from pyspark.sql import DataFrame

from ..label_encoder import LabelEncoder
from ..standard_scaler import StandardScaler


class BaseDataPipeline(ABC):
    """Abstract base class for all data preprocessing pipelines.

    Defines the core interface that all concrete pipeline implementations must follow.
    """

    @abstractmethod
    def fit(self, df: DataFrame) -> None:
        """Learn preprocessing parameters from the training data.

        Args:
            df (DataFrame): Input data to learn preprocessing parameters from
        """
        pass

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply the learned transformations to the data.

        Args:
            df (DataFrame): Input data to transform

        Returns:
            Transformed data
        """
        pass

    @abstractmethod
    def fit_transform(self, df: DataFrame) -> DataFrame:
        """Convenience method to fit and transform in one step.

        Args:
            df (DataFrame): Input data to fit and transform

        Returns:
            Transformed data
        """
        pass


class NumCatPipeline(BaseDataPipeline):
    """Processes datasets containing both numeric and categorical features.

    Handles complete preprocessing pipeline for mixed-type data:
    - Label encoding for categorical features
    - Standard scaling for numeric features

    Args:
        categorical_columns: List of column names containing categorical data
        numeric_columns: List of column names containing numeric data
        label_encoder: Optional preconfigured LabelEncoder instance
        standard_scaler: Optional preconfigured StandardScaler instance
        label_encoder_kwargs (Dict[str, Any]): kwargs for LabelEncoder.
        standard_scaler_kwargs (Dict[str, Any]): kwargs for StandardScaler.

    Example:
        >>> processor = NumCatPipeline(
        ...     categorical_columns=["category1", "category2"],
        ...     numeric_columns=["feature1", "feature2"]
        ... )
        >>> processor.fit(training_data)
        >>> processed_data = processor.transform(new_data)
    """

    def __init__(
        self,
        categorical_columns: Union[list[str], None],
        numeric_columns: Union[list[str], None],
        label_encoder: Union[None, LabelEncoder] = None,
        standard_scaler: Union[None, StandardScaler] = None,
        label_encoder_kwargs: Dict[str, Any] = None,
        standard_scaler_kwargs: Dict[str, Any] = None,
    ):
        assert categorical_columns is not None or numeric_columns is not None
        self.cat_cols = categorical_columns
        self.num_cols = numeric_columns

        def proc_none_kwargs(x):
            return x if x is not None else {}

        label_encoder_kwargs = proc_none_kwargs(label_encoder_kwargs)
        standard_scaler_kwargs = proc_none_kwargs(standard_scaler_kwargs)

        if label_encoder is not None:
            self.label_encoder = label_encoder
        elif self.cat_cols is not None:
            self.label_encoder = LabelEncoder(
                self.cat_cols,
                spec_tokens={
                    "unk": 0
                },  # Необходимо для преобразования неизвестных значений
                **label_encoder_kwargs,
            )
        else:
            self.label_encoder = None

        if standard_scaler is not None:
            self.standard_scaler = standard_scaler
        elif self.num_cols is not None:
            self.standard_scaler = StandardScaler(
                columns=self.num_cols, **standard_scaler_kwargs
            )
        else:
            self.standard_scaler = None

    def fit(self, df: DataFrame):
        if self.num_cols is not None:
            assert sorted(self.num_cols) == sorted(self.standard_scaler.columns), (
                "num_cols is differ with columns in StandardScaler"
            )
            self.standard_scaler.fit(df)

        if self.cat_cols is not None:
            assert sorted(self.cat_cols) == sorted(self.label_encoder.columns), (
                "cat_cols is differ with columns in LabelEncoder"
            )
            self.label_encoder.fit(df)

    def transform(self, df: DataFrame) -> DataFrame:
        if self.cat_cols is not None:
            df = self.label_encoder.transform(df)
        if self.num_cols is not None:
            df = self.standard_scaler.transform(df)
        return df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        self.fit(df)
        df = self.transform(df)
        return df

    def dump(self):
        """
        Creates a dictionary representation of the label encoder.

        Returns:
            dict[str, any]:
                dictionary containing the attributes of the label encoder
        """
        processor_attr = deepcopy(self.__dict__)
        processor_attr["label_encoder"] = (
            deepcopy(self.label_encoder.dump())
            if self.label_encoder is not None
            else None
        )
        processor_attr["standard_scaler"] = (
            deepcopy(self.standard_scaler.dump())
            if self.standard_scaler is not None
            else None
        )
        return processor_attr

    @classmethod
    def load(cls, attr_dict: dict[str, any]):
        """Creates a new instance of the NumCatPipeline class from a dictionary.

        Args:
            attr_dict (dict[str, any]):
                Dictionary containing the attributes of the NumCatPipeline

        Returns:
            clas: A new instance of the NumCatPipeline
        """
        instance = cls.__new__(cls)
        loaded_attrs = deepcopy(attr_dict)
        if loaded_attrs.get("label_encoder"):
            instance.label_encoder = LabelEncoder.load(
                loaded_attrs.pop("label_encoder")
            )
        else:
            print("Not found label_encoder")
            instance.label_encoder = None

        if loaded_attrs.get("standard_scaler"):
            instance.standard_scaler = StandardScaler.load(
                loaded_attrs.pop("standard_scaler")
            )
        else:
            print("Not found standard_sclaer")
            instance.standard_scaler = None

        instance.__dict__.update(loaded_attrs)
        return instance
