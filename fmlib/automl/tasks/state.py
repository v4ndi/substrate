"""Values exchanged by task orchestration and artifact services."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import polars as pl

from fmlib.automl.backends.interface import ModelBackend
from fmlib.automl.data import FeatureSchema, ParquetSource

_BackendT = TypeVar("_BackendT", bound=ModelBackend)


@dataclass(frozen=True)
class ModelEntry(Generic[_BackendT]):
    """Fitted backend and the schema and metrics associated with it."""

    backend: _BackendT
    schema: FeatureSchema
    hidden_dimensions: dict[str, int]
    best_params: dict[str, Any]
    validation_metric: float
    layout: str = "global"
    group_value: Any | None = None
    single_group_global: bool = False
    single_group_value: Any | None = None


@dataclass(frozen=True)
class TrainingInput:
    """One split as a model part receives it, materialized or not.

    Boosting needs the rows in memory: it builds engine matrices out of numpy
    arrays, and its own limitation is that the split has to fit. TabNN needs
    the opposite -- it streams batches out of preprocessed parquet and never
    holds a split -- so it receives the source and reads it itself.

    Attributes:
        source: The parquet source of the whole split, always present.
        frame: Materialized rows of this model part, or ``None`` when the
            backend reads the source itself.
        group_value: The group this part is trained on under ``per_group``,
            ``None`` for the global part. A non-materializing backend needs it
            because the filtering it stands for has not been applied.
    """

    source: ParquetSource
    frame: pl.DataFrame | None = None
    group_value: Any | None = None

    def require_frame(self) -> pl.DataFrame:
        """Return the materialized rows, or say who failed to ask for them.

        Returns:
            The materialized frame of this model part.

        Raises:
            RuntimeError: If this input was built without materializing.
        """
        if self.frame is None:
            msg = (
                "This training input was not materialized: the backend family is "
                "registered as streaming its own data, but the fit path asked for "
                "a frame"
            )
            raise RuntimeError(msg)
        return self.frame


RUNTIME_UNSET: Any = object()
