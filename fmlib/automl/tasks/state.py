"""Values exchanged by boosting orchestration and artifact services."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from fmlib.automl.backends.interface import ModelBackend
from fmlib.automl.data import FeatureSchema

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


RUNTIME_UNSET: Any = object()
