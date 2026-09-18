"""Boosting-engine adapters."""

from .binary import BinaryBoostingBackend
from .hyperopt import (
    BoostingFitResult,
    fit_boosting_model,
    resolve_default_search_space,
    suggest_params,
)
from .multiclass import MulticlassBoostingBackend
from .regression import RegressionBoostingBackend
from .uplift import UPLIFT_SCORE_COLUMNS, UpliftBoostingBackend

__all__ = [
    "UPLIFT_SCORE_COLUMNS",
    "BinaryBoostingBackend",
    "BoostingFitResult",
    "MulticlassBoostingBackend",
    "RegressionBoostingBackend",
    "UpliftBoostingBackend",
    "fit_boosting_model",
    "resolve_default_search_space",
    "suggest_params",
]
