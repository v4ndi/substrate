"""Local, single-machine preprocessing backend (pyarrow + numpy, no Spark).

Streaming ``fit`` / ``transform`` for datasets larger than RAM. The artifact
(``dump()`` / ``load()``) is interchangeable with
:mod:`avatar.preprocessing.spark`.
"""

from .label_encoder import LabelEncoder
from .pipeline import (
    EventSequencePreprocessor,
    NumCatPipeline,
    TabularPreprocessor,
)
from .standard_scaler import StandardScaler

__all__ = [
    "LabelEncoder",
    "StandardScaler",
    "NumCatPipeline",
    "TabularPreprocessor",
    "EventSequencePreprocessor",
]
