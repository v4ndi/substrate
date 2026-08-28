"""Backend-agnostic core for :mod:`avatar.preprocessing`.

This package holds everything that does not depend on a compute engine: the
streaming statistics accumulators, the pure-numpy transform math, the offset-map
builder and the parquet batch iterator. Both the Spark backend and the local
(pyarrow + numpy) backend are meant to reuse it.
"""

from .accumulators import MeanStdAccumulator, ValueCountAccumulator
from .encode import CategoricalMapper, signed_log1p, standardize
from .io import dataset_num_rows, iter_record_batches
from .offsets import build_offset_map

ARTIFACT_VERSION = 2

__all__ = [
    "ARTIFACT_VERSION",
    "MeanStdAccumulator",
    "ValueCountAccumulator",
    "CategoricalMapper",
    "signed_log1p",
    "standardize",
    "iter_record_batches",
    "dataset_num_rows",
    "build_offset_map",
]
