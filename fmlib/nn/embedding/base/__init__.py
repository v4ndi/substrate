"""Embedding pieces shared by both modalities."""

from fmlib.nn.embedding.base.embedding import BaseEmbedding
from fmlib.nn.embedding.base.primitives import (
    HashEmbedding,
    LinearEmbeddings,
    _check_input_shape,
)

__all__ = [
    "BaseEmbedding",
    "HashEmbedding",
    "LinearEmbeddings",
    "_check_input_shape",
]
