"""Embedding pieces shared by both modalities."""

from avatar.nn.embedding.base.embedding import BaseEmbedding
from avatar.nn.embedding.base.primitives import (
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
