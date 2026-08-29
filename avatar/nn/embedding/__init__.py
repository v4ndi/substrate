"""Embedding layers, split into three sub-packages:

- ``avatar.nn.embedding.base`` -- primitives shared by both stacks
  (``BaseEmbedding``, ``LinearEmbeddings``, ``HashEmbedding``).
- ``avatar.nn.embedding.sequential`` -- event-sequence embeddings
  (``EventSequenceEmbedding``, temporal/positional encodings).
- ``avatar.nn.embedding.tabular`` -- tabular embeddings
  (``TabularEmbedding``, numeric embeddings, hidden-state aggregators).

The full public API is re-exported here so ``avatar.nn.embedding.<Name>`` keeps
working (Hydra ``_target_`` configs, external imports).
"""

from avatar.nn.embedding.base import BaseEmbedding, HashEmbedding, LinearEmbeddings
from avatar.nn.embedding.sequential import (
    BaseEventSequenceEmbedding,
    BaseTemporalEmbedding,
    EventSequenceEmbedding,
    TemporalPositionEncoding,
    Time2VecEmbedding,
)
from avatar.nn.embedding.tabular import (
    BaseHiddenStateAggregator,
    BaseTabularEmbedding,
    LayerNormConcatenate,
    LayerNormSum,
    NumericFeatureEmbedding,
    TabularEmbedding,
)

__all__ = [
    "BaseEmbedding",
    "BaseEventSequenceEmbedding",
    "BaseHiddenStateAggregator",
    "BaseTabularEmbedding",
    "BaseTemporalEmbedding",
    "EventSequenceEmbedding",
    "HashEmbedding",
    "LayerNormConcatenate",
    "LayerNormSum",
    "LinearEmbeddings",
    "NumericFeatureEmbedding",
    "TabularEmbedding",
    "TemporalPositionEncoding",
    "Time2VecEmbedding",
]
