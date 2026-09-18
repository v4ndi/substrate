"""Embedding layers, split into three sub-packages.

- ``fmlib.nn.embedding.base`` -- primitives shared by both stacks
  (``BaseEmbedding``, ``LinearEmbeddings``, ``HashEmbedding``).
- ``fmlib.nn.embedding.sequential`` -- event-sequence embeddings
  (``EventSequenceEmbedding``, temporal/positional encodings).
- ``fmlib.nn.embedding.tabular`` -- tabular embeddings
  (``TabularEmbedding``, numeric embeddings, hidden-state aggregators).

The full public API is re-exported here so ``fmlib.nn.embedding.<Name>`` keeps
working (Hydra ``_target_`` configs, external imports).
"""

from fmlib.nn.embedding.base import BaseEmbedding, HashEmbedding, LinearEmbeddings
from fmlib.nn.embedding.sequential import (
    BaseEventSequenceEmbedding,
    BaseTemporalEmbedding,
    EventSequenceEmbedding,
    TemporalPositionEncoding,
    Time2VecEmbedding,
)
from fmlib.nn.embedding.tabular import (
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
