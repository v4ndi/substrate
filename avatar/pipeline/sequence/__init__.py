"""Pipelines over event sequences.

Three uses of the same backbone: pool a sequence into one client embedding
(``SequenceModelWithAggregation``), classify from it
(``SequenceClassification``), or pretrain it by predicting the next K events
(``NextKTokensPrediction``).
"""

from .agg_hidden_states import SequenceModelWithAggregation
from .classification import SequenceClassification
from .next_k_tokens import NextKTokensPrediction

__all__ = [
    "NextKTokensPrediction",
    "SequenceClassification",
    "SequenceModelWithAggregation",
]
