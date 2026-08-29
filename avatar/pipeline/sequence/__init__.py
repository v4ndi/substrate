from .agg_hidden_states import SequenceModelWithAggregation
from .classification import SequenceClassification
from .next_k_tokens import NextKTokensPrediction

__all__ = [
    "NextKTokensPrediction",
    "SequenceClassification",
    "SequenceModelWithAggregation",
]
