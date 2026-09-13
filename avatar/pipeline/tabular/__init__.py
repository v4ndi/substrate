"""Supervised pipelines over tabular batches.

``TabularWithAggregatedStates`` is the representation stack (embedding ->
encoder -> pooling); ``TabularClassification`` and ``SupervisedLearner`` put a
head and a loss on top of it.
"""

from .classification import TabularClassification
from .supervised import SupervisedLearner
from .tabular_aggregation import TabularWithAggregatedStates

__all__ = ["SupervisedLearner", "TabularClassification", "TabularWithAggregatedStates"]
