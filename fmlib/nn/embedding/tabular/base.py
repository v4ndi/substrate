"""Base class for tabular embeddings."""

from fmlib.data.tabular.batch import TabularBatch
from fmlib.nn.embedding.base.embedding import BaseEmbedding


class BaseTabularEmbedding(BaseEmbedding):
    """Base class for tabular embedding layers.

    Args:
        hidden_size: int - Dimensionality of the embeddings
    """

    def __init__(self, hidden_size: int):
        super().__init__(hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, tab_features: TabularBatch):
        raise NotImplementedError("Forward method must be implemented by child classes")
