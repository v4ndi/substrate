import torch
import torch.nn as nn

from avatar.data.tabular_batch import TabularBatch
from avatar.nn.embedding import BaseEmbedding


class BaseTabularBackbone(nn.Module):
    def __init__(self, embedding: BaseEmbedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, tab_features: TabularBatch):
        """
        Args:
            tab_features (TabularBatch): input data
        """
        raise ValueError("Must be implemented in subclass")


class BaseTabularEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeds: torch.FloatTensor):
        """
        Args:
            input_embeds torch.Tensor: input embeddings
        """
        raise ValueError("Must be implemented in subclass")
