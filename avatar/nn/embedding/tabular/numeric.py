"""Per-feature embedding of standardised numeric columns."""

import torch
import torch.nn as nn
from torch import Tensor

from avatar.nn.embedding.base.primitives import LinearEmbeddings


class NumericFeatureEmbedding(nn.Module):
    """Embedding for numerical features with NaN-aware null embeddings.

    Wraps an inner numerical embedding (``LinearEmbeddings`` by default) and, where
    a feature value is NaN, replaces its embedding with a learnable per-feature
    "null" embedding.

    Args:
        n_features: the number of continuous features.
        hidden_size: the embedding size.
        numerical_embedding: embedding for non-NaN numerical features
        use_null_embedding: whether to use null embedding for NaN numerical features
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        numerical_embedding=None,
        use_null_embedding=True,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.use_null_embedding = use_null_embedding

        if self.n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features=}")
        if self.hidden_size <= 0:
            raise ValueError(f"d_embedding must be positive, however: {hidden_size=}")

        if numerical_embedding is not None:
            self.num_embed = numerical_embedding
        elif self.n_features is not None:
            self.num_embed = LinearEmbeddings(
                n_features=self.n_features, hidden_size=self.hidden_size
            )

        self.null_embedding = nn.Embedding(n_features, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        mask = torch.isnan(x)

        x = torch.nan_to_num(x, nan=0.0)
        num_embeds = self.num_embed(x)

        if not self.use_null_embedding:
            return num_embeds

        feature_indices = torch.arange(self.n_features, device=x.device)
        feature_indices = feature_indices.unsqueeze(0).expand(batch_size, -1)

        null_embeds = self.null_embedding(feature_indices)

        mask = mask.int().unsqueeze(-1)
        final_embeds = (num_embeds * (1 - mask)) + (null_embeds * mask)

        return final_embeds
