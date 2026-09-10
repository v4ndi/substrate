"""Categorical-attribute embedding for a downstream campaign model.

Used as the ``tabular_model`` of
:class:`~avatar.pipeline.tabular.TabularClassification` when the model is
trained over dumped embeddings plus two categorical campaign attributes — the
attributes go through this module, the embeddings go straight to the head.
"""

import torch
import torch.nn as nn


class MLPEmbedding(nn.Module):
    """Embed the two campaign categorical attributes into one flat vector.

    Unlike the tabular encoders it sits next to, this returns a ``(B, D)``
    vector rather than ``(B, F, D)`` feature tokens: the benchmark head consumes
    it directly, so there is nothing to aggregate.

    Args:
        embedding_dim_attr_2: embedding width for ``target_attr_2`` (5 classes).
        embedding_dim_attr_3: embedding width for ``target_attr_3`` (2 classes).

    Attributes:
        output_dim: width of the concatenated output, read by the downstream head.
    """

    def __init__(self, embedding_dim_attr_2: int = 4, embedding_dim_attr_3: int = 2):
        super().__init__()
        self.cat_emb_2 = nn.Embedding(
            num_embeddings=5, embedding_dim=embedding_dim_attr_2
        )
        self.cat_emb_3 = nn.Embedding(
            num_embeddings=2, embedding_dim=embedding_dim_attr_3
        )
        self.output_dim = embedding_dim_attr_2 + embedding_dim_attr_3

    def forward(self, tab_features, **kwargs):
        emb_targets_2 = self.cat_emb_2(tab_features.cat_features[:, 0])
        emb_targets_3 = self.cat_emb_3(tab_features.cat_features[:, 1])
        return torch.cat([emb_targets_2, emb_targets_3], dim=1)
