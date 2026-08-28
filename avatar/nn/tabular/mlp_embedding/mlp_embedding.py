import torch
import torch.nn as nn


class MLPEmbedding(nn.Module):
    """
        This class is used how
        tabular_backbone in TabularClassification pipeline
        to embed target_attr_2 and target_attr_3 for MLPBenchmark
    Args:
       embedding_dim_attr_2: Embedding dimension for target_attr_2
       embedding_dim_attr_3: Embedding dimension for target_attr_3
    """

    def __init__(self, embedding_dim_attr_2=4, embedding_dim_attr_3=2):
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
