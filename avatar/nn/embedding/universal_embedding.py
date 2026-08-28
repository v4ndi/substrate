from typing import Any

import torch
import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.embedding.embeddings import LinearEmbeddings


class UniversalEmbedding(nn.Module):
    def __init__(
        self,
        attr_type: str,
        hidden_size: int,
        n_classes: int = None,
        one_hot_size: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attr_type = attr_type
        if attr_type == "numeric":
            self.embed = LinearEmbeddings(n_features=1, hidden_size=self.hidden_size)
        elif attr_type == "categorical":
            self.embed = nn.Embedding(
                num_embeddings=n_classes, embedding_dim=self.hidden_size
            )
        elif attr_type == "one_hot":
            self.embed = LinearEmbeddings(
                n_features=one_hot_size, hidden_size=self.hidden_size
            )

    def forward(self, feature_chain: torch.Tensor):
        if self.attr_type in ["linear", "basic"]:
            return self.embed(feature_chain)
        if self.attr_type == "embedding":
            return feature_chain.mean(dim=-1)
        sizes = feature_chain.sum(dim=-1)  # [bs, seq_len]
        return self.embed(feature_chain).sum(dim=-1) / (sizes + 1e-8)


class UniversalEventEmbedding(nn.Module):
    def __init__(
        self, hidden_size: int, columns_meta: dict[str, dict[str, Any]]
    ):  # categorical, numerical, one_hot
        super().__init__()
        self.hidden_size = hidden_size
        self.columns_meta = columns_meta
        self.col2embed = nn.ModuleDict()
        self._init_embeds()

    def _init_embeds(self):
        for column, info in self.columns_meta.items():
            print(info)
            self.col2embed[column] = UniversalEmbedding(
                hidden_size=self.hidden_size,
                attr_type=info["type"],
                n_classes=info["n_classes"]
                if "n_classes" in info and info["n_classes"] is not None
                else None,
                one_hot_size=info["one_hot_size"] if "one_hot_size" in info else None,
            )

    def forward(self, seq_features: EventSequenceBatch):
        embeds_list = []
        for col in self.columns_meta.keys():
            embeds_list.append(self.col2embed[col](seq_features[col]))
        return torch.cat(embeds_list, dim=-1)
