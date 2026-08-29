import torch
from torch import nn

from avatar.nn.tabular.base_tabular import BaseTabularEncoder
from avatar.nn.tabular.ste import (
    CrossAttentionEncoderBlock,
)
from avatar.nn.utils import get_aggregation_layer
from avatar.nn.utils.ffn import FeedForwardNetwork


class FeedForwardHead(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, dropout_p: float = 0.15, scale: int = 4
    ):
        super().__init__()

        hidden_size = scale * input_dim

        self.block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.SELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.block(x)


class TreatmentCrossAttnEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_p: float = 0.15,
    ):
        """
        Args:
            hidden_size: int - hidden size for encoder blocks
            num_heads: int, - number of attention heads
            num_layers: int - number of encoder layers
            attn_dropout: float = 0.15 - attention dropout
        """
        super().__init__()
        self.input_ffn = FeedForwardNetwork(input_dim=hidden_size, dropout_p=dropout_p)
        self.blocks = nn.ModuleList([
            CrossAttentionEncoderBlock(
                emb_dim=hidden_size,
                num_heads=num_heads,
                attn_dropout=dropout_p,
                ffn_dropout=dropout_p,
            )
            for _ in range(num_layers)
        ])

    def forward(self, feature_embeds, treat_embeds):
        """
        feature_embeds: Tensor - (batch_size, n_features, emb_dim)
        treat_embeds: Tensor - (batch_size, treatment_n_features, emb_dim)
        """
        x = self.input_ffn(treat_embeds)
        for block in self.blocks:
            x = block.forward(key=feature_embeds, value=feature_embeds, query=x)
        return x


class MultiTreatmentSTE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        tabular_encoder: BaseTabularEncoder,
        treatment_encoder: BaseTabularEncoder,
        aggregation_config: dict | None = None,
        hidden_state_dim: int | None = None,
        embedding_processor=None,
    ):
        """
        Args:
            embedding: nn.Module - layer for encoding tabular features to embedings,
            num_heads: int, - number of attention heads
            num_layers: int - number of encoder layers
            attn_dropout: float = 0.15 - attention dropout
            need_weights: bool = False - return attn_scores or not
        """
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()
        self.feature_encoder = tabular_encoder
        self.treatment_encoder = treatment_encoder

        final_embed_dim = hidden_size
        self.proj_layer = None
        if hidden_state_dim is not None:
            self.proj_layer = nn.Sequential(
                nn.LayerNorm(hidden_state_dim), nn.Linear(hidden_state_dim, hidden_size)
            )
            final_embed_dim += hidden_size

        self.basic_classification_head = nn.Sequential(
            FeedForwardHead(final_embed_dim, 1, scale=4), nn.Sigmoid()
        )

        treatement_head_scale = 2 if hidden_state_dim is not None else 4
        self.treatment_classification_head = nn.Sequential(
            FeedForwardHead(
                final_embed_dim + hidden_size, 1, scale=treatement_head_scale
            ),
            nn.Sigmoid(),
        )
        self.agg_layer = get_aggregation_layer(**aggregation_config)
        self.embedding_processor = embedding_processor

    def forward(
        self,
        tabular_embeds: torch.Tensor,
        treatment_embeds: torch.Tensor,
        targets: torch.Tensor = None,
        seq_hidden_states: torch.Tensor = None,
    ):
        """
        Args:
            tabular_embeds: Tensor - (batch_size, n_features, emb_dim)
            treatment_embeds: Tensor - (batch_size, treatment_n_features, emb_dim)
            seq_hidden_embeds: Tensor - (batch_size, emb_dim)
            targets: Tensor - (batch_size, )
        Returns:
            basic_probs: Tensor - (batch_size, )
            uplift: Tensor - (batch_size, treatment_n_features)

        treatment_n_features:
            implied, basic_probs[i] is basic probability of class 1 for sample i in contol group
                     treatment_probs[i, g] - is probablity of class 1 for sample i in treatment group g
        """
        feature_embeds = self.feature_encoder(tabular_embeds)

        feature_embedding = self.agg_layer(feature_embeds)

        if (
            self.embedding_processor is not None
            and targets is not None
            and self.training
        ):
            feature_embedding, targets = self.embedding_processor.forward(
                feature_embedding, targets
            )

        if seq_hidden_states is not None and self.proj_layer is not None:  # late fusion
            feature_embedding = torch.cat(
                [feature_embedding, self.proj_layer(seq_hidden_states)], dim=-1
            )

        basic_probs = self.basic_classification_head(feature_embedding).squeeze(-1)

        treatment_embedding = self.treatment_encoder(
            feature_embeds, treatment_embeds
        ).squeeze(1)

        treatment_probs = self.treatment_classification_head(
            torch.concatenate([treatment_embedding, feature_embedding], dim=1)
        ).squeeze(-1)

        if targets is None:
            return basic_probs, treatment_probs
        return basic_probs, treatment_probs, targets
