from typing import Literal

import torch
import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.embedding.base_embedding import BaseEventSequenceEmbedding
from avatar.nn.embedding.position_embeddings import BaseTemporalEmbedding
from avatar.nn.feature_encoder.base_feature_encoder import BaseSequenceFeatureEncoder


class IntraFeatureAttention(nn.Module):
    """Intra-feature attention mechanism for processing 2D event sequences.

    This module implements a scaled dot-product attention mechanism that operates
    across feature dimensions within each event in a sequence. Each feature attends
    to other features within the same event.

    Args:
        hidden_size (int): Dimensionality of input features and attention outputs.

    Input Shapes:
        - q, k, v: (batch_size, seq_len, num_features, hidden_size)
        - event_attention_mask: (batch_size, seq_len, num_features, num_features) or None

    Output Shape:
        - (batch_size, seq_len, num_features, hidden_size)

    Attributes:
        w_q (nn.Linear): Query transformation layer
        w_k (nn.Linear): Key transformation layer
        w_v (nn.Linear): Value transformation layer
        output (nn.Linear): Output projection layer

    Example:
        >>> attn = IntraFeatureAttention(hidden_size=64)
        >>> features = torch.randn(32, 10, 5, 64)  # (batch, seq_len, num_features, hidden)
        >>> output = attn(features, features, features)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def calcuate_attn_scores(
        self, q, k, event_attention_mask=None
    ) -> torch.FloatTensor:
        """Compute attention scores between features.

        Args:
            q: Query tensor (batch_size, seq_len, num_features, hidden_size)
            k: Key tensor (batch_size, seq_len, num_features, hidden_size)
            event_attention_mask: Optional attention mask (batch_size, seq_len, num_features, num_features)

        Returns:
            Attention scores tensor (batch_size, seq_len, num_features, num_features)
        """
        k = k.transpose(-2, -1)
        scores = torch.einsum("bsij, bsjk->bsik", [q, k]) / (self.hidden_size**0.5)
        if event_attention_mask is not None:
            event_attention_mask = event_attention_mask.to(scores.device)
            scores = scores.masked_fill(event_attention_mask == 0, float("-inf"))
        scores = torch.nn.functional.softmax(scores, dim=-1)
        return scores

    def forward(self, q, k, v, event_attention_mask=None):
        """Forward pass of the attention mechanism.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            event_attention_mask: Optional attention mask

        Returns:
            Attention-weighted output tensor
        """
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        scores = self.calcuate_attn_scores(q, k, event_attention_mask)
        output = torch.einsum("bsij,bsjk->bsik", [scores, v])
        output = self.output(output)
        return output


class AggFFNBlock(nn.Module):
    """Feature aggregation block combining attention and feed-forward layers.

    This block processes feature embeddings through:
    1. Intra-feature self-attention with residual connection
    2. Feed-forward network with SELU activation and layer normalization
    3. Sequence-level mean pooling and L2 normalization

    Args:
        cell_size (int): Number of features per event
        hidden_size (int): Dimensionality of input/output features
        dropout_p (float): Dropout probability (default: 0.1)

    Input Shapes:
        - inputs_embeds: (batch_size, seq_len, num_features, hidden_size)
        - event_attention_mask: (batch_size, seq_len, num_features, num_features) or None

    Output Shape:
        - (batch_size, hidden_size)  # L2-normalized pooled output

    Example:
        >>> block = AggFFNBlock(cell_size=5, hidden_size=64)
        >>> embeddings = torch.randn(32, 10, 5, 64)  # (batch, seq_len, num_features, hidden)
        >>> output = block(embeddings)  # shape (32, 64)
    """

    def __init__(
        self,
        cell_size: int,
        hidden_size: int,
        dropout_p: float = 0.1,
        aggregation_mode: str = "attention",
    ):
        super().__init__()
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self._set_mode_params(aggregation_mode)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_size * 4),
            nn.SELU(),
            nn.LayerNorm([cell_size, hidden_size * 4]),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm([cell_size, hidden_size]),
        )

    def _set_mode_params(self, aggregation_mode: str):
        if aggregation_mode not in ["attention", "mean", "linear"]:
            raise ValueError(f"aggregation_mode {aggregation_mode} is not supported")
        self.attn = None
        if aggregation_mode == "attention":
            self.attn = IntraFeatureAttention(self.hidden_size)
        self.linear = None
        if aggregation_mode == "linear":
            self.linear = nn.Linear(self.cell_size, 1)
        self.aggregation_mode = aggregation_mode

    def _get_original_mask(self, event_attention_mask: torch.LongTensor = None):
        if event_attention_mask is None:
            return None
        # TODO: mb change logic to not been conditioned on last elemnt
        # I suppose, that last elemnt is always attention on time, which is exists for every event
        return event_attention_mask[
            :, :, -1, :
        ]  # last row is attention on time (exists only in ones elements)

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        event_attention_mask: torch.LongTensor = None,
    ) -> torch.FloatTensor:
        """Forward pass through the aggregation block.

        Args:
            inputs_embeds: Input feature embeddings
            event_attention_mask: Optional attention mask

        Returns:
            L2-normalized mean-pooled sequence representation
        """
        output = (
            inputs_embeds
            + self.attn(
                q=inputs_embeds,
                k=inputs_embeds,
                v=inputs_embeds,
                event_attention_mask=event_attention_mask,
            )
            if self.aggregation_mode == "attention"
            else inputs_embeds
        )

        output = output + self.ffn(output)  # [bs, seq_len, n_features, emb_dim]
        feature_mask = self._get_original_mask(
            event_attention_mask
        )  # [bs, seq_len, n_features]

        masked_aggregated_output = output * feature_mask.unsqueeze(-1)
        if self.aggregation_mode == "mean" or self.aggregation_mode == "attention":
            num_useful_features = feature_mask.sum(dim=-1).unsqueeze(-1)
            aggregated_output = (masked_aggregated_output).sum(
                dim=-2
            ) / num_useful_features
        else:
            aggregated_output = self.linear(
                masked_aggregated_output.permute(0, 1, 3, 2)
            ).squeeze(-1)

        return torch.nn.functional.normalize(aggregated_output, dim=-1)


def get_event_attention_mask(
    seq_features: EventSequenceBatch, n_features: int, columns_meta
) -> torch.LongTensor:
    batch_size, seq_len = seq_features.attention_mask.shape
    attention_mask = (
        torch.ones(batch_size, seq_len, n_features).long().to(seq_features.device)
    )
    for i, (_, meta) in enumerate(columns_meta.items()):
        attention_mask[:, :, i] = seq_features.event_attn_mask(
            event_id=meta["event_id"]
        )
    expaned_attn_mask = torch.einsum(
        "bsij,bsjk->bsik",
        [attention_mask.unsqueeze(-1), attention_mask.unsqueeze(-2)],
    )
    diag_matrix = (
        torch.eye(n_features, dtype=torch.long)
        .unsqueeze(0)
        .unsqueeze(1)
        .to(seq_features.device)
    )
    expaned_attn_mask = torch.logical_or(expaned_attn_mask, diag_matrix)

    return expaned_attn_mask.long()


class FeatureEncoder(BaseSequenceFeatureEncoder):
    """Feature-aware sequence encoder with attention across event attributes.

    This encoder processes event sequences by:
    1. Applying base embeddings to raw features
    2. Computing feature-specific attention masks
    3. Aggregating features through attention and feed-forward layers

    Inherits from BaseSequenceFeatureEncoder and adds feature-level attention.

    Args:
        embedding (avatar.nn.embedding.BaseEventSequenceEmbedding):
            Base embedding module for raw features
        dropout_p (float): Dropout probability (default: 0.1)
        pos_embedding (BaseTemporalEmbedding): Optional positional embedding module
        time_encoding (Optional[Literal["absolute", "delta"]]): Optional time encoding.
            Check BaseSequenceFeatureEncoder for more details
        log_time_values (bool): Whether to log-transform time values (default: False)

    Input Shapes:
        - seq_features: EventSequenceBatch containing:
            - Various feature tensors
            - attention_mask: (batch_size, seq_len)
            - event_ids: Optional event identifiers

    Output Shape:
        - (batch_size, seq_len, hidden_size)  # Final sequence representation

    Example:
        >>> embedding = EventSequenceEmbedding(...)
        >>> encoder = FeatureAttentionEncoder(embedding)
        >>> batch = EventSequenceBatch(...)
        >>> output = encoder(batch)  # shape (batch_size, seq_len hidden_size)
    """

    def __init__(
        self,
        embedding: BaseEventSequenceEmbedding,
        dropout_p: float = 0.1,
        pos_embedding: BaseTemporalEmbedding = None,
        time_encoding: Literal["absolute", "delta"] | None = None,
        log_time_values: bool = False,
        id_embedding: nn.Module = None,
        aggregation_mode: str = "attention",
    ):
        super().__init__(
            embedding=embedding,
            pos_embedding=pos_embedding,
            time_encoding=time_encoding,
            log_time_values=log_time_values,
        )
        self.cell_size = len(self.embedding.columns_meta)
        if self.time_encoding is not None:
            self.cell_size += 1
        self.agg = AggFFNBlock(
            cell_size=self.cell_size,
            hidden_size=self.embedding.hidden_size,
            dropout_p=dropout_p,
            aggregation_mode=aggregation_mode,
        )
        self.id_embedding = id_embedding

    def event_attention_mask(
        self, seq_features: EventSequenceBatch
    ) -> torch.LongTensor:
        return get_event_attention_mask(
            seq_features, self.cell_size, self.embedding.columns_meta
        )

    def forward(self, seq_features: EventSequenceBatch) -> torch.FloatTensor:
        inputs_embeds = super().forward(
            seq_features
        )  # batch_size, seq_len, n_features, emb_dim
        if seq_features.event_ids is not None:
            event_attention_mask = self.event_attention_mask(seq_features)
        else:
            event_attention_mask = None

        agg_embeds = self.agg(
            inputs_embeds=inputs_embeds, event_attention_mask=event_attention_mask
        )  # batch_size, seq_len, emb_dim

        if self.id_embedding is not None:
            id_embed = self.id_embedding(seq_features["epk_id"]).unsqueeze(
                1
            )  # [bs, 1, emb_dim]
            # agg_embeds = agg_embeds + id_embed
            agg_embeds = torch.cat(
                [id_embed, agg_embeds], dim=1
            )  # [bs, seq_len + 1, emb_dim]

        return agg_embeds


# Deprecated
class FeatureAttentionEncoder(FeatureEncoder):
    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "FeatureAttentionEncoder is deprecated, use FeatureEncoder instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
