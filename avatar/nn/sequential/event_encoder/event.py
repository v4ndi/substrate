from typing import Literal

import torch
import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.embedding import BaseEventSequenceEmbedding, BaseTemporalEmbedding
from avatar.nn.sequential.event_encoder.attention import (
    EventAggregator,
    build_event_attention_mask,
)
from avatar.nn.sequential.event_encoder.base import BaseEventEncoder


class EventEncoder(BaseEventEncoder):
    """Feature-aware sequence encoder with attention across event attributes.

    This encoder processes event sequences by:
    1. Applying base embeddings to raw features
    2. Computing feature-specific attention masks
    3. Aggregating features through attention and feed-forward layers

    Inherits from BaseEventEncoder and adds feature-level attention.

    Args:
        embedding (avatar.nn.embedding.BaseEventSequenceEmbedding):
            Base embedding module for raw features
        dropout_p (float): Dropout probability (default: 0.1)
        pos_embedding (BaseTemporalEmbedding): Optional positional embedding module
        time_encoding (Optional[Literal["absolute", "delta"]]): Optional time encoding.
            Check BaseEventEncoder for more details
        log_time_values (bool): Whether to log-transform time values (default: False)
        id_embedding (nn.Module): Optional embedding for a per-sequence id column;
            when given, its output is prepended as an extra leading token
        id_column (str): Column in the batch holding the id for ``id_embedding``
            (default: "epk_id")

    Input Shapes:
        - seq_features: EventSequenceBatch containing:
            - Various feature tensors
            - attention_mask: (batch_size, seq_len)
            - event_ids: Optional event identifiers

    Output Shape:
        - (batch_size, seq_len, hidden_size)  # Final sequence representation

    Example:
        >>> embedding = EventSequenceEmbedding(...)
        >>> encoder = EventEncoder(embedding)
        >>> batch = EventSequenceBatch(...)
        >>> output = encoder(batch)  # shape (batch_size, seq_len, hidden_size)
    """

    def __init__(
        self,
        embedding: BaseEventSequenceEmbedding,
        dropout_p: float = 0.1,
        pos_embedding: BaseTemporalEmbedding = None,
        time_encoding: Literal["absolute", "delta"] | None = None,
        log_time_values: bool = False,
        id_embedding: nn.Module = None,
        id_column: str = "epk_id",
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
        self.agg = EventAggregator(
            cell_size=self.cell_size,
            hidden_size=self.embedding.hidden_size,
            dropout_p=dropout_p,
            aggregation_mode=aggregation_mode,
        )
        self.id_embedding = id_embedding
        self.id_column = id_column

    def event_attention_mask(
        self, seq_features: EventSequenceBatch
    ) -> torch.LongTensor:
        return build_event_attention_mask(
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
            id_embed = self.id_embedding(seq_features[self.id_column]).unsqueeze(
                1
            )  # [bs, 1, emb_dim]
            # agg_embeds = agg_embeds + id_embed
            agg_embeds = torch.cat(
                [id_embed, agg_embeds], dim=1
            )  # [bs, seq_len + 1, emb_dim]

        return agg_embeds
