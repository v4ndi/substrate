import math

import torch
import torch.nn as nn

from avatar.data.sequential.batch import EventSequenceBatch
from avatar.nn.embedding.sequential.base import BaseTemporalEmbedding


class TemporalPositionEncoding(BaseTemporalEmbedding):
    """A module creating time-based position encodings.

    Args:
        embedding_dim: int - The desired size of the output embedding.
        Unlike many position embedding implementations, this does not need to be even.
        max_timepoint: float - the maximum observed timepoint,
        used to initialize the frequency space.
        scale_factor: float - a scaling factor for the embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_timepoint: float = 10000.0,
        scale_factor: float = 0.05,
    ):
        super().__init__(embedding_dim=embedding_dim)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-math.log(max_timepoint) / embedding_dim)
        )

        if self.embedding_dim % 2 == 0:
            self.sin_div_term = nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = nn.Parameter(div_term, requires_grad=False)
        else:
            self.sin_div_term = nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = nn.Parameter(div_term[:-1], requires_grad=False)

        self.scale_factor = scale_factor

    def relative_time(self, timestamp: torch.FloatTensor) -> torch.FloatTensor:
        assert timestamp.dim() == 2, (
            "Timestamp tensor must have shape (batch_size, seq_len)"
        )
        return timestamp - timestamp[:, 0].unsqueeze(1)

    def forward(self, features: EventSequenceBatch) -> torch.FloatTensor:
        """Forward pass.
        Args:
            batch: The input batch to process.

        Returns:
            The temporal position embeddings tensor of shape (batch_size, seq_len, emb_dim)
        """
        t = features.timestamps
        bsz, seq_len = t.shape
        device = t.device
        t = t.unsqueeze(-1)

        temporal_embeddings = torch.zeros(
            bsz, seq_len, self.embedding_dim, device=device
        )

        temporal_embeddings[:, :, 0::2] = torch.sin(
            t * self.sin_div_term.unsqueeze(0).unsqueeze(0)
        )
        temporal_embeddings[:, :, 1::2] = torch.cos(
            t * self.cos_div_term.unsqueeze(0).unsqueeze(0)
        )

        return temporal_embeddings * self.scale_factor


class Time2VecEmbedding(BaseTemporalEmbedding):
    """The module implements Time2Vec embedding technique from the paper
    "Time2Vec: Learning a Vector Representation of Time"


    Args:
        embedding_dim: int - The desired size of the output embedding.
        encoding_method: str - Algorithm implementation method.
        scale_factor: float - A scaling factor for the embedding.
        relative_time: bool - Flag that changes the time for the first coordinate.
    """

    def __init__(
        self,
        embedding_dim: int,
        encoding_method: str = "sin_cos_scale",
        scale_factor: float = 0.05,
        relative_time: bool = False,
    ):
        super().__init__(embedding_dim=embedding_dim)
        self.scale_factor = scale_factor
        self.period_weight = nn.Parameter(torch.randn(self.embedding_dim))
        self.period_bias = nn.Parameter(torch.randn(self.embedding_dim))
        self.trend_weight = nn.Parameter(torch.randn(1))
        self.trend_bias = nn.Parameter(torch.randn(1))
        realized_encoding_methods = [
            "sin_cos_no_scale",
            "sin_no_scale",
            "sin_scale",
            "sin_cos_scale",
            "sin_cos_scale_no_first_coord",
        ]
        if encoding_method not in realized_encoding_methods:
            raise ValueError(
                f"Invalid encoding_method: {encoding_method}. Must be one of {realized_encoding_methods}"
            )
        self.encoding_method = encoding_method
        self.relative_time_flag = relative_time

    def relative_time(self, timestamp: torch.Tensor):
        assert timestamp.dim() == 2, (
            "Timestamp tensor must have shape (batch_size, seq_len)"
        )
        return timestamp - timestamp[:, 0].unsqueeze(1)

    def forward(self, features: EventSequenceBatch) -> torch.Tensor:
        time = features.timestamps
        batch_size, seq_len = time.shape
        device = time.device

        periodic_component = torch.empty(
            (batch_size, seq_len, self.embedding_dim), device=device
        )
        if self.relative_time_flag:
            trend_component = (
                self.trend_weight * self.relative_time(time) + self.trend_bias
            ).unsqueeze(-1)
        else:
            trend_component = (self.trend_weight * time + self.trend_bias).unsqueeze(-1)

        t_expanded = time.unsqueeze(-1)
        if self.encoding_method == "sin_cos_no_scale":
            periodic_component[:, :, 0::2] = torch.sin(
                t_expanded * self.period_weight[0::2] + self.period_bias[0::2]
            )
            periodic_component[:, :, 1::2] = torch.cos(
                t_expanded * self.period_weight[1::2] + self.period_bias[1::2]
            )
            t_encoded = torch.cat(
                [trend_component, periodic_component[:, :, 1::]], dim=-1
            )
            return t_encoded
        elif self.encoding_method == "sin_no_scale":
            periodic_component = torch.sin(
                t_expanded * self.period_weight + self.period_bias
            )
            t_encoded = torch.cat(
                [trend_component, periodic_component[:, :, 1::]], dim=-1
            )
            return t_encoded
        elif self.encoding_method == "sin_scale":
            periodic_component = torch.sin(
                t_expanded * self.period_weight + self.period_bias
            )
            t_encoded = torch.cat(
                [trend_component, periodic_component[:, :, 1::]], dim=-1
            )
            return t_encoded * self.scale_factor
        elif self.encoding_method == "sin_cos_scale":
            periodic_component[:, :, 0::2] = torch.sin(
                t_expanded * self.period_weight[0::2] + self.period_bias[0::2]
            )
            periodic_component[:, :, 1::2] = torch.cos(
                t_expanded * self.period_weight[1::2] + self.period_bias[1::2]
            )
            t_encoded = torch.cat(
                [trend_component, periodic_component[:, :, 1::]], dim=-1
            )
            return t_encoded * self.scale_factor
        elif self.encoding_method == "sin_cos_scale_no_first_coord":
            periodic_component[:, :, 0::2] = torch.sin(
                t_expanded * self.period_weight[0::2] + self.period_bias[0::2]
            )
            periodic_component[:, :, 1::2] = torch.cos(
                t_expanded * self.period_weight[1::2] + self.period_bias[1::2]
            )
            return periodic_component * self.scale_factor
