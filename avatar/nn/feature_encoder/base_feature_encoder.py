from typing import Literal

import torch
import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.embedding import (
    BaseEventSequenceEmbedding,
    BaseTemporalEmbedding,
    LinearEmbeddings,
)


class BaseSequenceFeatureEncoder(nn.Module):
    """
    A class used to encode sequence features using embeddings and positional encoding.

    ...

    Attributes
    ----------
    embedding : BaseEventSequenceEmbedding
        The base event sequence embedding.
    pos_embedding : TemporalPositionEncoding, optional
        The temporal position encoding.
    time_encoding : str, optional
        The type of time encoding to use. Can be "absolute" or "delta".
    log_time_values : bool, optional
        Whether to log the time values.
    """

    def __init__(
        self,
        embedding: BaseEventSequenceEmbedding,
        pos_embedding: BaseTemporalEmbedding = None,
        time_encoding: Literal["absolute", "delta"] | None = None,
        log_time_values: bool = False,
    ):
        super().__init__()
        self.embedding = embedding
        self.pos_embedding = pos_embedding
        self.log_time_values = log_time_values
        self._set_time_encoding(time_encoding)

    def _set_time_encoding(self, time_encoding: Literal["absolute", "delta"] | None):
        self.time_encoding = time_encoding
        if self.time_encoding is None:
            self.time_encoding = "delta"
            print("time encoding is set to delta")
        assert self.time_encoding in ["delta", "absolute"], (
            "supportds only delta and absolute time encoding"
        )
        if self.time_encoding is not None:
            self.time_encoding_layer = LinearEmbeddings(
                n_features=1, hidden_size=self.embedding.hidden_size
            )

    def forward(self, seq_features: EventSequenceBatch) -> torch.FloatTensor:
        embeds = self.embedding(seq_features)

        if self.pos_embedding is not None:
            if embeds.dim() == 4:
                embeds += self.pos_embedding(seq_features).unsqueeze(2)
            elif embeds.dim() == 3:
                embeds += self.pos_embedding(seq_features)
            else:
                raise ValueError("Invalid input dimension")

        if self.time_encoding is not None:
            if self.time_encoding == "delta":
                input_timestamps = seq_features.get_timedeltas()
            elif self.time_encoding == "absolute":
                input_timestamps = seq_features.timestamps
            else:
                raise ValueError("Invalid time_encoding option")

            if self.log_time_values:
                input_timestamps = torch.log(input_timestamps + 1)
                assert input_timestamps.isnan().sum() == 0, (
                    "Found nan values in torch.log(delta_timestamps + 1)"
                )

            time_embeds = self.time_encoding_layer(input_timestamps.unsqueeze(-1))

            # !!!Necessarily!!! Concatenate only from the right side.
            if embeds.dim() == 4:
                embeds = torch.cat((embeds, time_embeds), dim=-2)
            elif embeds.dim() == 3:
                embeds = torch.cat((embeds, time_embeds), dim=-1)
            else:
                raise ValueError("Invalid input dimension")

        return embeds
