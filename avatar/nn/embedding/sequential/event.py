from typing import Any

import torch
import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.embedding.base.primitives import LinearEmbeddings
from avatar.nn.embedding.sequential.base import BaseEventSequenceEmbedding


class EventSequenceEmbedding(BaseEventSequenceEmbedding):
    """Embedding layer for event sequence data that handles both categorical and numeric features.

    This class creates appropriate embeddings for each column in the input sequence based on
    their metadata (type, number of classes, etc.). Categorical features are embedded using
    torch.nn.Embedding, while numeric features use linear projections.

    Args:
        hidden_size: Dimensionality of the output embeddings.
        columns_meta: Dictionary mapping column names to their metadata specifications.
            Each value should be a dictionary with:
            - 'type': Either "categorical" or "numeric"
            - 'n_classes': For categorical, the number of distinct values (including spec tokens).
                          For numeric, must be 1.
            - Optional 'clip': For categorical, the maximum value (exclusive upper bound).
            - Optional 'event_id': Identifier(s) for the event type.

    Input Shape:
        - Forward input: EventSequenceBatch containing tensors for each column.
          Each tensor should have shape (batch_size, sequence_length).

    Output Shape:
        - Returns: torch.Tensor of shape (batch_size, sequence_length, n_columns, hidden_size)
          where n_columns is the number of columns in columns_meta.

    Example:
        >>> columns_meta = {
        ...     "txn_mcc": {"type": "categorical", "n_classes": 10, "clip": 8},
        ...     "price": {"type": "numeric", "n_classes": 1}
        ... }
        >>> embedding = EventSequenceEmbedding(hidden_size=32, columns_meta=columns_meta)
        >>> batch = EventSequenceBatch({"event_type": ..., "value": ...})
        >>> output = embedding(batch)  # shape: (batch_size, seq_len, 2, 32)

    Attributes:
        embedding_layer: nn.ModuleDict containing the embedding layers for each column.
        column_types: Tuple[str] containing the type of each column ("categorical"/"numeric").
        column_clips: Tuple[Optional[int]] containing clip values for categorical columns.
        column_n_classes: Tuple[int] containing n_classes for each column.

    Note:
        - For categorical columns with clip values, inputs are clamped to [0, clip-1].
        - Numeric columns are projected to hidden_size via a linear layer.
        - All embeddings are concatenated along the feature dimension (dim=2).
    """

    def __init__(
        self,
        hidden_size: int,
        columns_meta: dict[str, dict[str, Any]],
        std_noise: float | None = None,
    ):
        super().__init__(hidden_size=hidden_size, columns_meta=columns_meta)

        # Pre-process column types and parameters
        self.column_types = []
        self.column_clips = []
        self.column_n_classes = []
        self.std_noise = std_noise

        embedding_dict = {}
        for key, item in self.columns_meta.items():
            self.column_types.append(item["type"])
            self.column_clips.append(item.get("clip", None))
            self.column_n_classes.append(item["n_classes"])

            if item["type"] == "categorical":
                embedding_dict[key] = nn.Embedding(
                    num_embeddings=item.get("clip", item["n_classes"]),
                    embedding_dim=self.hidden_size,
                    padding_idx=0,
                )
            else:
                embedding_dict[key] = LinearEmbeddings(
                    n_features=1, hidden_size=self.hidden_size
                )

        self.embedding_layer = nn.ModuleDict(embedding_dict)
        # Convert to tuples/lists for better torch compilation
        self.column_types = tuple(self.column_types)
        self.column_clips = tuple(self.column_clips)
        self.column_n_classes = tuple(self.column_n_classes)

    def forward(self, features: EventSequenceBatch) -> torch.FloatTensor:
        sequence_embedding = []

        for idx, column in enumerate(self._columns_keys):
            col_type = self.column_types[idx]
            if self.column_clips[idx] is not None:
                if col_type == "categorical":
                    features[column] = features[column].clamp(
                        max=self.column_clips[idx] - 1
                    )
                else:
                    features[column] = features[column].clamp(
                        max=self.column_clips[idx]
                    )
            col_features = features[column]

            if col_type == "categorical":
                embedding = self.embedding_layer[column](col_features).unsqueeze(2)
            else:
                embedding = self.embedding_layer[column](col_features.unsqueeze(-1))

            sequence_embedding.append(embedding)
        sequence_embedding = torch.cat(sequence_embedding, dim=2)
        if self.std_noise is not None and self.training:
            sequence_embedding = (
                sequence_embedding
                + torch.randn_like(sequence_embedding) * self.std_noise
            )
        return sequence_embedding
