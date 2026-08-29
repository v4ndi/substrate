from typing import Any

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.nn.parameter import Parameter

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.data.tabular_batch import TabularBatch
from avatar.nn.embedding.base_embedding import (
    BaseEventSequenceEmbedding,
    BaseTabularEmbedding,
)
from avatar.nn.embedding.hidden_state_agg import BaseHiddenStateAggregator


def _check_input_shape(x: Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValueError(
            f"The input must have at least one dimension, however: {x.ndim=}"
        )
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            "The last dimension of the input was expected to be"
            f" {expected_n_features}, however, {x.shape[-1]=}"
        )


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    Shape

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    Examples
    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, hidden_size: int) -> None:
        """
        Args:
            n_features: the number of continuous features.
            hidden_size: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features=}")
        if hidden_size <= 0:
            raise ValueError(f"d_embedding must be positive, however: {hidden_size=}")

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, hidden_size))
        self.bias = Parameter(torch.empty(n_features, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_shape(x, self.weight.shape[0])
        return torch.addcmul(self.bias, self.weight, x[..., None])


class PLEEmbedding(nn.Module):
    """
    PLE embeddings for continuous features.
    Args:
        bins_path: path to the yaml file containing the bins (list[list[float]])
        hidden_size: the embedding size.
        version: the version of the PLE embeddings from rtdl_num_embeddings.PiecewiseLinearEmbeddings
        activation: whether to use activation function in rtdl_num_embeddings.PiecewiseLinearEmbeddings
    """

    def __init__(
        self,
        bins_path: str,
        hidden_size: int,
        version: str = "B",
        activation: bool = False,
    ) -> None:
        import rtdl_num_embeddings

        if hidden_size <= 0:
            raise ValueError(f"d_embedding must be positive, however: {hidden_size=}")

        super().__init__()
        with open(bins_path) as f:
            bins = yaml.safe_load(f)
        bins = [torch.Tensor(lst) for _, lst in bins.items()]

        self.embeds = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, d_embedding=hidden_size, activation=activation, version=version
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.embeds(x)


class NumEmbedding(nn.Module):
    """Embedding for numerical features

    Args:
        n_features: the number of continuous features.
        hidden_size: the embedding size.
        numerical_embedding: embedding for not Nan numerical features
        use_null_embedding: whether to use null embedding for numerical features
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


class TabularEmbedding(BaseTabularEmbedding):
    """Embedding for tabular data
    Args:
        num_numerical_features (int): total number of numerical features
        vocab_size (int): total number of unique categorical values + special tokens like 'unk'
        hidden_size (int): hidden size of the embedding
        std_noise (float): standard deviation of the noise
        nn_embedding_config: additional kwargs for nn.Embedding
        hidden_state_aggregator (Optional[BaseHiddenStateAggregator]):
            Additional module for combining tabular embeddings with extrenal hidden states
        numerical_embedding: embedding for not Nan numerical features
        num_embedding: embedding for all numerical features
        use_null_embedding (bool): whether to use null embedding for numerical features

    If you have only categorical features, you can set num_numerical_features to None,
    if only numerical features you can set vocab_size to None
    """

    def __init__(
        self,
        num_numerical_features: int | None,
        vocab_size: int | None,
        hidden_size: int,
        std_noise: float | None = None,
        nn_embedding_config: dict[str, any] | None = None,
        hidden_state_aggregator: BaseHiddenStateAggregator | None = None,
        numerical_embedding=None,
        num_embedding=None,
        use_null_embedding=True,
    ):
        if nn_embedding_config is None:
            nn_embedding_config = {}
        super().__init__(hidden_size=hidden_size)

        if num_numerical_features is not None:
            self.num_embed = NumEmbedding(
                n_features=num_numerical_features,
                hidden_size=self.hidden_size,
                numerical_embedding=numerical_embedding,
                use_null_embedding=use_null_embedding,
            )
        else:
            self.num_embed = num_embedding

        if vocab_size is not None:
            self.cat_embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
                **nn_embedding_config,
            )
        self.std_noise = std_noise
        self.hidden_state_aggregator = hidden_state_aggregator

    def forward(self, tab_features: TabularBatch) -> torch.FloatTensor:
        """
        Args:
            tab_features: TabularBatch
            tab_features.cat_features.shape = (batch_size, n_cat_features)
            tab_features.num_features.shape = (batch_size, m_num_features)
            tab_features.hidden_states.shape = (batch_size, hidden_size)
        """
        cat_features = tab_features.cat_features
        num_features = tab_features.num_features
        assert cat_features is not None or num_features is not None

        if cat_features is not None:
            cat_embeds = self.cat_embed(
                cat_features
            )  # batch_size, n_cat_features, emb_dim

        if num_features is not None:
            num_embeds = self.num_embed(
                num_features
            )  # batch_size, m_num_features, emb_dim

        if cat_features is not None and num_features is not None:
            combined = torch.cat([cat_embeds, num_embeds], dim=1)
        elif cat_features is None:
            combined = num_embeds
        elif num_features is None:
            combined = cat_embeds

        if self.training and self.std_noise is not None:
            combined = combined + (
                torch.randn(combined.size()).to(combined.device) * self.std_noise
            )
        if self.hidden_state_aggregator is not None:
            combined = self.hidden_state_aggregator(
                hidden_states=tab_features.hidden_states, embeddings=combined
            )
        return combined  # batch_size, (n_cat_features + m_num_features), emb_dim


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
