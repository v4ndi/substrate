"""The tabular embedding: categorical ids and numeric values to feature tokens.

Categorical columns share one table, indexed by the cumulative-offset ids the
preprocessor emits, which is why ``vocab_size`` is a single number across all
of them.
"""

import torch
import torch.nn as nn

from avatar.data.tabular.batch import TabularBatch
from avatar.nn.embedding.tabular.base import BaseTabularEmbedding
from avatar.nn.embedding.tabular.hidden_state_agg import BaseHiddenStateAggregator
from avatar.nn.embedding.tabular.numeric import NumericFeatureEmbedding


class TabularEmbedding(BaseTabularEmbedding):
    """Embed a tabular batch into one token per feature.

    All categorical columns share a single embedding table, indexed by the
    cumulative-offset ids the preprocessor emits — which is why ``vocab_size``
    is one number rather than one per column.

    Either family may be absent: pass ``num_numerical_features=None`` for a
    purely categorical table, or ``vocab_size=None`` for a purely numeric one.

    Args:
        num_numerical_features: How many numeric features, or None if there
            are none.
        vocab_size: Size of the shared categorical table, including special
            tokens such as ``unk``. None if there are no categorical features.
        hidden_size: Width of each feature token.
        std_noise: Standard deviation of Gaussian noise added to numeric
            features during training; None disables it.
        nn_embedding_config: Extra keyword arguments for ``nn.Embedding``.
        hidden_state_aggregator: How an external client embedding joins the
            tokens — see :mod:`avatar.nn.embedding.tabular.hidden_state_agg`.
        numerical_embedding: Embedding applied to non-NaN numeric features.
        num_embedding: Embedding applied to all numeric features.
        use_null_embedding: Give NaN numeric values their own learned vector
            instead of dropping them to zero.
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
            self.num_embed = NumericFeatureEmbedding(
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
        """Embed the batch into feature tokens.

        Args:
            tab_features: Batch with ``cat_features``
                ``(batch_size, n_cat_features)``, ``num_features``
                ``(batch_size, n_num_features)`` and optional ``hidden_states``.

        Returns:
            ``(batch_size, n_features, hidden_size)``, where ``n_features`` is
            the sum of both families plus any token the hidden-state aggregator
            added.
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
