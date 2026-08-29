import torch
import torch.nn as nn

from avatar.data.tabular_batch import TabularBatch
from avatar.nn.embedding.tabular.base import BaseTabularEmbedding
from avatar.nn.embedding.tabular.hidden_state_agg import BaseHiddenStateAggregator
from avatar.nn.embedding.tabular.numeric import NumericFeatureEmbedding


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
