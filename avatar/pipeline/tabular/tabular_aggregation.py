import torch.nn as nn

from avatar.data.tabular_batch import TabularBatch
from avatar.nn.embedding import BaseTabularEmbedding
from avatar.nn.tabular import BaseTabularEncoder
from avatar.nn.utils import get_aggregation_layer


class TabularWithAggregatedStates(nn.Module):
    """Embed a tabular batch, contextualise the feature tokens, pool to a vector.

    ``embedding(batch) -> encoder(embeds) -> aggregation`` — the standard tabular
    representation stack. ``output_dim`` reports the pooled vector size for the
    downstream head.

    Args:
        embedding: tabular embedding layer (``TabularBatch -> (B, F, D)``).
        encoder: tabular feature-token encoder (``(B, F, D) -> BaseTabularOutput``).
        aggregation_config: config for ``avatar.nn.utils.get_aggregation_layer``.
        late_fusion_hidden_state_dim: reserved; adds to ``output_dim`` so the head
            can be sized for a concatenated external hidden state (the concat
            itself currently happens in the downstream pipeline).
    """

    def __init__(
        self,
        embedding: BaseTabularEmbedding,
        encoder: BaseTabularEncoder,
        aggregation_config=None,
        late_fusion_hidden_state_dim: int | None = None,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.agg_layer = get_aggregation_layer(**aggregation_config)
        if late_fusion_hidden_state_dim is not None:
            self.layer_norm = nn.LayerNorm(late_fusion_hidden_state_dim)
        else:
            self.layer_norm = None

        if late_fusion_hidden_state_dim is not None:
            self.output_dim = late_fusion_hidden_state_dim
        else:
            self.output_dim = 0

        agg_output_dim = (
            self.agg_layer.output_dim
            if self.agg_layer.output_dim is not None
            else self.embedding.hidden_size
        )
        self.output_dim += agg_output_dim

    def forward(self, tab_features: TabularBatch):
        embeds = self.embedding(tab_features)  # batch_size, num_features, hidden_size
        states = self.encoder(embeds)  # BaseTabularOutput
        aggregated_states = self.agg_layer(states)  # batch_size, hidden_size
        return aggregated_states
