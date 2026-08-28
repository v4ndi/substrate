import torch.nn as nn

from avatar.data.tabular_batch import TabularBatch
from avatar.nn.tabular import BaseTabularBackbone
from avatar.nn.utils import get_aggregation_layer


class TabularWithAggregatedStates(nn.Module):
    """
    A simple module that takes a tabular backbone and adds a final aggregation layer
    and optionally concat external hidden_state

    Args:
        backbone: BaseTabularBackbone
        aggregation_config: aggregation config for avatar.nn.utils.get_aggregation_layer
        late_fusion_hidden_state_dim: int if not None self.layer_norm(tab_features.hidden_states)
            and concat with aggregated_states.
            aggregated_states (batch_size, emb_dim)
            tab_features.hidden_states (batch_size, hidden_size)
            output (batch_size, emb_dim + hidden_size)
    """

    def __init__(
        self,
        backbone: BaseTabularBackbone,
        aggregation_config={"name": "mean"},
        late_fusion_hidden_state_dim: int = None,
    ):
        super().__init__()
        self.backbone = backbone
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
            else self.backbone.embedding.hidden_size
        )
        self.output_dim += agg_output_dim

    def forward(self, tab_features: TabularBatch):
        states = self.backbone(tab_features)  # batch_size, num_features, hidden_size
        aggregated_states = self.agg_layer(states)  # batch_size, hidden_size

        # optionally concat with tab_features.hidden_states
        # DEPRECATED
        # if self.layer_norm is not None:
        #     hidden_states = self.layer_norm(tab_features.hidden_states)
        #     aggregated_states = torch.cat([aggregated_states, hidden_states], dim=-1)

        return aggregated_states
