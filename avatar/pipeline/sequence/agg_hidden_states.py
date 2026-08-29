import torch
import torch.nn as nn

from avatar.nn.sequence import BaseSequenceModel
from avatar.nn.utils.agg import get_aggregation_layer
from avatar.outputs import BaseSequenceOutput


class SequenceModelWithAggregation(nn.Module):
    """Aggregate hidden states from sequence model
    Args:
        sequence_model: BaseSequenceModel,
        model_weights: str = None,
        freeze_backbone: bool = False,
        aggregation_layer: dict[str, any] = {"name": "mean"},
    """

    def __init__(
        self,
        sequence_model: BaseSequenceModel,
        model_weights: str | None = None,
        freeze_backbone: bool = False,
        aggregation_config: dict[str, any] | None = None,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()
        self.model = sequence_model
        self.aggregation_layer = get_aggregation_layer(**aggregation_config)

        if model_weights is not None:
            self.load_model_weights(model_weights)

        if freeze_backbone:
            for param in sequence_model.parameters():
                param.requires_grad = False

        if self.aggregation_layer.layer_idx < -1:
            self.model.output_hidden_states = True

    def load_model_weights(self, model_weights):
        state_dict = torch.load(model_weights)
        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, seq_features, **kwargs):
        states = self.model(seq_features=seq_features)
        output = self.aggregation_layer(states, seq_features.attention_mask)
        return BaseSequenceOutput(
            last_hidden_state=output,  # batch_size, hidden_size
            router_logits=states.router_logits
            if hasattr(states, "router_logits")
            else None,
        )
