import torch  # noqa: F401
import torch.nn.functional as F
from transformers import PreTrainedModel

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.sequential.event_encoder.base import BaseEventEncoder
from avatar.nn.sequential.model.base import BaseSequenceModel
from avatar.outputs import BaseSequenceOutput


class TransformersWrapper(BaseSequenceModel):
    """
    Wrapper for transformer backbones also compatibility with Huggingface's transformers.

    Args:
        event_encoder (avatar.nn.sequential.BaseEventEncoder): The event encoder.
        backbone (transformers.PreTrainedModel): The backbone model.
        output_hidden_states (bool): Whether to output hidden states. Defaults to False.
    """

    def __init__(
        self,
        event_encoder: BaseEventEncoder,
        backbone: PreTrainedModel,
        output_hidden_states: bool = False,
    ):
        super().__init__(event_encoder=event_encoder, backbone=backbone)
        self.output_hidden_states = output_hidden_states

    def forward(self, seq_features: EventSequenceBatch):
        """
        Forward pass of the transformers wrapper.

        Args:
            seq_features: EventSequenceBatch. The sequence features.
        Returns:
            BaseSequenceOutput: The output of the sequence model.
        """
        inputs_embeds = self.event_encoder(seq_features)
        attention_mask = seq_features.attention_mask
        if attention_mask.shape[:2] != inputs_embeds.shape[:2]:
            attention_mask = F.pad(
                attention_mask,
                (inputs_embeds.shape[1] - attention_mask.shape[1], 0),
                mode="constant",
                value=1,
            )
        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
        )

        last_hidden_state = output.last_hidden_state
        if seq_features.attention_mask.shape[:2] != last_hidden_state.shape[:2]:
            last_hidden_state = last_hidden_state[
                :, -seq_features.attention_mask.shape[1] :, :
            ]  # remove meta tokens in beginning
        return BaseSequenceOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=output.hidden_states,
            router_logits=output.router_logits
            if hasattr(output, "router_logits")
            else None,
        )
