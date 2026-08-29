from transformers import AutoConfig, AutoModel

from avatar.nn.sequence.base_sequence_model import BaseSequenceBackbone


class ModelPruningWrapper(BaseSequenceBackbone):
    """A wrapper class for model pruning operations. It allows creating a pruned version of a pre-trained model based on a given configuration.

    Args:
        path_to_model (str): Path to the pre-trained model.
        pruning_config (dict): Configuration for pruning the model.
        original_weight (bool): Whether to use the original weights for pruning.
    """

    def __init__(
        self,
        path_to_model: str,
        pruning_config: dict | None = None,
        original_weight: bool = True,
    ):
        """Initializes the ModelPruningWrapper with a pre-trained model and optional pruning configuration.

        Raises:
            ValueError: If the pruning_config contains keys that are not present in the model's configuration.
        """
        super().__init__()
        original_model = AutoModel.from_pretrained(path_to_model)
        if pruning_config is not None:
            new_config = AutoConfig.from_pretrained(path_to_model)
            for key, value in pruning_config.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
                else:
                    raise ValueError(f"Source model does not have attribute {key}")

            self.pruned_model = AutoModel.from_config(new_config)
            if original_weight:
                self.weght_setter(self.pruned_model, original_model)
        else:
            self.pruned_model = original_model

    @staticmethod
    def weght_setter(new_model, old_model):
        """Sets the weights of the pruned model using the weights from the original model.

        This method recursively copies weights from the original model to the pruned model.

        Args:
            new_model: The pruned model to which weights will be copied.
            old_model: The original model from which weights will be copied.
        """
        if any(new_model.children()):
            for name, child in new_model.named_children():
                ModelPruningWrapper.weght_setter(child, getattr(old_model, name))
        else:
            if hasattr(new_model, "weight"):
                dim = len(new_model.weight.shape)
                if dim == 1:
                    dim_0 = new_model.weight.size(0)
                    new_model.weight.data = old_model.weight.data[:dim_0]
                elif dim == 2:
                    dim_0, dim_1 = new_model.weight.size(0), new_model.weight.size(1)
                    new_model.weight.data = old_model.weight.data[:dim_0, :dim_1]

    def forward(self, inputs_embeds, attention_mask, output_hidden_states):
        return self.pruned_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
