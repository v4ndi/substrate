"""Classification directly from an event sequence."""

import torch
import torch.nn as nn

from avatar.data.sequential.batch import EventSequenceBatch
from avatar.losses.base import Loss
from avatar.losses.classification import ClassificationLoss
from avatar.nn.sequential import BaseSequenceModel
from avatar.nn.utils.agg import get_aggregation_layer
from avatar.outputs import SequenceOutput


class SequenceClassification(nn.Module):
    """A sequence classification model with configurable aggregation and fine-tuning options.

    This module combines a sequence model with an aggregation layer and classification head
    to perform sequence classification tasks. Supports:
    - Various sequence aggregation methods (mean, max, etc.)
    - Backbone model freezing with selective unfreezing
    - Pretrained weight loading
    - Customizable dropout and hidden layers

    Args:
        sequence_model: The base sequence model that processes input sequences
        hidden_size (int): Dimensionality of the sequence model's hidden states
        num_classes (int): Number of output classes (default: 2)
        model_weights (str): Path to pretrained weights for the sequence model (default: None)
        dropout_p (float): Dropout probability for regularization (default: 0.15)
        aggregation_layer (dict): Configuration for sequence aggregation layer (default: {"name": "mean"})
        freeze_backbone (bool): Whether to freeze the sequence model parameters (default: False)
        loss (Loss): Loss module. Defaults to cross-entropy over ``num_classes``;
            inject a different one from config to change the objective.

    Example:
        >>> seq_model = BaseSequenceModel(...)
        >>> classifier = SequenceClassification(
        ...     seq_model,
        ...     hidden_size=256,
        ...     num_classes=5,
        ...     freeze_backbone=True,
        ...     unfreeze_params=["layer_norm"]
        ... )
        >>> outputs = classifier(seq_features, targets)
    """

    def __init__(
        self,
        sequence_model: BaseSequenceModel,
        hidden_size: int,
        num_classes: int = 2,
        model_weights: str | None = None,
        dropout_p: float = 0.15,
        aggregation_layer: dict[str, any] | None = None,
        freeze_backbone: bool = False,
        unfreeze_params: list | None = None,
        loss: Loss | None = None,
    ):
        if unfreeze_params is None:
            unfreeze_params = []
        if aggregation_layer is None:
            aggregation_layer = {"name": "mean"}
        super().__init__()
        self.model = sequence_model
        self.aggregation_layer = get_aggregation_layer(**aggregation_layer)
        if self.aggregation_layer.layer_idx < -1:
            self.model.output_hidden_states = True

        # TODO вынести все FFN в отдельный модуль
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout1d(p=dropout_p),
            nn.Linear(hidden_size, num_classes),
        )
        self.loss = (
            loss
            if loss is not None
            else ClassificationLoss(num_classes=num_classes, task_type="classification")
        )

        if model_weights is not None:
            self.load_model_weights(model_weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            for params in unfreeze_params:
                self.unfreeze_module(self.model, params)

    def unfreeze_module(self, model_obj: nn.Module, attr_name: str) -> None:
        """Unfreezes specific parameters in the model by name pattern.

        Args:
            model_obj: The model containing parameters to unfreeze
            attr_name: String pattern to match against parameter names
        """
        for name, params in model_obj.named_parameters():
            if attr_name in name:
                params.requires_grad = True

    def load_model_weights(self, model_weights: str) -> None:
        """Loads pretrained weights for the sequence model.

        Args:
            model_weights: Path to the pretrained weights file

        Raises:
            FileNotFoundError: If the weights file doesn't exist
            RuntimeError: If there's a mismatch in state_dict shapes
        """
        state_dict = torch.load(model_weights)
        self.model.load_state_dict(state_dict, strict=True)

    def forward(
        self, seq_features: EventSequenceBatch, targets: torch.Tensor = None, **kwargs
    ) -> SequenceOutput:
        attention_mask = seq_features.attention_mask
        output = self.model(seq_features=seq_features)
        output = self.aggregation_layer(output, attention_mask)
        logits = self.classification_head(output)

        loss = self.loss(logits, targets).loss if targets is not None else None

        return SequenceOutput(logits=logits, loss=loss)
