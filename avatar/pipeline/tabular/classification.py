"""Classification / regression head over a tabular representation."""

import torch
import torch.nn as nn

from avatar.losses.base import Loss
from avatar.losses.classification import ClassificationLoss
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import TabularOutput
from avatar.pipeline.tabular.tabular_aggregation import TabularWithAggregatedStates


class TabularClassification(nn.Module):
    """A neural network for tabular data classification or regression.

    This module combines a tabular feature encoder with an output head to perform either:
    - Multi-class classification
    - Binary classification
    - Regression

    The architecture consists of:
    1. A tabular feature encoder (provided)
    2. An output head with:
       - Dropout regularization
       - Linear layer + SELU activation
       - Batch normalization
       - Final linear projection

    Args:
        tabular_model: The tabular feature encoder model that produces embeddings
        num_classes: Number of output classes (1 for regression/binary classification)
        dropout_p: Dropout probability for regularization (default: 0.2)
        task_type: Type of task, either 'classification' or 'regression'

    Raises:
        ValueError: If invalid combination of num_classes and task_type is provided

    Example:
        >>> encoder = TabularWithAggregatedStates(...)
        >>> model = TabularClassification(encoder, num_classes=3)
        >>> outputs = model(tab_features, targets=labels)
    """

    def __init__(
        self,
        num_classes: int,
        tabular_model: TabularWithAggregatedStates = None,
        dropout_p: float = 0.2,
        task_type: str = "classification",
        out_head_hidden_dim: int = 256,
        extra_hidden_dim: int = 0,
        hidden_proj_dim: int | None = None,
        output_head=None,
        l1_loss_weight: float = 0.0,
        loss: Loss | None = None,
    ):
        super().__init__()
        self.encoder = tabular_model
        self.extra_hidden_dim = extra_hidden_dim
        self._init_hidden_proj(
            extra_hidden_dim=extra_hidden_dim,
            hidden_proj_dim=hidden_proj_dim,
            dropout_p=dropout_p,
        )

        self._init_out_head(
            extra_hidden_dim=extra_hidden_dim,
            hidden_proj_dim=hidden_proj_dim,
            out_head_hidden_dim=out_head_hidden_dim,
            num_classes=num_classes,
            dropout_p=dropout_p,
            output_head=output_head,
        )
        self.l1_loss_weight = l1_loss_weight
        self.loss = (
            loss
            if loss is not None
            else ClassificationLoss(
                num_classes=num_classes,
                task_type=task_type,
                l1_weight=l1_loss_weight,
            )
        )

    def _init_out_head(
        self,
        extra_hidden_dim: int,
        hidden_proj_dim: int,
        out_head_hidden_dim: int,
        num_classes: int,
        dropout_p: float,
        output_head=None,
    ):
        if output_head is not None:
            self.out_head = output_head
            return
        head_input_dim = 0
        if self.encoder is not None:
            head_input_dim += self.encoder.output_dim
        if hidden_proj_dim is not None:
            head_input_dim += hidden_proj_dim
        else:
            head_input_dim += extra_hidden_dim

        self.out_head = FeedForwardNetwork(
            input_dim=head_input_dim,
            hidden_dim=out_head_hidden_dim,
            output_dim=num_classes,
            activation=nn.SELU,
            normalization=nn.BatchNorm1d,
            dropout_p=dropout_p,
        )

    def _init_hidden_proj(
        self, extra_hidden_dim: int, hidden_proj_dim: int, dropout_p: float
    ):
        if hidden_proj_dim is None:
            if extra_hidden_dim > 0:
                self.proj = nn.LayerNorm(extra_hidden_dim)
            else:
                self.proj = None
            return
        self.proj = nn.Sequential(
            nn.LayerNorm(extra_hidden_dim),
            FeedForwardNetwork(
                input_dim=extra_hidden_dim,
                hidden_dim=hidden_proj_dim,
                output_dim=hidden_proj_dim,
                dropout_p=dropout_p,
                normalization=nn.LayerNorm,
                dropout_first=True,
                end_normalization=self.extra_hidden_dim > 0
                and self.encoder is not None,
                need_output_linear=self.extra_hidden_dim > 0
                and self.encoder is not None,
            ),
        )

    def forward(self, tab_features, targets=None, **kwargs):
        # ``TabularBatch.hidden_states`` is a dict of column name -> tensor, or
        # None when no hidden-state column was configured. Concatenating the
        # values in insertion order is the convention SLearner and
        # SupervisedLearner already use.
        hidden_states = tab_features.hidden_states
        if hidden_states is not None:
            hidden_states = torch.cat(tuple(hidden_states.values()), dim=1)
            hidden_states = torch.where(hidden_states.isnan(), 0, hidden_states)
        output = (
            self.encoder(tab_features) if self.encoder is not None else hidden_states
        )

        if self.extra_hidden_dim > 0 and self.encoder is not None:
            output = torch.cat(
                [output, self.proj(hidden_states)], dim=-1
            )  # late fusion
        elif self.extra_hidden_dim > 0:
            output = self.proj(output)
        logits = self.out_head(output)

        loss = None
        l1_loss = None

        if targets is not None:
            result = self.loss(logits, targets, model=self.encoder)
            loss = result.loss
            l1_loss = result.components.get("l1", 0)

        return TabularOutput(logits=logits, loss=loss, auxilary_loss=l1_loss)
