from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LossOutput:
    """What a loss module returns.

    Attributes:
        loss: The scalar the trainer calls ``backward()`` on. ``None`` for
            multi-head losses, where the cross-rank token weighting happens in
            the trainer instead (see
            :func:`avatar.train.loss_reduce.calculate_output_loss`).
        components: The individual terms. When ``num_items`` is set these are
            the per-head losses the trainer reduces; otherwise they are for
            logging only and ``loss`` already contains their sum.
        num_items: Valid item count per head, keyed like ``components``. Its
            presence is what tells the trainer to token-weight across ranks.
    """

    loss: torch.Tensor | None = None
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    num_items: dict[str, torch.Tensor] | None = None


@dataclass
class BaseSequenceOutput:
    """Base output container for sequence models.

    Attributes:
        last_hidden_state: Final hidden states from the model
            Shape: (batch_size, sequence_length, hidden_size)
        hidden_states: Tuple of all hidden states (if output_hidden_states=True)
            Each tensor shape: (batch_size, sequence_length, hidden_size)
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseTabularOutput:
    """Base output container for tabular models.

    Attributes:
        last_hidden_state: Final hidden states from the model
            Shape: (batch_size, num_features, hidden_size)
        hidden_states: Tuple of all hidden states (if output_hidden_states=True)
            Each tensor shape: (batch_size, num_features, hidden_size)
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseClassificationOutput:
    """Base output container for classification tasks.

    Attributes:
        logits: Unnormalized model predictions
            Shape: (batch_size, num_classes)
        loss: Computed loss value (if labels were provided)
            Shape: scalar
    """

    logits: torch.FloatTensor = None
    loss: torch.Tensor = None


@dataclass
class SequenceOutput(BaseSequenceOutput):
    """Extended sequence output with additional task-specific attributes.

    Attributes:
        logits: Model predictions for sequence tasks
            Shape: task-dependent (typically (batch_size, sequence_length, num_classes))
        loss: Computed loss value
            Shape: scalar
        losses: Dictionary of individual loss components
        num_items: Number of valid items in the batch (for proper loss normalization)
        aggregated_hidden_state: Pooled sequence representation
            Shape: (batch_size, hidden_size)
    """

    logits: torch.FloatTensor | None = None
    loss: torch.Tensor | None = None
    losses: dict[str, torch.FloatTensor] | None = None
    num_items: int | None = None
    aggregated_hidden_state: torch.FloatTensor | None = None


@dataclass
class TabularOutput(BaseTabularOutput):
    """Extended tabular output with classification attributes.

    Attributes:
        logits: Model predictions for tabular data
            Shape: (batch_size, num_classes)
        loss: Computed loss value
            Shape: scalar
    """

    loss: torch.Tensor = None
    logits: torch.FloatTensor = None
    auxilary_loss: torch.FloatTensor | None = None


@dataclass
class BaseUpliftOutput:
    """Base output container for uplift modeling tasks.

    Attributes:
        loss: Computed uplift loss value
            Shape: scalar
        conversion: Binary conversion indicators
            Shape: (batch_size,)
        treatment: Binary treatment indicators
            Shape: (batch_size,)
        treatment_probs: Predicted conversion probabilities under treatment
            Shape: (batch_size,)
        control_probs: Predicted conversion probabilities under control
            Shape: (batch_size,)
        uplift: Computed uplift values (treatment_probs - control_probs)
            Shape: (batch_size,)
    """

    loss: torch.FloatTensor = None
    uplift_loss: torch.FloatTensor | None = None
    conversion: torch.LongTensor = None
    treatment: torch.LongTensor = None
    treatment_probs: torch.FloatTensor = None
    control_probs: torch.FloatTensor = None
    uplift: torch.FloatTensor = None


@dataclass
class MultiGroupUpliftOutput(BaseUpliftOutput):
    """Uplift output extended with group information.

    Attributes:
        group: Group identifiers for each sample
            Shape: (batch_size,)
    """

    group: torch.LongTensor | None = None


@dataclass
class MultiTaskxGroupUpliftOutput(MultiGroupUpliftOutput):
    """Uplift output extended with group information.

    Attributes:
        task_name: List of task names
            Shape: (batch_size,)
    """

    task_name: torch.LongTensor | None = None
    task_losses: torch.FloatTensor | None = None


@dataclass
class MultiTaskxGroupResponseOutput(TabularOutput):
    """Uplift output extended with group information.

    Attributes:
        task_name: List of task names
            Shape: (batch_size,)
    """

    conversion: torch.LongTensor = None
    group: torch.LongTensor | None = None
    task_name: torch.LongTensor | None = None
    task_losses: torch.FloatTensor | None = None


@dataclass
class MMoEOutput(MultiTaskxGroupResponseOutput):
    aux: dict[str, Any] = None


@dataclass
class MMoeUpliftOutput(MultiGroupUpliftOutput):
    """Uplift output for Multi-task Mixture-of-Experts models.

    Attributes:
        task_name: List of task names
        logits: Raw model outputs before sigmoid
            Shape: (batch_size, num_tasks)
        task_gated_weights: Gating weights for each task
            Shape: (batch_size, num_tasks, num_experts)
    """

    task_name: list[str] | None = None
    logits: torch.FloatTensor | None = None
    task_gated_weights: torch.FloatTensor | None = None
