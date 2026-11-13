"""Output dataclasses: what each pipeline returns.

The trainer only ever reads ``loss`` (or ``losses`` plus ``num_items`` for
multi-head models); everything else exists so that metrics can read what a
particular task produced. That is why a metric is bound to a pipeline: it
requires the fields of the output that pipeline returns.
"""

from dataclasses import dataclass, field

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

    Returned by the sequence encoders under :mod:`avatar.nn.sequential`. No
    pipeline consumes it today — the sequence pipelines were removed — but the
    encoders are still built and tested, and this is their return type.

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
