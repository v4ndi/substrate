"""Multi-head next-k-token loss.

Owns the label shifting, the criteria and the per-head weighting for
:class:`~avatar.pipeline.sequence.next_k_tokens.NextKTokensPrediction`. The
prediction heads stay on the pipeline — they carry parameters, so they are part
of the model — and hand their logits here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from avatar.losses.base import Loss
from avatar.outputs import LossOutput

__all__ = ["HeadPrediction", "NextKTokensLoss"]


@dataclass
class HeadPrediction:
    """One prediction head's contribution at one horizon.

    Attributes:
        logits: Head output over the full sequence, before shifting.
        input_ids: The values to predict, before shifting.
        n_classes: 1 for a numeric head, >1 for a categorical one.
        attention_mask: Which positions count; numeric heads only.
        pad_token: Label value masked out of a categorical head.
        scale_by_coef_in_eval: Whether the horizon coefficient applies outside
            training too. True only for the timedelta head, which has always
            behaved that way.
    """

    logits: torch.Tensor
    input_ids: torch.Tensor
    n_classes: int
    attention_mask: torch.Tensor | None = None
    pad_token: int = 0
    scale_by_coef_in_eval: bool = False


class NextKTokensLoss(Loss):
    """Per-head, per-horizon losses with their valid-item counts.

    Returns a :class:`~avatar.outputs.LossOutput` with ``loss=None``: the terms
    are token-weighted across ranks by the trainer, which is the only place
    that knows the global item counts.

    Args:
        horizon: Number of future steps predicted.
        feature_loss_weights: Per-feature multipliers, applied in training only.
        horizion_loss_weight: Exponent for the per-horizon discount
            ``(idx + 1) ** horizion_loss_weight``; further horizons weigh less.
        numeric_loss: Criterion for numeric heads. Must not reduce, since the
            attention mask is applied to the elementwise result.
        categorical_loss: Criterion for categorical heads.
    """

    def __init__(
        self,
        horizon: int,
        feature_loss_weights: dict[str, float] | None = None,
        horizion_loss_weight: float = 0.0,
        numeric_loss: nn.Module | None = None,
        categorical_loss: nn.Module | None = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.feature_loss_weights = dict(feature_loss_weights or {})
        self.coefs = [(coef + 1) ** horizion_loss_weight for coef in range(horizon)]
        self.numeric_loss = (
            numeric_loss if numeric_loss is not None else nn.L1Loss(reduction="none")
        )
        self.categorical_loss = (
            categorical_loss
            if categorical_loss is not None
            else nn.CrossEntropyLoss(reduction="sum")
        )

    @staticmethod
    def create_labels(
        input_ids, logits, horizon_offset, n_classes: int, pad_token: int = 0
    ):
        """Shift labels forward and logits back so position i predicts i+offset.

        ``n_classes == 1`` -> numeric, else categorical.
        """
        with torch.no_grad():
            labels = input_ids.detach().clone()
            if n_classes > 1:
                pad_tokens_mask = labels == pad_token
                labels[pad_tokens_mask] = -100
        shifted_labels = labels[:, horizon_offset:].contiguous()
        shifted_logits = logits[:, :-horizon_offset, :].contiguous()

        return shifted_logits, shifted_labels

    def calculate_loss(
        self,
        shifted_logits,
        shifted_labels,
        n_classes,
        attention_mask=None,
        horizon_offset=None,
    ):
        """Return the head's summed loss and its count of valid items.

        ``n_classes == 1`` -> numeric, else categorical.
        """
        assert shifted_logits.shape[1] == shifted_labels.shape[1]
        if n_classes == 1:
            shifted_logits = shifted_logits.squeeze(dim=-1)
            loss = self.numeric_loss(shifted_logits, shifted_labels)

            num_items = attention_mask[:, horizon_offset:]
            loss *= attention_mask[:, horizon_offset:]

            return loss.sum(), num_items.sum()
        elif n_classes > 1:
            loss = self.categorical_loss(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
            )
            num_items = (shifted_labels != -100).sum()

            return loss, num_items
        else:
            raise ValueError()

    def head_loss(
        self, head: HeadPrediction, horizon_offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted_logits, shifted_labels = self.create_labels(
            input_ids=head.input_ids,
            logits=head.logits,
            horizon_offset=horizon_offset,
            n_classes=head.n_classes,
            pad_token=head.pad_token,
        )
        return self.calculate_loss(
            shifted_logits=shifted_logits,
            shifted_labels=shifted_labels,
            n_classes=head.n_classes,
            attention_mask=head.attention_mask,
            horizon_offset=horizon_offset,
        )

    def forward(self, horizons: list[dict[str, HeadPrediction]]) -> LossOutput:
        """Score every head at every horizon.

        Args:
            horizons: One mapping of head name to prediction per horizon, in
                horizon order; ``horizons[i]`` is offset ``i + 1``.
        """
        components: dict[str, torch.Tensor] = {}
        num_items: dict[str, torch.Tensor] = {}

        for horizon_idx, heads in enumerate(horizons):
            horizon_offset = horizon_idx + 1
            coef = self.coefs[horizon_idx]
            for name, head in heads.items():
                loss, head_num_items = self.head_loss(head, horizon_offset)
                if self.training:
                    loss = loss * self.feature_loss_weights.get(name, 1.0)
                    if not head.scale_by_coef_in_eval:
                        loss = loss / coef
                if head.scale_by_coef_in_eval:
                    # The timedelta head has always divided by the horizon
                    # coefficient in eval as well; kept as-is deliberately.
                    loss = loss / coef
                key = f"{name}_head_{horizon_idx}"
                components[key] = loss
                num_items[key] = head_num_items

        return LossOutput(loss=None, components=components, num_items=num_items)
