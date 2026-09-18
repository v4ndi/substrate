"""A worked example of a custom :class:`~fmlib.losses.Loss`.

Focal loss for imbalanced binary classification: down-weights examples the
model already gets right, so the gradient keeps coming from the hard ones.

What it demonstrates is the contract — returning a
:class:`~fmlib.outputs.LossOutput` — and the one thing that is *not* part of
it: the argument list. That is the pipeline's business, so a replacement loss
must accept what the pipeline it plugs into passes. Here that is
``(logits, targets, model=None)``, matching
:class:`~fmlib.losses.ClassificationLoss`, which is what
:class:`~fmlib.pipeline.tabular.SupervisedLearner` calls.

Inject it from a config::

    model:
      _target_: fmlib.pipeline.tabular.SupervisedLearner
      num_classes: 2
      loss:
        _target_: examples.custom_loss.loss.FocalLoss
        gamma: 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fmlib.losses import Loss, LossOutput


class FocalLoss(Loss):
    """Cross-entropy scaled by ``(1 - p_t) ** gamma``.

    Args:
        gamma: Focusing strength. 0 is plain cross-entropy; the usual choice
            is 2.
        alpha: Optional per-class weights, passed to the underlying
            cross-entropy.

    Returns:
        :class:`~fmlib.outputs.LossOutput` whose ``loss`` is the focal term
        and whose ``components`` carry the unweighted cross-entropy, so both
        are visible in the logs and it is obvious what the focusing did.
    """

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        weight = None if alpha is None else torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> LossOutput:
        """Score ``logits`` against ``targets``.

        ``model`` is accepted and ignored: the pipeline passes it so that a
        loss can regularise the encoder's weights, and accepting it keeps this
        loss droppable anywhere ClassificationLoss goes.
        """
        cross_entropy = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        probability = torch.exp(-cross_entropy)
        focal = ((1.0 - probability) ** self.gamma * cross_entropy).mean()
        return LossOutput(
            loss=focal,
            components={"cross_entropy": cross_entropy.mean().detach()},
        )
