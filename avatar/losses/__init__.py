"""Losses, as injectable modules owned by the pipeline.

See ``docs/guides/losses.md`` for the contract and how to write one.
"""

from avatar.losses.base import CompositeLoss, Loss, LossOutput
from avatar.losses.classification import ClassificationLoss, build_task_loss_fn
from avatar.losses.regularization import L1RegularizationLoss

from .kld_loss import (
    ContrastiveLoss,
    KLDLoss,
    KLDxContrastiveGridLoss,
    KLDxContrastiveLoss,
    ResearchLosses,
)

__all__ = [
    "ClassificationLoss",
    "CompositeLoss",
    "ContrastiveLoss",
    "KLDLoss",
    "KLDxContrastiveGridLoss",
    "KLDxContrastiveLoss",
    "L1RegularizationLoss",
    "Loss",
    "LossOutput",
    "ResearchLosses",
    "build_task_loss_fn",
]
