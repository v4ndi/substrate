"""Losses, as injectable modules owned by the pipeline.

See ``docs/guides/losses.md`` for the contract and how to write one.
"""

from avatar.losses.base import CompositeLoss, Loss, LossOutput
from avatar.losses.classification import ClassificationLoss, build_task_loss_fn
from avatar.losses.direct_uplift_loss import DirectUpliftLoss
from avatar.losses.next_k_tokens import HeadPrediction, NextKTokensLoss
from avatar.losses.regularization import L1RegularizationLoss

from .gold_fish import GoldFishLoss
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
    "DirectUpliftLoss",
    "GoldFishLoss",
    "HeadPrediction",
    "KLDLoss",
    "KLDxContrastiveGridLoss",
    "KLDxContrastiveLoss",
    "L1RegularizationLoss",
    "Loss",
    "LossOutput",
    "NextKTokensLoss",
    "ResearchLosses",
    "build_task_loss_fn",
]
