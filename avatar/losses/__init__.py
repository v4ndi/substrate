from avatar.losses.direct_uplift_loss import DirectUpliftLoss
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
    "ContrastiveLoss",
    "DirectUpliftLoss",
    "GoldFishLoss",
    "KLDLoss",
    "KLDxContrastiveGridLoss",
    "KLDxContrastiveLoss",
    "L1RegularizationLoss",
    "ResearchLosses",
]
