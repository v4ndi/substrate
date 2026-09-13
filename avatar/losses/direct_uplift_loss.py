"""Optimise uplift directly rather than the two outcomes."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectUpliftLoss(nn.Module):
    """Optimise the uplift itself rather than the two outcome probabilities."""

    def __init__(self, positive_treat_weight: float = 1.0):
        super().__init__(self)
        self.uplift_loss_weight = torch.tensor()

    def forward(
        self,
        treatment_logits: torch.FloatTensor,
        control_logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor,
    ) -> torch.Tensor:
        z_targets = 4 * is_treat * targets - 2 * targets
        weights = torch.where(targets == 1, self.uplift_loss_weight, 1)
        treatment_positive_probs = F.softmax(treatment_logits, dim=-1)[:, 1]
        control_positive_probs = F.softmax(control_logits, dim=-1)[:, 1]
        uplift_loss = torch.mean(
            weights
            * ((treatment_positive_probs - control_positive_probs) - z_targets) ** 2
        )
        return uplift_loss
