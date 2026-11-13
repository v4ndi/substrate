"""L1 penalty over selected parameters."""

import torch
import torch.nn as nn


class L1RegularizationLoss:
    """L1 penalty over parameters whose name contains a substring.

    Args:
        apply_substr: Only parameters whose qualified name contains this are
            penalised; the default targets embedding tables.
    """

    def __init__(self, apply_substr: str = ""):
        self.apply_substr = apply_substr

    def __call__(self, module: nn.Module):
        assert isinstance(module, nn.Module), (
            "L1 regularization loss can only be applied to a model (instance of nn.Module)"
        )
        loss = 0
        for name, param in module.named_parameters():
            if name.find(self.apply_substr) == -1:
                continue
            loss = loss + torch.norm(param, 1)
        return loss
