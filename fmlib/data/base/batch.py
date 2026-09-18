"""Helpers shared by every batch container."""

from typing import Any

import torch

__all__ = ["move_to_device"]


def move_to_device(
    data: torch.Tensor | Any, device: torch.device
) -> torch.Tensor | Any | None:
    """Safely moves data to the specified torch device while handling None values.

    Args:
        data: Input data to move. Can be:
            - A torch.Tensor (will be moved to device)
            - A dict of tensors (each value is moved)
            - None (returns None)
        device: Target device (e.g., 'cuda:0' or torch.device('cpu'))
    """
    if data is not None:
        if isinstance(data, dict):
            return {key: val.to(device) for key, val in data.items()}

        return data.to(device)
    else:
        return None
