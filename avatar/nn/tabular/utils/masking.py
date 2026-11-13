"""Padding masks for feature tokens."""

import torch


def build_feature_padding_mask(
    attention_mask: torch.LongTensor | None,
) -> torch.BoolTensor | None:
    """Convert a keep-mask into a ``nn.MultiheadAttention`` key-padding mask.

    Args:
        attention_mask: ``(B, F)`` tensor, 1 = keep the feature token, 0 = pad.
            ``None`` means "no padding".

    Returns:
        ``(B, F)`` bool tensor where ``True`` marks positions to **ignore**
        (the convention ``nn.MultiheadAttention`` expects), or ``None``.
    """
    if attention_mask is None:
        return None
    return ~attention_mask.bool()
