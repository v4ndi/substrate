import torch
import torch.nn as nn

from avatar.nn.tabular.base.encoder import BaseTabularEncoder
from avatar.nn.tabular.base.layers import EncoderBlock
from avatar.nn.tabular.utils.masking import build_feature_padding_mask
from avatar.outputs import BaseTabularOutput


class TabularTransformer(BaseTabularEncoder):
    """Set-transformer over feature tokens: ``(B, F, D) -> (B, F, D)``.

    A stack of pre-norm transformer encoder blocks (multi-head self-attention +
    feed-forward). No positional encoding - feature order carries no meaning; the
    per-feature signal comes from the embedding layer.

    Args:
        hidden_size: embedding dim ``D`` (must match the embedding layer).
        num_heads: attention heads per block.
        num_layers: number of stacked encoder blocks.
        attn_dropout: dropout on attention weights and the block FFN.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        attn_dropout: float = 0.15,
    ):
        super().__init__(hidden_size=hidden_size)
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_size, num_heads, attn_dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> BaseTabularOutput:
        """
        Args:
            inputs_embeds: ``(B, F, D)`` feature-token embeddings.
            attention_mask: ``(B, F)``, 1 = keep, 0 = pad. Optional.
            output_hidden_states: also return the per-block hidden states.
        """
        key_padding_mask = build_feature_padding_mask(attention_mask)

        collected: list[torch.FloatTensor] = []
        hidden = inputs_embeds
        for block in self.blocks:
            hidden = block(hidden, key_padding_mask=key_padding_mask)
            if output_hidden_states:
                collected.append(hidden)

        return BaseTabularOutput(
            last_hidden_state=hidden,
            hidden_states=tuple(collected) if output_hidden_states else None,
        )
