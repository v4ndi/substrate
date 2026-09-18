"""Transformer building blocks for the tabular encoder."""

import torch
import torch.nn as nn

from fmlib.nn.utils.ffn import FeedForwardNetwork


class SublayerConnection(nn.Module):
    """Residual connection with layer norm and dropout.

    Implements the standard "Add & Norm" operation used in transformers:
    ``output = x + Dropout(Sublayer(LayerNorm(x)))``

    Args:
        size (int): Dimensionality of input features
        dropout (float): Dropout probability. Default: 0.15

    Shape:
        - Input x: (..., size)
        - Sublayer output: (..., size)
        - Output: (..., size) (same as input)

    Example:
        >>> connection = SublayerConnection(size=512)
        >>> x = torch.randn(32, 10, 512)
        >>> sublayer = nn.Linear(512, 512)
        >>> out = connection(x, sublayer)  # shape (32, 10, 512)
    """

    def __init__(self, size: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, sublayer: nn.Module) -> torch.FloatTensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """A single transformer encoder block with multi-head attention and feed-forward network.

    This block implements the standard transformer encoder architecture with:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    3. Dropout for regularization

    Args:
        emb_dim: Embedding dimension of input features
        num_heads: Number of attention heads
        attn_dropout: Dropout probability for attention weights. Default: 0.15
        ffn_dropout: Dropout probability for feed-forward network. Default: 0.15
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        attn_dropout: float = 0.15,
        ffn_dropout: float = 0.15,
    ):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ffn = FeedForwardNetwork(
            input_dim=emb_dim,
            hidden_dim=emb_dim * 4,
            output_dim=emb_dim,
            dropout_p=ffn_dropout,
            activation=nn.SELU,
            normalization=nn.LayerNorm,
        )
        self.input_sublayer = SublayerConnection(emb_dim)
        self.output_sublayer = SublayerConnection(emb_dim)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        x = self.input_sublayer(
            x,
            lambda _x: self.multi_head_attn(
                key=_x,
                value=_x,
                query=_x,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )[0],
        )
        x = self.output_sublayer(x, self.ffn)
        return self.dropout(x)
