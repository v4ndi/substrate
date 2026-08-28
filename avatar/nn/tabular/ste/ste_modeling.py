from typing import Optional, Union

import torch
from torch import nn

from avatar.data.tabular_batch import TabularBatch
from avatar.nn.embedding import BaseEmbedding
from avatar.nn.tabular.base_tabular import BaseTabularBackbone, BaseTabularEncoder
from avatar.nn.tabular.moe.moe_modeling import GateTopK, MoE
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import BaseTabularOutput


class SublayerConnection(nn.Module):
    """Residual connection with layer norm and dropout.

    Implements the standard "Add & Norm" operation used in transformers:
    output = x + Dropout(Sublayer(LayerNorm(x)))

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
        >>> # Any sublayer (attention, FFN, etc.)
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
        key_padding_mask: Optional[torch.Tensor] = None,
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


class CrossAttentionEncoderBlock(EncoderBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, key, value, query, key_padding_mask=None, need_weights=False):
        x = self.input_sublayer(
            query,
            lambda _q: self.multi_head_attn(
                key=key,
                value=value,
                query=_q,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0],
        )
        x = self.output_sublayer(x, self.ffn)
        return self.dropout(x)


class STEv2Block(BaseTabularEncoder):
    """STEv2 block for tabular data processing.

    A stack of transformer encoder blocks designed for processing tabular/feature-based data.
    Each block consists of multi-head self-attention and feed-forward layers with residual connections.

    Args:
        hidden_size: Dimensionality of input and output features
        num_heads: Number of attention heads in each encoder block
        num_layers: Number of stacked encoder blocks
        attn_dropout: Dropout probability for attention weights. Default: 0.15
        need_weights: Whether to return attention weights from all layers. Default: False
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        attn_dropout: float = 0.15,
        need_weights: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(self.hidden_size, num_heads, attn_dropout)
            for _ in range(num_layers)
        ])
        self.need_weights = need_weights

    def forward(
        self,
        input_embeds: torch.FloatTensor,
        key_padding_mask=None,
    ):
        """
        Args:
            input_embeds: torch.FloatTensor - encoded features, (batch_size, n_features, emb_dim)
            key_padding_mask (_type_, optional): padding mask for attention. Defaults to None.
            need_weights (bool, optional): return attn_scores or not. Defaults to False.
        Returns:
            out: Tensor - hidden_states from encoder blocks
                (batch_size, n_features, emb_dim)
        """
        out = input_embeds

        for encoder in self.encoder_blocks:
            out = encoder.forward(
                x=out, key_padding_mask=key_padding_mask, need_weights=self.need_weights
            )
        return out


class STEv2(BaseTabularBackbone):
    """STEv2 (Spatio-Temporal Encoder v2) model for tabular data processing.

    A transformer-based architecture for tabular data that:
    1. Encodes input features using the specified embedding layer
    2. Processes them through stacked transformer encoder blocks
    3. Returns the encoded representations

    Args:
        embedding: Feature embedding layer (subclass of BaseEmbedding)
        num_heads: Number of attention heads in each encoder block
        num_layers: Number of stacked encoder blocks
        attn_dropout: Dropout probability for attention weights. Default: 0.15
        need_weights: Whether to return attention weights from all layers. Default: False
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        num_heads: int,
        num_layers: int,
        attn_dropout: float = 0.15,
        need_weights: bool = False,
    ):
        super().__init__(embedding=embedding)
        self.hidden_size = embedding.hidden_size
        self.output_dim = (
            self.hidden_size
        )  # for compatibility with avatar.pipeline.tabular.TabularClassification
        self.stev2_block = STEv2Block(
            hidden_size=self.hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            need_weights=need_weights,
        )

    def forward(self, tab_features: Union[TabularBatch, torch.FloatTensor]):
        """
        Args:
            tab_features: TabularBatch
            key_padding_mask Optional: padding mask for attention. Defaults to None.
        Returns:
            avatar.ouptuts.BaseTabularOutput
        """
        if isinstance(tab_features, torch.Tensor):
            out = tab_features
        else:
            out = self.embedding(tab_features)

        out = self.stev2_block(out)

        return BaseTabularOutput(
            last_hidden_state=out,
            hidden_states=None,
        )


# TODO Влад должен написать docstring + example
class MoEEncoderBlock(nn.Module):
    """Moe Encoder Block.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_experts: int,
        num_active_experts: int,
        attn_dropout: float = 0.15,
        aggregation_config: dict[str, any] = {"name": "mean"},
    ):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        if num_active_experts == -1 or num_active_experts > num_experts:
            num_active_experts = num_experts

        gate = GateTopK(
            num_experts=num_experts,
            hidden_size=emb_dim,
            num_active_experts=num_active_experts,
            aggregation_config=aggregation_config,
        )
        experts = [FeedForwardNetwork(emb_dim, emb_dim * 4) for _ in range(num_experts)]
        self.moe = MoE(experts, gate, hidden_size=emb_dim)

        self.input_sublayer = SublayerConnection(emb_dim)
        self.output_sublayer = SublayerConnection(emb_dim)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, key_padding_mask=None, need_weights=False):
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
        x = self.output_sublayer(x, lambda x: self.moe(x).logits)
        return self.dropout(x)


class MoESTEv2(STEv2):
    """
    Args:
        embedding: nn.Module - layer for encoding tabular features to embedings,
        num_heads: int, - number of attention heads
        num_layers: int - number of encoder layers
        num_experts: int - num of experts in MoE layer
        num_active_experts: int - num of experts, applied for specific input element
        aggregation_config: dict - for aggregation in MoE gate layer
        attn_dropout: float = 0.15 - attention dropout
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        num_heads: int,
        num_layers: int,
        num_experts: int,
        num_active_experts: int,
        aggregation_config: dict[str, any] = {"name": "mean"},
        attn_dropout: float = 0.15,
        return_tensor=False,
    ):
        super().__init__(
            embedding=embedding,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
        )
        self.return_tensor = return_tensor
        self.emb = embedding
        self.hidden_size = self.emb.hidden_size
        del self.encoder_blocks
        self.encoder_blocks = nn.ModuleList([
            MoEEncoderBlock(
                self.hidden_size,
                num_heads,
                num_experts,
                num_active_experts,
                attn_dropout,
                aggregation_config,
            )
            for _ in range(num_layers)
        ])

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if self.return_tensor and hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        return out
