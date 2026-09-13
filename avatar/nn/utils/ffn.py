"""The configurable feed-forward block used by every head."""

import torch
from torch import nn


def init_activation(activation: str) -> nn.Module:
    """Resolve an activation by name.

    Raises:
        ValueError: The name is not a known activation.
    """
    if activation == "relu":
        return nn.ReLU
    if activation == "selu":
        return nn.SELU
    elif activation == "silu":
        return nn.SiLU
    elif activation == "gelu":
        return nn.GELU
    raise ValueError(f"Unknown activation function name: {activation}")


def init_normalization(normalization: str) -> nn.Module:
    """Resolve a normalisation layer by name.

    Raises:
        ValueError: The name is not a known normalisation.
    """
    assert normalization in ["layer_norm", "batch_norm"], (
        "Only layer_norm and batch_norm are supported"
    )
    if normalization == "layer_norm":
        return nn.LayerNorm
    elif normalization == "batch_norm":
        return nn.BatchNorm1d

    raise ValueError(f"Unknown normalization name: {normalization}")


class FeedForwardNetwork(nn.Module):
    """The configurable feed-forward block used by every head.

    Order of dropout, activation and normalisation is configurable because the
    heads in this repository disagree about it, and both orders are in use.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout_p: int = 0.15,
        activation: str | nn.Module | None = nn.SELU,
        normalization: str | nn.Module | None = nn.LayerNorm,
        end_normalization: bool = False,
        embedding_processor=None,
        dropout_first: bool = False,
        activation_before_normalization: bool = False,
        need_output_linear: bool = True,
    ):
        # ``None`` means "same as input_dim" and is resolved a few lines below,
        # so it has to be tested before the comparison, not after it: the old
        # order raised TypeError on the very value the docstring allows.
        assert input_dim > 0, "Input dimension must be greater than 0"
        assert output_dim is None or output_dim > 0, (
            "Output dimension must be greater than 0"
        )
        assert hidden_dim is None or hidden_dim > 0, (
            "Hidden dimension must be greater than 0"
        )
        assert need_output_linear or not end_normalization, (
            "You provide 2 normalizations sequentially, if need_output_linear is False, you need to provide end_normalization=False"
        )
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.end_normalization = end_normalization
        self.embedding_processor = embedding_processor

        if isinstance(normalization, str):
            normalization = init_normalization(normalization)

        if isinstance(activation, str):
            activation = init_activation(activation)

        blocks = []
        if dropout_first:
            blocks.append(nn.Dropout(dropout_p))
        blocks.append(nn.Linear(input_dim, hidden_dim))
        if activation_before_normalization:
            blocks.append(activation())
            blocks.append(normalization(hidden_dim))
        else:
            blocks.append(normalization(hidden_dim))
            blocks.append(activation())
        if not dropout_first:
            blocks.append(nn.Dropout(dropout_p))
        if need_output_linear:
            blocks.append(nn.Linear(hidden_dim, output_dim))
        if self.end_normalization:
            blocks.append(normalization(output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        need_embed_processor_flag = (
            self.embedding_processor is not None and targets is not None
        )
        assert need_embed_processor_flag or self.embedding_processor is None, (
            "If embedding processor is not None, targets must be provided"
        )

        if need_embed_processor_flag and self.training:
            x, targets = self.embedding_processor.forward(x, targets)

        x = self.net(x)

        if need_embed_processor_flag and self.training:
            return x, targets
        return x
