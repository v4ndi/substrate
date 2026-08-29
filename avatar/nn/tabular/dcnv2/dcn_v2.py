import torch
import torch.nn as nn

from avatar.nn.utils import FeedForwardNetwork


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(
            nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)
        )

    def forward(self, X_0):
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i


class DCNv2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        output_dim: int | None = None,
        dropout_p: float = 0.15,
    ):
        super().__init__()
        self.cross_net = CrossNetV2(input_dim=hidden_dim, num_layers=num_layers)
        self.parallel_dnn = nn.Sequential(*[
            FeedForwardNetwork(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout_p=dropout_p,
                need_output_linear=False,
            )
            for _ in range(num_layers)
        ])

        self.output_ffn = FeedForwardNetwork(
            input_dim=hidden_dim * 2,
            output_dim=output_dim if output_dim is not None else hidden_dim,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
            need_output_linear=output_dim is None,
        )

    def forward(self, input: torch.Tensor):
        parallel_output = torch.cat(
            [self.cross_net(input), self.parallel_dnn(input)], dim=-1
        )
        output = self.output_ffn(parallel_output)
        return output
