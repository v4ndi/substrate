# TODO Написать тесты на основные модули, добавить docstring на каждый модуль

import torch
import torch.nn.functional as F
from torch import nn

from avatar.nn.utils import get_aggregation_layer
from avatar.outputs import MoeTabularOutput


class FFN(nn.Module):
    def __init__(self, dropout_p, hidden_size, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, input):
        return self.model(input)


class UniversalGate(nn.Module):
    """
    Find weights for each expert, according to the input
    """

    def __init__(
        self,
        backbone: nn.Module,
        dropout_p: float,
        hidden_size: int,
        num_experts: int,
        aggregation_config: dict[str, any] | None = None,
    ):
        """
        backbone: nn.Module: [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size]
        """
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()
        self.backbone = backbone
        self.aggregateion_layer = get_aggregation_layer(**aggregation_config)
        self.softmax = nn.Softmax(dim=-1)
        self.ffn = FFN(
            dropout_p=dropout_p, hidden_size=hidden_size, output_dim=num_experts
        )

    def forward(self, input):
        out = self.backbone(input)
        out = self.aggregateion_layer(out)
        out = self.ffn(out)
        out = self.softmax(out)
        return out


class GateTopK(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        num_active_experts: int = 2,
        dropout_p: float = 0.15,
        aggregation_config: dict[str, any] | None = None,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()
        self.num_active_experts = num_active_experts
        self.aggregateion_layer = get_aggregation_layer(**aggregation_config)
        self.proj_layer = nn.Linear(hidden_size, hidden_size)
        self.noise_layer = nn.Linear(hidden_size, num_experts, bias=False)
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, num_experts),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.proj_layer(input)
        x = self.aggregateion_layer(x)

        noise_encoded_part = self.noise_layer(x)
        noise_part = torch.randn_like(noise_encoded_part) * F.softplus(
            noise_encoded_part
        )
        gate_part = self.gate_layer(x)

        out = noise_part + gate_part
        kth_stat = torch.topk(out, k=self.num_active_experts, dim=-1)[0][:, -1]
        out = torch.where(
            out >= kth_stat.unsqueeze(-1), out, -torch.inf * torch.ones_like(out)
        )

        return self.softmax(out)


class MLPGate(UniversalGate):
    """
    Classical mlp gate
    """

    def __init__(
        self,
        num_experts: int,
        dropout_p: float,
        hidden_size: int,
        scale=4,
        aggregation_config: dict[str, any] | None = None,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        inner_hidden = hidden_size * scale
        mlp = nn.Sequential(
            nn.Linear(hidden_size, inner_hidden),
            nn.LayerNorm(inner_hidden),
            nn.SELU(),
            nn.Dropout(dropout_p),
            nn.Linear(inner_hidden, hidden_size),
        )
        super().__init__(
            backbone=mlp,
            dropout_p=dropout_p,
            hidden_size=hidden_size,
            num_experts=num_experts,
            aggregation_config=aggregation_config,
        )


class MoE(nn.Module):
    def __init__(
        self,
        experts: list[nn.Module],
        gate: nn.Module,
        hidden_size=None,
        embedding=None,
    ):
        """
        experts: list of experts (expert - module: [bs, seq_len, emb_dim] -> [bs, seq_len, emb_dim])
        gate: module: [bs, seq_len, emb_dim] -> [bs, num_experts]
        hidden_size: int (DEPRECATED)  TODO: remove
        embedding: embed module
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.embedding = embedding

    def _to_orig_positions(self, positions, batch_size, data):
        """
        restore original_data from data, there data is original_data[positions]

        Args:
            positions, there data := original_data[positions]
            batch_size: len(original_data)
            data := original_data[positions]

        return: partial_orig_data, where partial_orig_data[positions] = data
            and partial_orig_data[~positions] = 0

        """
        orig_position_one_hot = F.one_hot(positions, num_classes=batch_size)
        orig_position_one_hot = orig_position_one_hot.to(data.device).to(data.dtype).T
        return torch.einsum("bi,i...->b...", orig_position_one_hot, data)

    def _get_not_null_expert_positions(self, weights, expert_idx):
        """
        weigts: [bs, num_experts]
        return: list - [not null batch idxs for each expert]
        """
        eps = 1e-8
        return torch.where(weights[:, expert_idx] > eps)[0]

    def forward(self, input):
        """
        Args:
            input: dict (if need use embedding layer) or tensor
        """
        if self.embedding is not None:
            input = self.embedding(**input)
        gated_weights = self.gate(input)
        gated_weights = gated_weights.reshape(
            *gated_weights.shape, 1, 1
        )  # [bs, num_experts, 1, 1]

        weighted_sum_experts = 0
        for i, expert in enumerate(self.experts):
            expert_idx = self._get_not_null_expert_positions(gated_weights, i)
            expert_out = expert(input[expert_idx])  # [expert_bs, seq_len, emb_dim]
            expert_out = self._to_orig_positions(
                expert_idx, input.shape[0], expert_out
            )  # [bs, seq_len, emb_dim]

            weighted_sum_experts += expert_out * gated_weights[:, i]
        return MoeTabularOutput(
            logits=weighted_sum_experts,
            loss=None,
            task_gated_weights=gated_weights[:, :, 0, 0],
        )


class ExpertsWrapper(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(self, input):
        out = []
        for expert in self.experts:
            expert_out = expert(input)
            if hasattr(expert_out, "last_hidden_state"):
                out.append(expert_out.last_hidden_state)
            else:
                out.append(expert_out)
        return out


class MLPExperts(nn.Module):
    """
    mlp experts wrapper
    """

    def __init__(self, num_experts: int, dropout_p: float, hidden_size: int, scale=4):
        super().__init__()
        inner_hiden = hidden_size * scale
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, inner_hiden),
                nn.LayerNorm(inner_hiden),
                nn.SELU(),
                nn.Dropout(dropout_p),
                nn.Linear(inner_hiden, hidden_size),
            )
            for _ in range(num_experts)
        ])

    def forward(self, input):
        return [expert(input) for expert in self.experts]
