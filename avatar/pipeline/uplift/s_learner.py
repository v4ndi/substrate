import torch
import torch.nn as nn

from avatar.data.tabular_batch import TabularBatch
from avatar.nn.embedding.base_embedding import BaseTabularEmbedding
from avatar.nn.tabular import BaseTabularEncoder
from avatar.nn.utils import get_aggregation_layer
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import MultiGroupUpliftOutput
from avatar.pipeline.uplift.treatment_interaction import (
    BaseTreatmentInteraction,
    ConcatTreatmentInteraction,
)


class SLearner(nn.Module):
    """Simple S-Learner model

    Args:
        embedding: BaseTabularEmbedding - tabular embedding layer
        tabular_encoder: BaseTabularEncoder - tabular encoder layer
        aggregation_config: dict - dictionary with aggregation parameters
        n_groups: int - number of groups for uplift; Defualt: None
        hidden_state_dim: int - number of hidden state dimensions; Defualt: None
        dropout_head: float - dropout rate in the output head; Defualt: 0.15
        separate_heads: bool - whether to have separate heads for treatment and
            control; Defualt: False
        treatment_interaction: BaseTreatmentInteraction; Default None -> ConcatTreatmentInteraction
        calculate_train_uplift: bool - whether to forward model twice
            for calculating uplift during training; Defualt: False
        exchange_treatment_group: bool ??
        out_head: nn.Module - output head; Defualt: FFN
        proj_hiddens_to_dim: int - project hidden states to this dimension; Defualt: None
        normalize_hidden_states: dict[str, int] - dictionary with hidden states to normalize; Defualt: None
        loss_fn: nn.Module = nn.CrossEntropyLoss() | Any
    """

    def __init__(
        self,
        embedding: BaseTabularEmbedding,
        tabular_encoder: BaseTabularEncoder,
        aggregation_config=None,
        n_groups: int | None = None,
        hidden_state_dim: int | None = None,
        dropout_head: float = 0.15,
        separate_heads: bool = False,
        treatment_interaction: BaseTreatmentInteraction = None,
        group_interaction: BaseTreatmentInteraction = None,
        calculate_train_uplift: bool = False,
        exchange_treatment_group: bool = False,
        out_head: nn.Module = None,
        proj_hiddens_to_dim: int | None = None,
        normalize_hidden_states: dict[str, int] | None = None,  # добавить в докстринг
        loss_fn: nn.Module | None = None,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()
        self.calculate_train_metrics = calculate_train_uplift

        if normalize_hidden_states is not None:
            self.external_embeddings_normalize = nn.ModuleDict({
                key: nn.LayerNorm(dim) for key, dim in normalize_hidden_states.items()
            })
        else:
            self.external_embeddings_normalize = None

        if self.external_embeddings_normalize is not None:
            self.hidden_state_dim = sum(normalize_hidden_states.values())
        else:
            self.hidden_state_dim = hidden_state_dim

        if self.hidden_state_dim is not None:
            if proj_hiddens_to_dim is not None:
                self.proj = nn.Sequential(
                    nn.LayerNorm(self.hidden_state_dim),
                    nn.Linear(self.hidden_state_dim, proj_hiddens_to_dim),
                )
                self.hidden_state_dim = proj_hiddens_to_dim
            else:
                self.proj = nn.LayerNorm(self.hidden_state_dim)
        else:
            self.proj = None

        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        self.embedding = embedding
        self.tabular_backbone = tabular_encoder
        self.treatment_features = nn.Embedding(
            num_embeddings=2, embedding_dim=self.embedding.hidden_size
        )
        if n_groups is not None:
            self.group_features = nn.Embedding(
                num_embeddings=n_groups, embedding_dim=self.embedding.hidden_size
            )
        else:
            self.group_features = None

        self.n_groups = n_groups
        self.agg_layer = get_aggregation_layer(**aggregation_config)

        if self.hidden_state_dim is None:
            hidden_size = self.embedding.hidden_size
        else:
            hidden_size = self.embedding.hidden_size + self.hidden_state_dim

        if separate_heads:
            self.out_head = nn.ModuleDict({
                "treatment": FeedForwardNetwork(
                    input_dim=hidden_size,
                    hidden_dim=hidden_size,
                    output_dim=2,
                    dropout_p=dropout_head,
                    activation=nn.SELU,
                    normalization=nn.BatchNorm1d,
                    end_normalization=False,
                    dropout_first=True,
                    activation_before_normalization=True,
                ),
                "control": FeedForwardNetwork(
                    input_dim=hidden_size,
                    hidden_dim=hidden_size,
                    output_dim=2,
                    dropout_p=dropout_head,
                    activation=nn.SELU,
                    normalization=nn.BatchNorm1d,
                    end_normalization=False,
                    dropout_first=True,
                    activation_before_normalization=True,
                ),
            })
        else:
            self.out_head = (
                FeedForwardNetwork(
                    input_dim=hidden_size,
                    hidden_dim=hidden_size,
                    output_dim=2,
                    dropout_p=dropout_head,
                    activation=nn.SELU,
                    normalization=nn.BatchNorm1d,
                    end_normalization=False,
                    dropout_first=True,
                    activation_before_normalization=True,
                )
                if out_head is None
                else out_head
            )

        self.treatment_interaction = treatment_interaction
        if treatment_interaction is None:
            self.treatment_interaction = ConcatTreatmentInteraction()

        self.group_interaction = group_interaction
        if group_interaction is None:
            self.group_interaction = ConcatTreatmentInteraction()

        self.exchange_treatment_group = exchange_treatment_group

    def _forward_branch(
        self,
        tab_features: TabularBatch,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor = None,
        group: torch.LongTensor = None,
        hidden_state: torch.FloatTensor = None,
    ):
        embeddings = self.embedding(tab_features)  # [bs, n_features, dim]
        treat_embeds = self.treatment_features(is_treat).unsqueeze(1)  # [bs, 1, dim]
        embeddings = self.treatment_interaction(
            states=embeddings, treatment_embeddings=treat_embeds
        )

        if self.group_features is not None:
            group_embeds = self.group_features(group).unsqueeze(1)  # [bs, 1, dim]
            embeddings = self.group_interaction(
                states=embeddings, treatment_embeddings=group_embeds
            )

        last_hidden_states = self.tabular_backbone(embeddings)
        aggregated_states = self.agg_layer(last_hidden_states)

        # Initialize full logits tensor
        device = aggregated_states.device
        logits = torch.zeros((is_treat.shape[0], 2), device=device)  # [bs, 2]

        # Late Fusion Concat
        if hidden_state is not None and self.hidden_state_dim is not None:
            combined_features = torch.cat(
                [aggregated_states, self.proj(hidden_state)], dim=1
            )
        else:
            combined_features = aggregated_states

        # Check if we're using separate heads
        if hasattr(self, "out_head") and isinstance(self.out_head, nn.ModuleDict):
            # Dual-head case (treatment and control separate)
            if is_treat.bool().sum() > 0:  # Treatment cases
                logits[is_treat.bool()] = self.out_head["treatment"](
                    combined_features[is_treat.bool()]
                )
            if (~is_treat.bool()).sum() > 0:  # Control cases
                logits[~is_treat.bool()] = self.out_head["control"](
                    combined_features[~is_treat.bool()]
                )
        else:
            # Single-head case
            logits = self.out_head(combined_features)
        loss = None
        if targets is not None:
            if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                loss = self.loss_fn(logits, targets)
            else:
                loss = self.loss_fn(
                    logits=logits,
                    is_treat=is_treat,
                    dist=combined_features,
                    targets=targets,
                )

        return logits, loss

    def forward(
        self,
        tab_features: TabularBatch,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor = None,
        group: torch.LongTensor = None,
        **kwargs,
    ):
        device = is_treat.device
        hidden_state = tab_features.hidden_states

        if self.external_embeddings_normalize is not None:
            for key, normalize_layer in self.external_embeddings_normalize.items():
                hidden_state[key] = normalize_layer(hidden_state[key])
            hidden_state = torch.cat(tuple(hidden_state.values()), dim=1)
        elif hidden_state is not None:
            hidden_state = torch.cat(tuple(hidden_state.values()), dim=1)
        else:
            hidden_state = None

        loss = None
        if self.training:
            if self.exchange_treatment_group:
                train_group = torch.where(
                    is_treat == 1,
                    group,
                    (self.n_groups - 1) * torch.ones_like(group, device=device),
                )
            else:
                train_group = group
            _, loss = self._forward_branch(
                tab_features=tab_features,
                is_treat=is_treat,
                targets=targets,
                group=train_group,
                hidden_state=hidden_state,
            )
        t_probs, c_probs, uplift = None, None, None

        if self.calculate_train_metrics or not self.training:
            with torch.no_grad():
                treat_logits, _ = self._forward_branch(
                    tab_features=tab_features,
                    is_treat=torch.ones_like(is_treat).to(device),
                    targets=targets,
                    group=group,
                    hidden_state=hidden_state,
                )
                if self.exchange_treatment_group:
                    calib_contol_group = (self.n_groups - 1) * torch.ones_like(
                        group, device=device
                    )
                else:
                    calib_contol_group = group
                control_logits, _ = self._forward_branch(
                    tab_features=tab_features,
                    is_treat=torch.zeros_like(is_treat).to(device),
                    targets=targets,
                    group=calib_contol_group,
                    hidden_state=hidden_state,
                )
                t_probs = torch.softmax(treat_logits, dim=-1)[:, 1]
                c_probs = torch.softmax(control_logits, dim=-1)[:, 1]
                uplift = t_probs - c_probs

        if loss is None:
            loss = torch.tensor(0.0).to(device)

        return MultiGroupUpliftOutput(
            loss=loss,
            conversion=targets,
            treatment=is_treat,
            treatment_probs=t_probs,
            control_probs=c_probs,
            uplift=uplift,
            group=group,
        )
