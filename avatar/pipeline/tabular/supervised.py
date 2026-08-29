import torch
import torch.nn as nn

from avatar.data.tabular_batch import TabularBatch
from avatar.nn.embedding.base_embedding import BaseTabularEmbedding
from avatar.nn.tabular import BaseTabularEncoder
from avatar.nn.utils import get_aggregation_layer
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import TabularOutput
from avatar.pipeline.uplift.treatment_interaction import (
    BaseTreatmentInteraction,
    ConcatTreatmentInteraction,
)


class SupervisedLearner(nn.Module):
    """
    A neural network for tabular data classification in response setting.
     Args:
        embedding: BaseTabularEmbedding - tabular embedding layer
        tabular_encoder: BaseTabularEncoder - tabular encoder layer
        aggregation_config: dict - dictionary with aggregation parameters
        n_groups: int - number of groups for uplift; Defualt: None
        hidden_state_dim: int - number of hidden state dimensions; Defualt: None
        dropout_head: float - dropout rate in the output head; Defualt: 0.15
        group_interaction: BaseTreatmentInteraction; Default None -> ConcatTreatmentInteraction
        out_head: nn.Module - output head; Defualt: FFN
        proj_hiddens_to_dim: int - project hidden states to this dimension; Defualt: None
        normalize_hidden_states: dict[str, int] - dictionary with hidden states to normalize; Defualt: None
        loss_fn: nn.Module = nn.BCEWithLogitsLoss() | Any
    """

    def __init__(
        self,
        embedding: BaseTabularEmbedding,
        tabular_encoder: BaseTabularEncoder,
        aggregation_config=None,
        n_groups: int | None = None,
        hidden_state_dim: int | None = None,
        dropout_head: float = 0.15,
        group_interaction: BaseTreatmentInteraction = None,
        out_head: nn.Module = None,
        proj_hiddens_to_dim: int | None = None,
        normalize_hidden_states: dict[str, int] | None = None,
        loss_fn: nn.Module | None = None,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()

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

        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()

        self.embedding = embedding
        self.tabular_backbone = tabular_encoder

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

        self.out_head = (
            FeedForwardNetwork(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                output_dim=1,
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

        self.group_interaction = group_interaction
        if group_interaction is None:
            self.group_interaction = ConcatTreatmentInteraction()

    def _forward_branch(
        self,
        tab_features: TabularBatch,
        targets: torch.LongTensor = None,
        group: torch.LongTensor = None,
        hidden_state: torch.FloatTensor = None,
    ):
        embeddings = self.embedding(tab_features)  # [bs, n_features, dim]

        if self.group_features is not None:
            group_embeds = self.group_features(group).unsqueeze(1)  # [bs, 1, dim]
            embeddings = self.group_interaction(
                states=embeddings, treatment_embeddings=group_embeds
            )

        last_hidden_states = self.tabular_backbone(embeddings)
        aggregated_states = self.agg_layer(last_hidden_states)

        # Initialize full logits tensor
        device = aggregated_states.device
        logits = torch.zeros((targets.shape[0], 1), device=device)  # [bs, 2]

        # Late Fusion Concat
        if hidden_state is not None and self.hidden_state_dim is not None:
            combined_features = torch.cat(
                [aggregated_states, self.proj(hidden_state)], dim=1
            )
        else:
            combined_features = aggregated_states

        logits = self.out_head(combined_features)
        loss = None
        if targets is not None:
            loss = self.loss_fn(logits.squeeze(1), targets.float())
        return logits, loss

    def forward(
        self,
        tab_features: TabularBatch,
        targets: torch.LongTensor = None,
        group: torch.LongTensor = None,
        **kwargs,
    ):
        _ = targets.device
        hidden_state = tab_features.hidden_states

        if self.external_embeddings_normalize is not None:
            for key, normalize_layer in self.external_embeddings_normalize.items():
                hidden_state[key] = normalize_layer(hidden_state[key])
            hidden_state = torch.cat(tuple(hidden_state.values()), dim=1)
        elif hidden_state is not None:
            hidden_state = torch.cat(tuple(hidden_state.values()), dim=1)
        else:
            hidden_state = None

        logits, loss = None, None
        if self.training:
            _, loss = self._forward_branch(
                tab_features=tab_features,
                targets=targets,
                group=group,
                hidden_state=hidden_state,
            )

        if not self.training:
            with torch.no_grad():
                logits, loss = self._forward_branch(
                    tab_features=tab_features,
                    targets=targets,
                    group=group,
                    hidden_state=hidden_state,
                )
        return TabularOutput(logits=logits, loss=loss)
