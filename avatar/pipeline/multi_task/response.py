"""Multi-task response: several tasks, one backbone, no treatment flag."""

import torch
import torch.nn as nn

from avatar.data.tabular.batch import TabularBatch
from avatar.nn.tabular import BaseTabularEncoder
from avatar.nn.utils import get_aggregation_layer
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import MultiTaskxGroupResponseOutput
from avatar.pipeline.uplift.treatment_interaction import (
    BaseTreatmentInteraction,
    ConcatTreatmentInteraction,
)


class MultiTaskResponse(nn.Module):
    """Several response tasks over a shared (or per-task) backbone.

    Simpler than :class:`~avatar.pipeline.multi_task.mmoe.MMoE`: there is no
    gating. Sharing is decided structurally — pass a single module to share it
    across tasks, or a ``{task: module}`` mapping to give each task its own.

    Args:
        embedding: One embedding shared by all tasks, or one per task.
        tabular_encoder: One encoder shared by all tasks, or one per task.
        heads: ``{task name: MultiTaskHead}``; sized from the backbone here.
        multi_task_loss: Combines the per-task losses into one scalar.
        n_groups: Number of campaign groups; adds a group embedding.
        group_interaction: How that embedding meets the feature tokens.
        share_side_embeddings: Share the group embedding across tasks rather
            than giving each task its own.

    Returns:
        :class:`~avatar.outputs.MultiTaskxGroupResponseOutput`.
    """

    def __init__(
        self,
        embedding: nn.Module | dict[str, nn.Module],
        tabular_encoder: nn.Module | dict[str, nn.Module],
        heads: dict[str, nn.Module],
        multi_task_loss: nn.Module,
        n_groups: int | None = None,
        group_interaction: BaseTreatmentInteraction = None,
        share_side_embeddings: bool = False,
    ):
        super().__init__()

        self.embedding = (
            embedding if isinstance(embedding, nn.Module) else nn.ModuleDict(embedding)
        )
        self.tabular_encoder = (
            tabular_encoder
            if isinstance(tabular_encoder, nn.Module)
            else nn.ModuleDict(tabular_encoder)
        )
        _ = (
            [head.init_head(tabular_encoder.hidden_size) for _, head in heads.items()]
            if isinstance(tabular_encoder, nn.Module)
            else [
                heads.get(task_name).init_head(task_tabular_encoder.hidden_size)
                for task_name, task_tabular_encoder in self.tabular_encoder.items()
            ]
        )

        # Existing supported layouts:
        # 1. shared embedding + shared encoder
        # 2. shared embedding + per-task encoder
        # 3. per-task embedding + per-task encoder
        assert (
            (
                type(self.embedding) is not nn.ModuleDict
                and type(self.tabular_encoder) is not nn.ModuleDict
            )
            or (
                type(self.embedding) is nn.ModuleDict
                and type(self.tabular_encoder) is nn.ModuleDict
            )
            or (
                type(self.embedding) is not nn.ModuleDict
                and type(self.tabular_encoder) is nn.ModuleDict
            )
        )

        self.heads = nn.ModuleDict(heads)

        self.multi_task_loss = multi_task_loss

        if share_side_embeddings:
            if n_groups is not None:
                self.group_embeddings = nn.ModuleDict({
                    task_name: nn.Embedding(
                        num_embeddings=n_groups,
                        embedding_dim=self._get(self.embedding, task_name).hidden_size,
                    )
                    for task_name, _ in self.heads.items()
                })
            else:
                self.group_embeddings = None
        else:
            if n_groups is not None:
                self.group_embeddings = nn.Embedding(
                    num_embeddings=n_groups,
                    embedding_dim=(
                        self.embedding.hidden_size
                        if type(self.embedding) is not nn.ModuleDict
                        else self.embedding[
                            next(iter(self.embedding.keys()))
                        ].hidden_size
                    ),
                )
            else:
                self.group_embeddings = None

        self.group_interaction = group_interaction
        if group_interaction is None:
            self.group_interaction = ConcatTreatmentInteraction()

        self.n_groups = n_groups

        self.excluded_ids = self._get_nonshared_params()

    def _get(self, module, task_name):
        if module is None:
            return None
        if type(module) is nn.ModuleDict:
            return module[task_name]
        return module

    def _get_nonshared_params(self):
        nonshared_params = []
        modules = [
            self.embedding,
            self.tabular_encoder,
            self.heads,
            self.group_embeddings,
        ]
        for module in modules:
            if type(module) is nn.ModuleDict:
                _ = [
                    nonshared_params.extend(list(layers.parameters()))
                    for _, layers in self.module.items()
                ]
        excluded_ids = {id(p) for p in nonshared_params}
        return excluded_ids

    def _update_embeddings(self, embeddings, group, group_embeddings):
        if group_embeddings is not None:
            embeddings = self.group_interaction(
                states=embeddings,
                treatment_embeddings=group_embeddings(group).unsqueeze(1),
            )
        return embeddings

    def _forward_branch(
        self,
        tab_features: TabularBatch,
        targets: torch.LongTensor,
        group: torch.LongTensor,
        task: torch.LongTensor,
    ):
        device = targets.device
        logits = torch.zeros((targets.shape[0], 1), device=device)
        losses = {}
        cache = {}

        for task_name, head in self.heads.items():
            embedding = self._get(self.embedding, task_name)
            tabular_encoder = self._get(self.tabular_encoder, task_name)
            group_embeddings = self._get(self.group_embeddings, task_name)

            mask = task == int(task_name)
            if mask.sum().item() == 0:
                losses[task_name] = torch.tensor(0.0, device=device)
                continue

            embeddings = None
            if type(self.embedding) is nn.Module:
                if not self.share_side_embeddings:
                    if "emb" not in cache:
                        embeddings = embedding(tab_features)
                        cache["emb"] = embeddings
                    embeddings = cache["emb"]
                    embeddings = self._update_embeddings(
                        embeddings=embeddings[mask],
                        group=group[mask],
                        group_embeddings=group_embeddings,
                    )
                else:
                    if "emb" not in cache:
                        embeddings = embedding(tab_features)
                        embeddings = self._update_embeddings(
                            embeddings=embeddings,
                            group=group,
                            group_embeddings=group_embeddings,
                        )
                        cache["emb"] = embeddings
                    embeddings = cache["emb"]
            else:
                embeddings = embedding(tab_features[mask])
                embeddings = self._update_embeddings(
                    embeddings=embeddings,
                    group=group[mask],
                    group_embeddings=group_embeddings,
                )

            if (
                type(self.tabular_encoder) is not nn.ModuleDict
                and self.share_side_embeddings
            ):
                if "bb" not in cache:
                    combined_features = tabular_encoder(
                        embeddings=embeddings, hidden_state=tab_features.hidden_states
                    )
                    cache["bb"] = combined_features
                combined_features = cache["bb"]
                task_logits, task_loss = head(
                    combined_features[mask], targets=targets[mask]
                )
            else:
                combined_features = tabular_encoder(
                    embeddings=embeddings,
                    hidden_state=tab_features[mask].hidden_states,
                )
                task_logits, task_loss = head(combined_features, targets=targets[mask])
            logits[mask] = task_logits
            losses[task_name] = task_loss

        loss = self.multi_task_loss(
            losses,
            dist=combined_features
            if type(self.tabular_encoder) is not nn.ModuleDict
            else embeddings
            if type(self.embedding) is not nn.ModuleDict
            else None,
            params=self.parameters(),
            is_excluded=lambda p: id(p) in self.excluded_ids,
        )
        return logits, loss

    def forward(
        self,
        tab_features: TabularBatch,
        targets: torch.LongTensor,
        group: torch.LongTensor,
        task_name: torch.LongTensor,
        **kwargs,
    ):
        device = targets.device

        logits, loss = None, None
        if self.training:
            _, loss = self._forward_branch(
                tab_features=tab_features,
                targets=targets,
                group=group,
                task=task_name,
            )
        if not self.training:
            with torch.no_grad():
                logits, loss = self._forward_branch(
                    tab_features=tab_features,
                    targets=targets,
                    group=group,
                    task=task_name,
                )
        if loss is None:
            loss = torch.tensor(0.0).to(device)

        return MultiTaskxGroupResponseOutput(
            loss=loss,
            conversion=targets,
            logits=logits,
            group=group,
            task_name=task_name,
        )


class MultiTaskBackbone(nn.Module):
    """Encoder plus pooling plus external-embedding fusion, without gating.

    The non-MoE counterpart of
    :class:`~avatar.pipeline.multi_task.mmoe.MMoEBackbone`: same output
    contract, no experts and no gates.

    Args:
        tabular_encoder: Encoder over the feature tokens.
        aggregation_config: Config for
            :func:`~avatar.nn.utils.get_aggregation_layer`; defaults to mean.
        hidden_state_dim: Width of external embeddings concatenated after
            pooling.
        proj_hiddens_to_dim: Project them to this width first.
        normalize_hidden_states: ``{name: width}`` — normalise each named
            external embedding separately; supersedes ``hidden_state_dim``.
    """

    def __init__(
        self,
        tabular_encoder: BaseTabularEncoder,
        aggregation_config=None,
        hidden_state_dim: int | None = None,
        proj_hiddens_to_dim: int | None = None,
        normalize_hidden_states: dict[str, int] | None = None,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()
        self.tabular_backbone = tabular_encoder

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

        self.agg_layer = get_aggregation_layer(**aggregation_config)

        if self.hidden_state_dim is None:
            self.hidden_size = aggregation_config.get("emb_dim")
        else:
            self.hidden_size = aggregation_config.get("emb_dim") + self.hidden_state_dim

    def forward(
        self,
        embeddings: torch.FloatTensor,
        hidden_state: torch.FloatTensor = None,
        **kwargs,
    ):
        if self.external_embeddings_normalize is not None:
            for key, normalize_layer in self.external_embeddings_normalize.items():
                hidden_state[key] = normalize_layer(hidden_state[key])
            hidden_state = torch.cat(tuple(hidden_state.values()), dim=1)
        elif hidden_state is not None:
            hidden_state = torch.cat(tuple(hidden_state.values()), dim=1)
        else:
            hidden_state = None

        last_hidden_states = self.tabular_backbone(embeddings)
        aggregated_states = self.agg_layer(last_hidden_states)

        if hidden_state is not None and self.hidden_state_dim is not None:
            combined_features = torch.cat(
                [aggregated_states, self.proj(hidden_state)], dim=1
            )
        else:
            combined_features = aggregated_states

        return combined_features


class MultiTaskHead(nn.Module):
    """One task's response head, sized once the backbone width is known.

    Args:
        dropout_head: Dropout in the default feed-forward head.
        separate_heads: Accepted for symmetry with the uplift head; a response
            task has a single branch, so it changes nothing here.
        out_head: Replace the default head entirely.
        loss_fn: Loss over this task's logits; defaults to
            ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        dropout_head: float = 0.15,
        separate_heads: bool = False,
        out_head: nn.Module = None,
        loss_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.dropout_head = dropout_head
        self.out_head = out_head
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()

    def init_head(
        self,
        hidden_size: int,
    ):
        self.out_head = (
            FeedForwardNetwork(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                output_dim=1,
                dropout_p=self.dropout_head,
                activation=nn.SELU,
                normalization=nn.BatchNorm1d,
                end_normalization=False,
                dropout_first=True,
                activation_before_normalization=True,
            )
            if self.out_head is None
            else self.out_head
        )

    def forward(
        self,
        combined_features: torch.FloatTensor,
        targets: torch.LongTensor,
        **kwrags,
    ):
        device = targets.device
        logits = torch.zeros(targets.shape[0], 1, device=device)

        logits = self.out_head(combined_features)
        loss = None
        if targets is not None:
            if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
                loss = self.loss_fn(logits.squeeze(1), targets.float())
            else:
                loss = self.loss_fn(
                    logits=logits,
                    dist=combined_features,
                    targets=targets,
                )
        return logits, loss
