"""MMoE and PLE: several tasks sharing one pool of experts.

The shape is the same for both — embedding, a backbone of experts with a
per-task gate, one head per task, one composite loss. They differ only in how
the expert pool is organised:

* **MMoE** — every expert is visible to every task; the gate decides the mix.
* **PLE (CGC)** — the pool is split into shared experts plus a private group
  per task, so a task's gate sees ``[shared] + [its own]``. That containment is
  the point: it stops one task's gradient from dragging every expert.

``PLE`` is therefore ``MMoE`` with :class:`PLEBackbone` substituted, and the
class body is empty on purpose.
"""

import copy
import math

import torch
import torch.nn as nn

from avatar.data.tabular.batch import TabularBatch
from avatar.nn.tabular import BaseTabularEncoder
from avatar.nn.utils import get_aggregation_layer
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import MMoEOutput
from avatar.pipeline.uplift.treatment_interaction import (
    BaseTreatmentInteraction,
    ConcatTreatmentInteraction,
)


def inverse_sigmoid(probability: float) -> float:
    """Logit of ``probability``, clamped away from 0 and 1.

    Used to initialise gate parameters from a desired open probability, so
    ``init_shared=0.8`` means "start with this gate ~80% open" rather than
    naming a raw logit.
    """
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    return math.log(probability / (1.0 - probability))


class HierarchicalFeatureGate(nn.Module):
    """Learned per-feature gate with a shared and a task-specific component.

    Decides, per feature token, how much of it each task should see. Two sets
    of gate logits are learned — one shared across tasks, one per task — and
    combined, so a task can suppress a feature the others keep.

    Args:
        num_features: Number of feature tokens.
        num_tasks: Number of tasks.
        init_shared: Initial open probability of the shared gate.
        init_task_specific: Initial open probability of the task gates. Starts
            low on purpose: tasks begin from the shared view and specialise
            only where it pays.
        temperature: Sharpness of the gate; lower is closer to hard selection.

    Shape:
        ``x: (batch, num_features, embedding_dim)``, ``task_name: (batch,)``.
    """

    def __init__(
        self,
        num_features: int,
        num_tasks: int,
        init_shared: float = 0.8,
        init_task_specific: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_tasks = num_tasks
        self.temperature = temperature

        self.shared_logits = nn.Parameter(
            torch.full(
                (num_features,),
                inverse_sigmoid(init_shared),
            )
        )
        self.task_logits = nn.Parameter(
            torch.full(
                (num_tasks, num_features),
                inverse_sigmoid(init_task_specific),
            )
        )

    def get_gates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = torch.sigmoid(self.shared_logits / self.temperature)
        task_specific = torch.sigmoid(self.task_logits / self.temperature)

        effective = (
            shared.unsqueeze(0) + task_specific - shared.unsqueeze(0) * task_specific
        )

        return shared, task_specific, effective

    def forward(self, x: torch.Tensor, task_name: torch.LongTensor) -> torch.Tensor:
        # task_name = (
        #     str(int(task_name)) if torch.is_tensor(task_name) else str(task_name)
        # )

        _, _, effective = self.get_gates()
        batch_gates = effective[task_name]
        gated_x = x * batch_gates.unsqueeze(-1)

        return gated_x

    def regularization(
        self,
        lambda_shared: float = 1e-4,
        lambda_task: float = 1e-4,
        lambda_overlap: float = 1e-5,
    ):
        shared, task_specific, _ = self.get_gates()
        shared_penalty = shared.sum() / self.num_features
        task_penalty = task_specific.sum() / self.num_features

        overlap_penalty = (
            shared.unsqueeze(0) * task_specific
        ).sum() / self.num_features

        total = (
            lambda_shared * shared_penalty
            + lambda_task * task_penalty
            + lambda_overlap * overlap_penalty
        )

        return {
            "total": total,
            "shared": shared_penalty,
            "task": task_penalty,
            "overlap": overlap_penalty,
        }


class MLPExpert(nn.Module):
    """Pre-norm residual feed-forward block used as a cheap expert.

    Args:
        hidden_size: Token width.
        expansion: Inner width multiplier of the feed-forward network.
        dropout_p: Dropout inside the block.

    Shape:
        ``(batch, n_features, hidden_size) -> (batch, n_features, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: int = 2,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.SELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * expansion, hidden_size),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor):
        return x + self.ffn(self.norm(x))


class TabBackboneExpert(nn.Module):
    """A full tabular encoder used as one expert.

    Much heavier than :class:`MLPExpert` — each expert is a whole transformer —
    so this is for pools of a few experts, not dozens.

    Args:
        tabular_encoder: The encoder to use as the expert body.
        hidden_size: Token width; only needed when ``pre_norm`` is set.
        pre_norm: Layer-normalise the input before the encoder.
        residual: Add the input back to the encoder's output.

    Shape:
        ``(batch, n_features, hidden_size) -> (batch, n_features, hidden_size)``.
    """

    def __init__(
        self,
        tabular_encoder: BaseTabularEncoder,
        hidden_size: int | None = None,
        pre_norm: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        self.tabular_encoder = tabular_encoder
        self.norm = nn.LayerNorm(hidden_size) if pre_norm else nn.Identity()
        self.residual = residual

    def forward(self, x: torch.Tensor):
        y = self.tabular_encoder(self.norm(x))
        if self.residual:
            y = x + y
        return y


class TaskHead(nn.Module):
    """One task's output head, sized after the backbone is known.

    Constructed from config without knowing the backbone width, then wired up
    by the pipeline through :meth:`init_head` — which is why the head is not
    built in ``__init__``.

    Args:
        dropout_head: Dropout in the default feed-forward head.
        out_head: Replace the default head entirely.
        loss_fn: Loss over this task's logits; defaults to
            ``BCEWithLogitsLoss``. A custom loss is called with ``logits``,
            ``dist`` and ``targets`` instead.
    """

    def __init__(
        self,
        dropout_head: float = 0.15,
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
        """Build the head now that the backbone's output width is known."""
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
        device = combined_features.device
        logits = torch.zeros(combined_features.shape[0], 1, device=device)

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


class MMoEBackbone(nn.Module):
    """A pool of experts plus one gate per task.

    Every task mixes the same experts with its own learned weights, then the
    mixture is pooled and optionally concatenated with external embeddings.
    The gate weights are exposed on the output as ``task_gated_weights``, which
    is what :class:`~avatar.metrics.moe_reg.Entropy` and
    :class:`~avatar.metrics.moe_reg.Importance` read.

    Experts can be given directly (``experts``) or described
    (``expert_cls`` + ``expert_kwargs`` + ``num_experts``), in which case they
    are built as independent copies.

    Args:
        num_tasks: Number of tasks; determines how many gates exist.
        experts: Ready-made expert modules.
        expert_cls: Class to instantiate ``num_experts`` copies of instead.
        expert_kwargs: Arguments for ``expert_cls``.
        num_experts: How many experts to build from ``expert_cls``.
        shared_tabular_encoder: Encoder applied before the experts, shared by
            every task.
        aggregation_config: Config for
            :func:`~avatar.nn.utils.get_aggregation_layer`; defaults to mean.
        hidden_state_dim: Width of external embeddings concatenated after
            pooling.
        proj_hiddens_to_dim: Project those external embeddings to this width
            first.
        normalize_hidden_states: ``{name: width}`` — normalise each named
            external embedding separately; supersedes ``hidden_state_dim``.
        gate_hidden_dim: Hidden width of the gate network; ``None`` makes the
            gate a single linear layer.
        gate_dropout_p: Dropout inside the gate network.
    """

    def __init__(
        self,
        num_tasks: int,
        experts: list[nn.Module] | None = None,
        expert_cls: type[nn.Module] | None = None,
        expert_kwargs: dict | None = None,
        num_experts: int | None = None,
        shared_tabular_encoder: BaseTabularEncoder = None,
        aggregation_config=None,
        hidden_state_dim: int | None = None,
        proj_hiddens_to_dim: int | None = None,
        normalize_hidden_states: dict[str, int] | None = None,
        gate_hidden_dim: int | None = None,
        gate_dropout_p: float = 0.0,
    ):
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        super().__init__()

        self.tasks = [str(x) for x in range(num_tasks)]
        self.shared_tabular_encoder = shared_tabular_encoder
        self.input_agg_layer = get_aggregation_layer(
            **aggregation_config
        )  # input_agg_layer gate_agg_layer
        self.out_agg_layer = get_aggregation_layer(**aggregation_config)
        emb_dim = aggregation_config.get("emb_dim")
        if experts is None:
            experts = []
            expert_kwargs = {} if expert_kwargs is None else expert_kwargs
            for _ in range(num_experts):
                if isinstance(expert_cls, nn.Module):
                    experts.append(copy.deepcopy(expert_cls))
                else:
                    experts.append(expert_cls(**copy.deepcopy(expert_kwargs)))
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(self.experts)

        self.gates = nn.ModuleDict({
            task: self._make_gate(
                input_dim=emb_dim,
                num_experts=self.num_experts,
                hidden_dim=gate_hidden_dim,
                dropout_p=gate_dropout_p,
            )
            for task in self.tasks
        })

        if normalize_hidden_states is not None:
            self.external_embeddings_normalize = nn.ModuleDict({
                key: nn.LayerNorm(dim) for key, dim in normalize_hidden_states.items()
            })
            self.hidden_state_dim = sum(normalize_hidden_states.values())
        else:
            self.external_embeddings_normalize = None
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

        self.hidden_size = (
            emb_dim
            if self.hidden_state_dim is None
            else emb_dim + self.hidden_state_dim
        )
        self.last_gate = {}

    def _make_gate(self, input_dim, num_experts, hidden_dim=None, dropout_p=0.0):
        if hidden_dim is None:
            return nn.Linear(input_dim, num_experts)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_experts),
        )

    def _prepare_hidden_state(self, hidden_state):
        if self.external_embeddings_normalize is not None:
            hidden_state = torch.cat(
                tuple(
                    norm(hidden_state[key])
                    for key, norm in self.external_embeddings_normalize.items()
                ),
                dim=1,
            )
        elif hidden_state is not None:
            hidden_state = torch.cat(tuple(hidden_state.values()), dim=1)

        if hidden_state is not None and self.proj is not None:
            hidden_state = self.proj(hidden_state)
        return hidden_state

    def _gate_entropy_per_sample(self, gate: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return -(gate * torch.log(gate + eps)).sum(dim=-1)

    def forward(
        self,
        embeddings: torch.FloatTensor,
        hidden_state: torch.FloatTensor = None,
        task_name: int | str | None = None,
        **kwargs,
    ):
        task_name = (
            str(int(task_name)) if torch.is_tensor(task_name) else str(task_name)
        )

        x = embeddings
        if self.shared_tabular_encoder is not None:
            x = self.shared_tabular_encoder(x)
        gate_input = self.input_agg_layer(x)  # gate_agg_layer input_agg_layer
        gate = torch.softmax(self.gates[task_name](gate_input), dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (expert_outputs * gate[:, :, None, None]).sum(dim=1)
        aggregated_states = self.out_agg_layer(mixed)

        hidden_state = self._prepare_hidden_state(hidden_state)

        aux = {
            "gate": gate.detach(),
            "gate_entropy": self._gate_entropy_per_sample(gate).detach()
            / math.log(self.num_experts),
            "gate_entropy_loss": -self._gate_entropy_per_sample(gate).detach().mean(),
            "gate_max": gate.detach().max(dim=1).values,
            "gate_top1": gate.detach().argmax(dim=1),
        }

        if hidden_state is not None and self.hidden_state_dim is not None:
            return torch.cat([aggregated_states, hidden_state], dim=1), aux

        # self.last_gate[task_name] = gate.detach().mean(dim=0)
        return aggregated_states, aux


class MMoE(nn.Module):
    """Multi-gate mixture-of-experts over a tabular batch.

    Embeds the batch, runs it through :class:`MMoEBackbone`, and sends each
    record to the head of its own task. Every head contributes to one composite
    loss built by ``multi_task_loss``.

    Args:
        embedding: Tabular embedding layer.
        tabular_encoder: The expert backbone.
        heads: ``{task name: TaskHead}``. Head widths are set from the
            backbone here, so they need not be sized in config.
        multi_task_loss: Combines the per-head losses into the scalar the
            trainer optimises.
        n_groups: Number of campaign groups; adds a group embedding.
        add_treatment_feature: Also add a treatment embedding, for uplift-style
            multi-task runs.
        group_interaction: How the group embedding meets the feature tokens;
            defaults to concatenation.
        gate_entropy_loss_coef: Weight of an entropy penalty on the gates.
            Above 0 it pushes the gates to stay spread over experts instead of
            collapsing onto one.
        feature_gate: Optional :class:`HierarchicalFeatureGate` applied to the
            embedded features before the experts.

    Returns:
        :class:`~avatar.outputs.MMoEOutput`, carrying per-task logits and the
        gate weights the MoE diagnostics metrics read.
    """

    def __init__(
        self,
        embedding: nn.Module,
        tabular_encoder: MMoEBackbone,
        heads: dict[str, nn.Module],
        multi_task_loss: nn.Module,
        n_groups: int | None = None,
        add_treatment_feature: bool = False,
        group_interaction: BaseTreatmentInteraction = None,
        gate_entropy_loss_coef: float = 0.0,
        feature_gate: HierarchicalFeatureGate = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.tabular_encoder = tabular_encoder
        self.heads = nn.ModuleDict(heads)
        self.multi_task_loss = multi_task_loss

        for head in self.heads.values():
            head.init_head(tabular_encoder.hidden_size)

        self.group_embeddings = (
            nn.Embedding(num_embeddings=n_groups, embedding_dim=embedding.hidden_size)
            if n_groups is not None
            else None
        )
        self.group_interaction = group_interaction or ConcatTreatmentInteraction()

        if add_treatment_feature:
            self.treatment_embeddings = (
                nn.Embedding(num_embeddings=2, embedding_dim=embedding.hidden_size)
                if n_groups is not None
                else None
            )
            self.treatment_interaction = (
                group_interaction or ConcatTreatmentInteraction()
            )
        else:
            self.treatment_embeddings = None
            self.treatment_interaction = None

        self.gate_entropy_loss_coef = gate_entropy_loss_coef
        self.feature_gate = feature_gate

    def _add_group_embedding(self, embeddings, group):
        if self.group_embeddings is None:
            return embeddings
        return self.group_interaction(
            states=embeddings,
            treatment_embeddings=self.group_embeddings(group).unsqueeze(1),  #
        )

    def _add_treatment_embedding(self, embeddings, treatment):
        if self.treatment_embeddings is None:
            return embeddings
        return self.treatment_interaction(
            states=embeddings,
            treatment_embeddings=self.treatment_embeddings(treatment).unsqueeze(1),  #
        )

    def _forward_branch(
        self,
        tab_features: TabularBatch,
        group: torch.LongTensor,
        task: torch.LongTensor,
        targets: torch.LongTensor = None,
        **kwargs,
    ):
        device = task.device
        batch_size = task.shape[0]
        num_experts = self.tabular_encoder.num_experts

        logits = torch.zeros((batch_size, 1), device=device)
        losses = {}
        aux = {
            "gate": torch.zeros((batch_size, num_experts), device=device),
            "gate_entropy": torch.zeros(batch_size, device=device),
            "gate_entropy_loss": torch.tensor(0.0, device=device),
            "gate_max": torch.zeros(batch_size, device=device),
            "gate_top1": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

        embeddings = self.embedding(tab_features)
        embeddings = self._add_group_embedding(embeddings=embeddings, group=group)
        embeddings = self._add_treatment_embedding(
            embeddings=embeddings, treatment=kwargs.get("is_treat", None)
        )

        if self.feature_gate is not None:
            embeddings = self.feature_gate(embeddings, task)

        for task_name, head in self.heads.items():
            task_id = int(task_name)
            mask = task == task_id

            if mask.sum().item() < 2:
                losses[task_name] = torch.tensor(0.0, device=device)
                continue

            combined_features, task_aux = self.tabular_encoder(
                embeddings=embeddings[mask],
                hidden_state=tab_features[mask].hidden_states,
                task_name=task_id,
            )
            task_targets = targets[mask] if targets is not None else None
            task_logits, task_loss = head(combined_features, targets=task_targets)

            logits[mask] = task_logits
            if targets is not None:
                losses[task_name] = task_loss

            aux["gate"][mask] = task_aux["gate"]
            aux["gate_entropy"][mask] = task_aux["gate_entropy"]
            aux["gate_max"][mask] = task_aux["gate_max"]
            aux["gate_top1"][mask] = task_aux["gate_top1"]

            aux["gate_entropy_loss"] = (
                aux["gate_entropy_loss"] + task_aux["gate_entropy_loss"]
            )

        if targets is not None:
            loss = self.multi_task_loss(
                losses,
                dist=None,
                params=None,
                is_excluded=None,
            )

            loss = loss + self.gate_entropy_loss_coef * aux["gate_entropy_loss"]
        else:
            loss = None
        return logits, loss, aux

    def forward(
        self,
        tab_features: TabularBatch,
        group: torch.LongTensor,
        task_name: torch.LongTensor,
        targets: torch.LongTensor = None,
        **kwargs,
    ):
        device = task_name.device

        logits, loss, aux = None, None, None
        if self.training:
            _, loss, aux = self._forward_branch(
                tab_features=tab_features,
                targets=targets,
                group=group,
                task=task_name,
                **kwargs,
            )
        if not self.training:
            with torch.no_grad():
                logits, loss, aux = self._forward_branch(
                    tab_features=tab_features,
                    targets=targets,
                    group=group,
                    task=task_name,
                    **kwargs,
                )
        if loss is None:
            loss = torch.tensor(0.0, device=device)

        return MMoEOutput(
            loss=loss,
            conversion=targets,
            logits=logits,
            group=group,
            task_name=task_name,
            aux=aux,
        )


class PLEBackbone(MMoEBackbone):
    """Expert pool split into shared experts plus a private group per task.

    Each task's gate sees ``[shared experts] + [its own experts]``, so a task
    can specialise without competing for the whole pool. Everything else —
    pooling, external embeddings, exposed gate weights — is inherited from
    :class:`MMoEBackbone`.

    Args:
        num_tasks: Number of tasks.
        shared_expert_cls: Class of the shared experts.
        shared_expert_kwargs: Arguments for it.
        task_expert_cls: Class of the per-task experts.
        task_expert_kwargs: Arguments for it.
        num_shared_experts: Shared experts in the pool.
        num_specific_experts: Private experts **per task**, so the total number
            of expert modules is ``num_shared + num_tasks * num_specific``.
        **kwargs: Passed through to :class:`MMoEBackbone`.
    """

    def __init__(
        self,
        num_tasks: int,
        shared_expert_cls: type[nn.Module] | None = None,
        shared_expert_kwargs: dict | None = None,
        task_expert_cls: type[nn.Module] | None = None,
        task_expert_kwargs: dict | None = None,
        num_shared_experts: int = 2,
        num_specific_experts: int = 1,
        **kwargs,
    ):
        super().__init__(num_tasks=num_tasks, num_experts=0, **kwargs)

        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_experts = num_shared_experts + num_specific_experts

        shared_kw = {} if shared_expert_kwargs is None else shared_expert_kwargs

        def make_shared_expert():
            if isinstance(shared_expert_cls, nn.Module):
                return copy.deepcopy(shared_expert_cls)
            return shared_expert_cls(**copy.deepcopy(shared_kw))

        task_cls = task_expert_cls if task_expert_cls is not None else shared_expert_cls
        task_kw = task_expert_kwargs if task_expert_cls is not None else shared_kw

        def make_task_expert():
            if isinstance(task_cls, nn.Module):
                return copy.deepcopy(task_cls)
            return task_cls(**copy.deepcopy(task_kw))

        self.shared_experts = nn.ModuleList([
            make_shared_expert() for _ in range(num_shared_experts)
        ])
        self.task_experts = nn.ModuleDict({
            task: nn.ModuleList([
                make_task_expert() for _ in range(num_specific_experts)
            ])
            for task in self.tasks
        })

        emb_dim = kwargs.get("aggregation_config", {}).get("emb_dim")
        gate_hidden_dim = kwargs.get("gate_hidden_dim")
        gate_dropout_p = kwargs.get("gate_dropout_p", 0.0)

        self.gates = nn.ModuleDict({
            task: self._make_gate(
                input_dim=emb_dim,
                num_experts=self.num_experts,
                hidden_dim=gate_hidden_dim,
                dropout_p=gate_dropout_p,
            )
            for task in self.tasks
        })

        # Clean up the unused MMoE experts
        if hasattr(self, "experts"):
            del self.experts

    def forward(
        self,
        embeddings: torch.FloatTensor,
        hidden_state: torch.FloatTensor = None,
        task_name: int | str | None = None,
        **kwargs,
    ):
        task_name = (
            str(int(task_name)) if torch.is_tensor(task_name) else str(task_name)
        )

        x = embeddings
        if self.shared_tabular_encoder is not None:
            x = self.shared_tabular_encoder(x)

        gate_input = self.input_agg_layer(x)
        gate = torch.softmax(self.gates[task_name](gate_input), dim=-1)

        task_specific_experts = self.task_experts[task_name]
        experts_for_task = list(self.shared_experts) + list(task_specific_experts)

        expert_outputs = torch.stack([expert(x) for expert in experts_for_task], dim=1)
        mixed = (expert_outputs * gate[:, :, None, None]).sum(dim=1)
        aggregated_states = self.out_agg_layer(mixed)

        hidden_state = self._prepare_hidden_state(hidden_state)

        aux = {
            "gate": gate.detach(),
            "gate_entropy": self._gate_entropy_per_sample(gate).detach()
            / math.log(self.num_experts),
            "gate_entropy_loss": -self._gate_entropy_per_sample(gate).detach().mean(),
            "gate_max": gate.detach().max(dim=1).values,
            "gate_top1": gate.detach().argmax(dim=1),
        }

        if hidden_state is not None and self.hidden_state_dim is not None:
            return torch.cat([aggregated_states, hidden_state], dim=1), aux

        return aggregated_states, aux


class PLE(MMoE):
    """MMoE with a :class:`PLEBackbone`.

    Deliberately empty: the pipeline is identical and only the backbone
    differs, so the distinct class exists to name the architecture in configs
    and to let the PLE-aware metrics recognise the run.
    """
