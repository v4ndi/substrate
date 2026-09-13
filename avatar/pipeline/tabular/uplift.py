"""S-Learner: the supervised pipeline with treatment supplied as a feature.

That sentence is also the class hierarchy. ``SLearner`` subclasses
:class:`~avatar.pipeline.tabular.supervised.SupervisedLearner` and adds exactly
three things: a treatment token beside the feature tokens, a head two wide, and
the second and third forward passes that turn two outcome probabilities into an
uplift.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

from avatar.data.tabular.batch import TabularBatch
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import MultiGroupUpliftOutput
from avatar.pipeline.tabular.interaction import (
    BaseTreatmentInteraction,
    ConcatTreatmentInteraction,
)
from avatar.pipeline.tabular.supervised import SupervisedLearner

__all__ = ["SLearner"]


class SLearner(SupervisedLearner):
    """S-Learner: one shared model, the treatment flag fed in as a feature.

    Training runs the batch once, with each record's real treatment value.
    Scoring runs it **twice** — everything forced to treated, then everything
    forced to control — and reports ``uplift = P(y|treated) - P(y|control)``.
    That second pass is why evaluation costs roughly twice a plain classifier,
    and why ``calculate_train_uplift`` is off by default.

    Beyond :class:`~avatar.pipeline.tabular.supervised.SupervisedLearner`:

    Args:
        separate_heads: Give treatment and control their own head instead of
            sharing one. The shared backbone is unaffected, and ``out_head`` is
            ignored when this is set.
        treatment_interaction: How the treatment embedding meets the feature
            tokens. Defaults to concatenation, which adds a token — so
            ``aggregation_config.num_features`` counts one more than the
            features, and one more again when ``n_groups`` is set. This is the
            single most common sizing mistake in uplift configs.
        calculate_train_uplift: Also run the two scoring passes during
            training, so uplift metrics are available on the train set. Costs
            two extra forward passes per step.
        exchange_treatment_group: Reserve the **last** group id as a dedicated
            control group: control records are relabelled to ``n_groups - 1``
            during training, and the calibrated control pass uses that id too.
            Only meaningful together with ``n_groups``.
        loss_fn: Loss over the head's logits. Defaults to cross-entropy; a
            custom loss is called with ``logits``, ``is_treat``, ``dist`` and
            ``targets`` instead, which is what the research losses need.

    Returns:
        :class:`~avatar.outputs.MultiGroupUpliftOutput`. During training
        ``treatment_probs`` / ``control_probs`` / ``uplift`` are ``None``
        unless ``calculate_train_uplift`` is set.
    """

    required_inputs = ("tab_features", "is_treat")

    #: The uplift heads were built with dropout first and the activation ahead
    #: of the normalisation. Every checkpoint trained from an uplift config has
    #: that layer order baked in, so it is kept rather than unified with the
    #: supervised head's.
    head_layout: ClassVar[dict[str, bool]] = {
        "dropout_first": True,
        "activation_before_normalization": True,
        "end_normalization": False,
    }

    def __init__(
        self,
        separate_heads: bool = False,
        treatment_interaction: BaseTreatmentInteraction | None = None,
        calculate_train_uplift: bool = False,
        exchange_treatment_group: bool = False,
        loss_fn: nn.Module | None = None,
        **kwargs,
    ):
        self.separate_heads = separate_heads
        kwargs.setdefault("num_classes", 2)
        super().__init__(loss=loss_fn, **kwargs)

        self.treatment_features = nn.Embedding(
            num_embeddings=2, embedding_dim=self.hidden_size
        )
        self.treatment_interaction = (
            treatment_interaction or ConcatTreatmentInteraction()
        )
        self.calculate_train_metrics = calculate_train_uplift
        self.exchange_treatment_group = exchange_treatment_group

    def _init_loss(self, loss, num_classes, task_type, l1_weight):
        """Keep the uplift loss under its own name and calling convention.

        The attribute stays ``loss_fn`` because a loss with parameters — the
        gradnorm balancer is the only one — would otherwise have every
        checkpoint key renamed. The two calling conventions are the subject of
        P1 in ``docs/decisions/pipeline_boundaries.md``.
        """
        self.loss_fn = loss or nn.CrossEntropyLoss()

    def build_head(self, input_dim, output_dim, hidden_dim, dropout_p) -> nn.Module:
        """One head, or one per arm when ``separate_heads`` is set."""

        def head():
            return FeedForwardNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout_p=dropout_p,
                activation=nn.SELU,
                normalization=nn.BatchNorm1d,
                **self.head_layout,
            )

        if not self.separate_heads:
            return head()
        return nn.ModuleDict({"treatment": head(), "control": head()})

    def extra_tokens(
        self,
        group: torch.LongTensor | None = None,
        is_treat: torch.LongTensor | None = None,
        **kwargs,
    ):
        """Treatment first, then group — the order the token positions assume."""
        tokens = (
            (
                self.treatment_features(is_treat).unsqueeze(1),
                self.treatment_interaction,
            ),
        )
        return tokens + super().extra_tokens(group=group)

    def apply_head(self, features: torch.Tensor, is_treat: torch.LongTensor):
        """Score the pooled features, routing each arm to its own head if asked."""
        if not isinstance(self.out_head, nn.ModuleDict):
            return self.out_head(features)

        treated = is_treat.bool()
        logits = torch.zeros((is_treat.shape[0], 2), device=features.device)
        if treated.sum() > 0:
            logits[treated] = self.out_head["treatment"](features[treated])
        if (~treated).sum() > 0:
            logits[~treated] = self.out_head["control"](features[~treated])
        return logits

    def compute_loss(self, logits, targets, is_treat=None, features=None, **kwargs):
        """Score the head's output.

        Cross-entropy, or a research loss that also wants the arm and the
        pooled representation.
        """
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            return self.loss_fn(logits, targets), None
        loss = self.loss_fn(
            logits=logits, is_treat=is_treat, dist=features, targets=targets
        )
        return loss, None

    def _forward_branch(
        self,
        tab_features: TabularBatch,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor = None,
        group: torch.LongTensor = None,
    ):
        features = self.represent(
            tab_features, self.extra_tokens(group=group, is_treat=is_treat)
        )
        logits = self.apply_head(features, is_treat)
        loss = None
        if targets is not None:
            loss, _ = self.compute_loss(
                logits, targets, is_treat=is_treat, features=features
            )
        return logits, loss

    def _control_group(self, group: torch.LongTensor, is_treat=None):
        """Group ids for a control pass, honouring ``exchange_treatment_group``."""
        if not self.exchange_treatment_group:
            return group
        control = (self.n_groups - 1) * torch.ones_like(group, device=group.device)
        if is_treat is None:
            return control
        return torch.where(is_treat == 1, group, control)

    def forward(
        self,
        tab_features: TabularBatch,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor = None,
        group: torch.LongTensor = None,
        **kwargs,
    ) -> MultiGroupUpliftOutput:
        """One pass to learn from, two more to score the difference."""
        device = is_treat.device

        loss = None
        if self.training:
            _, loss = self._forward_branch(
                tab_features=tab_features,
                is_treat=is_treat,
                targets=targets,
                group=self._control_group(group, is_treat)
                if group is not None
                else None,
            )

        treatment_probs, control_probs, uplift = None, None, None
        if self.calculate_train_metrics or not self.training:
            with torch.no_grad():
                treat_logits, _ = self._forward_branch(
                    tab_features=tab_features,
                    is_treat=torch.ones_like(is_treat).to(device),
                    targets=targets,
                    group=group,
                )
                control_logits, _ = self._forward_branch(
                    tab_features=tab_features,
                    is_treat=torch.zeros_like(is_treat).to(device),
                    targets=targets,
                    group=self._control_group(group) if group is not None else None,
                )
                treatment_probs = torch.softmax(treat_logits, dim=-1)[:, 1]
                control_probs = torch.softmax(control_logits, dim=-1)[:, 1]
                uplift = treatment_probs - control_probs

        if loss is None:
            loss = torch.tensor(0.0).to(device)

        return MultiGroupUpliftOutput(
            loss=loss,
            conversion=targets,
            treatment=is_treat,
            treatment_probs=treatment_probs,
            control_probs=control_probs,
            uplift=uplift,
            group=group,
        )
