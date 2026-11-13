"""The supervised tabular pipeline: one pass, one head, one loss.

This is the whole tabular stack — embedding, encoder, pooling, external
embeddings, head, loss — in one class, because every tabular pipeline needs all
of it and there is only one way to assemble it.
:class:`~avatar.pipeline.tabular.uplift.SLearner` subclasses this one and adds
the treatment feature, which is what an S-Learner is.

**The module attribute names are load-bearing.** ``embedding``,
``tabular_backbone``, ``agg_layer``, ``proj``, ``out_head`` and
``group_features`` are the prefixes of every key in every checkpoint trained
from the uplift configs, so they are kept exactly as they were rather than
tidied into a sub-module — a neater object graph is not worth invalidating
saved models.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

from avatar.data.tabular.batch import TabularBatch
from avatar.losses.base import Loss
from avatar.losses.classification import ClassificationLoss
from avatar.nn.embedding import BaseTabularEmbedding
from avatar.nn.tabular import BaseTabularEncoder
from avatar.nn.utils import get_aggregation_layer
from avatar.nn.utils.ffn import FeedForwardNetwork
from avatar.outputs import TabularOutput
from avatar.pipeline.base import BasePipeline
from avatar.pipeline.tabular.interaction import (
    BaseTreatmentInteraction,
    ConcatTreatmentInteraction,
)

__all__ = ["SupervisedLearner"]


class SupervisedLearner(BasePipeline):
    """Supervised learning over a tabular batch.

    Covers three settings, which differ only in two arguments:

    ==================================  =============  ====================
    setting                             ``num_classes``  ``task_type``
    ==================================  =============  ====================
    binary classification (response)    ``1``          ``classification``
    regression                          ``1``          ``regression``
    multi-class classification          ``K > 1``      ``classification``
    ==================================  =============  ====================

    ``num_classes: 1`` means one number per record — one probability or one
    value — which is what ``ResponseMetrics`` and ``RegressionMetrics`` read.
    The head emits ``(B, 1)`` and collate functions emit ``(B,)``; reconciling
    the two is :meth:`~avatar.losses.ClassificationLoss.align`'s job, not this
    class's.

    Args:
        embedding: Tabular embedding layer. ``None`` runs the model on external
            embeddings alone, with no feature tokens at all.
        tabular_encoder: Encoder over the feature tokens. ``None`` alongside
            ``embedding``.
        aggregation_config: Config for
            :func:`~avatar.nn.utils.get_aggregation_layer`; defaults to mean
            pooling. Its ``num_features`` counts the tokens reaching the
            aggregation, so it must include the group token when ``n_groups``
            is set — and, in the uplift subclass, the treatment token too.
        num_classes: Width of the head.
        task_type: ``"classification"`` or ``"regression"``.
        hidden_state_dim: Width of the external embeddings concatenated after
            pooling (late fusion). ``None`` disables it.
        proj_hiddens_to_dim: Project the external embeddings to this width
            instead of only normalising them.
        normalize_hidden_states: ``{name: width}`` — layer-normalise each named
            external embedding separately before concatenating. When set it
            determines the total external width and ``hidden_state_dim`` is
            ignored.
        n_groups: Number of campaign groups. When set, a group embedding joins
            the feature tokens.
        group_interaction: How the group embedding meets the feature tokens.
            Defaults to concatenation, which adds a token.
        dropout_p: Dropout in the head.
        out_head_hidden_dim: Hidden width of the default head. ``None`` makes
            it equal to the head's input width.
        out_head: Replace the default head entirely.
        l1_weight: Weight of the L1 penalty over the embedding's parameters.
        loss: Replace the loss. Defaults to
            :class:`~avatar.losses.ClassificationLoss` built from
            ``num_classes`` and ``task_type``.

    Returns:
        :class:`~avatar.outputs.TabularOutput` — logits always, loss only when
        targets were given.
    """

    required_inputs = ("tab_features",)

    #: How the default head orders dropout, activation and normalisation. The
    #: uplift subclass was built with a different order and its checkpoints
    #: keep it, so the two are named here rather than hidden in a call.
    head_layout: ClassVar[dict[str, bool]] = {
        "dropout_first": False,
        "activation_before_normalization": False,
        "end_normalization": False,
    }

    def __init__(
        self,
        embedding: BaseTabularEmbedding | None = None,
        tabular_encoder: BaseTabularEncoder | None = None,
        aggregation_config: dict | None = None,
        num_classes: int = 1,
        task_type: str = "classification",
        hidden_state_dim: int | None = None,
        proj_hiddens_to_dim: int | None = None,
        normalize_hidden_states: dict[str, int] | None = None,
        n_groups: int | None = None,
        group_interaction: BaseTreatmentInteraction | None = None,
        dropout_p: float = 0.15,
        out_head_hidden_dim: int | None = None,
        out_head: nn.Module | None = None,
        l1_weight: float = 0.0,
        loss: Loss | None = None,
    ):
        super().__init__()
        self._init_features(embedding, tabular_encoder, aggregation_config)
        self._init_external_embeddings(
            hidden_state_dim, proj_hiddens_to_dim, normalize_hidden_states
        )
        self._init_groups(n_groups, group_interaction)
        self.out_head = out_head or self.build_head(
            input_dim=self.representation_dim,
            output_dim=num_classes,
            hidden_dim=out_head_hidden_dim,
            dropout_p=dropout_p,
        )
        self.l1_weight = l1_weight
        self._init_loss(loss, num_classes, task_type, l1_weight)

    # -- construction --------------------------------------------------------

    def _init_features(self, embedding, encoder, aggregation_config):
        self.embedding = embedding
        self.tabular_backbone = encoder
        if embedding is None:
            self.agg_layer = None
            return
        self.agg_layer = get_aggregation_layer(
            **(aggregation_config or {"name": "mean"})
        )

    def _init_external_embeddings(
        self, hidden_state_dim, proj_hiddens_to_dim, normalize_hidden_states
    ):
        if normalize_hidden_states is not None:
            self.external_embeddings_normalize = nn.ModuleDict({
                key: nn.LayerNorm(dim) for key, dim in normalize_hidden_states.items()
            })
            hidden_state_dim = sum(normalize_hidden_states.values())
        else:
            self.external_embeddings_normalize = None

        self.hidden_state_dim = hidden_state_dim
        if hidden_state_dim is None:
            self.proj = None
        elif proj_hiddens_to_dim is not None:
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden_state_dim),
                nn.Linear(hidden_state_dim, proj_hiddens_to_dim),
            )
            self.hidden_state_dim = proj_hiddens_to_dim
        else:
            self.proj = nn.LayerNorm(hidden_state_dim)

    def _init_loss(self, loss, num_classes, task_type, l1_weight):
        """Build the loss.

        Overridden by the uplift subclass, which keeps its own attribute name
        and calling convention.
        """
        self.loss = loss or ClassificationLoss(
            num_classes=num_classes, task_type=task_type, l1_weight=l1_weight
        )

    def _init_groups(self, n_groups, group_interaction):
        self.n_groups = n_groups
        if n_groups is None:
            self.group_features = None
        else:
            self.group_features = nn.Embedding(
                num_embeddings=n_groups, embedding_dim=self.hidden_size
            )
        self.group_interaction = group_interaction or ConcatTreatmentInteraction()

    def build_head(self, input_dim, output_dim, hidden_dim, dropout_p) -> nn.Module:
        """The default head: one feed-forward block down to ``output_dim``."""
        return FeedForwardNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_p=dropout_p,
            activation=nn.SELU,
            normalization=nn.BatchNorm1d,
            **self.head_layout,
        )

    # -- shapes --------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Token width, which the extra-token embeddings have to match.

        Raises:
            ValueError: There is no embedding to take the width from, so no
                extra token can be built either.
        """
        if self.embedding is None:
            raise ValueError(
                f"{type(self).__name__} was given no embedding, so it has no "
                "feature tokens; group and treatment embeddings have nothing "
                "to sit beside. Pass an embedding, or drop n_groups."
            )
        return self.embedding.hidden_size

    @property
    def representation_dim(self) -> int:
        """Width of the vector the head sees: pooled features plus externals."""
        width = 0
        if self.embedding is not None:
            width += self.agg_layer.output_dim or self.embedding.hidden_size
        if self.hidden_state_dim is not None:
            width += self.hidden_state_dim
        return width

    # -- the forward pass, in pieces the uplift subclass reuses ---------------

    def extra_tokens(self, group: torch.LongTensor | None = None, **kwargs):
        """Tokens to insert between embedding and encoder, in order.

        Each entry is ``(tokens, interaction)``: a ``(B, 1, D)`` embedding and
        the block that decides how it meets the feature tokens. Order is part
        of the contract — it fixes the token positions the aggregation sees.
        """
        if self.group_features is None:
            return ()
        return ((self.group_features(group).unsqueeze(1), self.group_interaction),)

    def encode(self, tab_features: TabularBatch, extra_tokens=()) -> torch.Tensor:
        """Embed, insert the extra tokens, contextualise, pool."""
        embeddings = self.embedding(tab_features)  # [bs, n_features, dim]
        for tokens, interaction in extra_tokens:
            embeddings = interaction(states=embeddings, treatment_embeddings=tokens)
        return self.agg_layer(self.tabular_backbone(embeddings))

    def external_embeddings(self, tab_features: TabularBatch):
        """Normalise and project the batch's external embeddings, or ``None``.

        The batch's own dict is left alone — this used to be normalised in
        place, which quietly edited the caller's batch.
        """
        hidden_states = tab_features.hidden_states
        if hidden_states is None or self.proj is None:
            return None
        if self.external_embeddings_normalize is not None:
            hidden_states = {
                key: self.external_embeddings_normalize[key](value)
                for key, value in hidden_states.items()
            }
        return self.proj(torch.cat(tuple(hidden_states.values()), dim=1))

    def represent(self, tab_features: TabularBatch, extra_tokens=()) -> torch.Tensor:
        """One vector per record: pooled feature tokens, then late fusion."""
        external = self.external_embeddings(tab_features)
        if self.embedding is None:
            if external is None:
                raise ValueError(
                    f"{type(self).__name__} has neither an embedding nor "
                    "external embeddings, so there is nothing to score. Give "
                    "it an embedding, or a hidden_state_dim and a batch "
                    "carrying that column."
                )
            return external
        pooled = self.encode(tab_features, extra_tokens)
        if external is None:
            return pooled
        return torch.cat([pooled, external], dim=1)

    def compute_loss(self, logits, targets, **kwargs):
        """Score the head's output. Returns ``(loss, auxiliary)``."""
        result = self.loss(logits, targets, model=self.embedding)
        return result.loss, result.components.get("l1", 0)

    def forward(
        self,
        tab_features: TabularBatch,
        targets: torch.Tensor = None,
        group: torch.LongTensor = None,
        **kwargs,
    ) -> TabularOutput:
        """Score the batch, and compute the loss when targets are given.

        ``targets`` is optional because inference has no labels, and logits
        come back either way so that train metrics have something to read.
        """
        features = self.represent(tab_features, self.extra_tokens(group=group))
        logits = self.out_head(features)

        loss, auxiliary = None, None
        if targets is not None:
            loss, auxiliary = self.compute_loss(logits, targets, features=features)
        return TabularOutput(logits=logits, loss=loss, auxilary_loss=auxiliary)
