"""Classification / regression loss with optional L1 weight regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn

from fmlib.losses.base import Loss
from fmlib.losses.regularization import L1RegularizationLoss
from fmlib.outputs import LossOutput

__all__ = ["ClassificationLoss", "build_task_loss_fn"]


def build_task_loss_fn(num_classes: int, task_type: str) -> nn.Module:
    """Pick the criterion implied by ``(num_classes, task_type)``.

    Args:
        num_classes: Size of the output head. 1 means binary or regression.
        task_type: ``"classification"`` or ``"regression"``.

    Raises:
        ValueError: For combinations that do not describe a real task.
    """
    if num_classes == 1 and task_type == "regression":
        return nn.MSELoss()
    if num_classes == 1 and task_type == "classification":
        return nn.BCEWithLogitsLoss()
    if num_classes > 1 and task_type == "classification":
        return nn.CrossEntropyLoss()
    raise ValueError(
        f"Invalid combination: task_type={task_type}, num_classes={num_classes}. "
        "Supported combinations:\n"
        "- num_classes=1 with task_type='regression'\n"
        "- num_classes=1 with task_type='classification'\n"
        "- num_classes>1 with task_type='classification'"
    )


class ClassificationLoss(Loss):
    """Task criterion plus an optional L1 penalty on the encoder's weights.

    Args:
        num_classes: Output head size, used to pick the default criterion.
        task_type: ``"classification"`` or ``"regression"``.
        loss_fn: Explicit criterion; overrides the ``(num_classes, task_type)``
            choice. This is the injection point — a focal or weighted loss goes
            here as a config change rather than a code change.
        l1_weight: Multiplier for the L1 penalty. Zero disables it entirely.
        l1_apply_substr: Only parameters whose name contains this substring are
            penalised; the default targets embedding tables.
    """

    def __init__(
        self,
        num_classes: int = 1,
        task_type: str = "classification",
        loss_fn: nn.Module | None = None,
        l1_weight: float = 0.0,
        l1_apply_substr: str = "embed",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else build_task_loss_fn(num_classes, task_type)
        )
        self.l1_weight = l1_weight
        self.l1_loss = L1RegularizationLoss(apply_substr=l1_apply_substr)

    def align(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make the head's output and the target agree on what one record is.

        A head of width one predicts a single number per record, so the two
        sides mean the same thing shaped ``(B,)``. They do not arrive that way:
        the head emits ``(B, 1)`` and the collate function emits ``(B,)``, and
        the criterion is left to reconcile them. ``BCEWithLogitsLoss`` refuses
        to — which is why binary classification could not run at all — and
        ``MSELoss`` silently broadcasts the pair into ``(B, B)``, averaging the
        error of every prediction against every *other* record's target. That
        loss trains the head towards the mean of the batch, and the only sign
        of it is a warning from torch.

        A wide head is left alone: there ``(B, K)`` logits against ``(B,)``
        class indices is exactly what cross-entropy wants.

        Raises:
            ValueError: The two still do not line up, which means the batch and
                the configured ``num_classes`` disagree about the task.
        """
        if targets.ndim == logits.ndim and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)

        if self.num_classes != 1:
            return logits, targets

        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        # One number per record: a class index is no longer an index here, it
        # is the value being predicted, and both criteria want it as a float.
        targets = targets.to(logits.dtype)

        if logits.shape != targets.shape:
            raise ValueError(
                f"a head of width 1 produced logits {tuple(logits.shape)} but "
                f"the targets are {tuple(targets.shape)}. One value per record "
                "is expected on both sides; check that num_classes matches the "
                "target column."
            )
        return logits, targets

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> LossOutput:
        """Score ``logits`` against ``targets``.

        Args:
            logits: Raw head outputs.
            targets: Ground truth. Shapes are reconciled by :meth:`align`
                first, so a ``(B,)`` target column works against a ``(B, 1)``
                head.
            model: Module to regularise. Required when ``l1_weight`` is set;
                the penalty walks its ``named_parameters``.
        """
        logits, targets = self.align(logits, targets)
        loss = self.loss_fn(logits, targets)
        components: dict[str, torch.Tensor] = {}
        if self.l1_weight > 0:
            if model is None:
                raise ValueError(
                    "ClassificationLoss has l1_weight > 0 but no model to "
                    "regularise; pass the module whose weights to penalise."
                )
            l1 = self.l1_loss(model)
            components["l1"] = l1
            loss = loss + l1 * self.l1_weight
        return LossOutput(loss=loss, components=components)
