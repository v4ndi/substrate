"""The loss contract.

Losses are ``nn.Module``s owned by the task-pipeline, never by the backbone and
never by the trainer:

    trainer  --calls-->  pipeline.forward(batch)  -->  output (has .loss)
                              |
                              +- backbone(...)          # fmlib.nn.*, no loss
                              +- self.loss(...)         # fmlib.losses.*

Keeping the loss out of the backbone is what lets the same encoder run
inference with no targets; keeping it out of the trainer is what lets the
trainer stay ignorant of every task's target semantics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fmlib.outputs import LossOutput

__all__ = ["CompositeLoss", "Loss", "LossOutput"]


class Loss(nn.Module):
    """Base class for losses.

    Subclasses take whatever their task needs — logits and targets, or a whole
    output object — and must return a :class:`~fmlib.outputs.LossOutput`.
    That return type is the entire contract; the argument list is the
    pipeline's business, since the pipeline is the only place that knows what
    the targets mean.
    """

    def forward(self, *args, **kwargs) -> LossOutput:  # pragma: no cover - abstract
        raise NotImplementedError


class CompositeLoss(Loss):
    """A weighted sum of named sub-losses.

    Args:
        losses: Mapping of name to sub-loss.
        weights: Per-name multipliers; missing names default to 1.0.

    Every sub-loss receives the same arguments, and each one's terms are folded
    into the result under its own name, so a composite of composites still
    yields a flat, readable component dict.
    """

    def __init__(
        self,
        losses: dict[str, Loss],
        weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = {name: 1.0 for name in losses}
        if weights:
            unknown = set(weights) - set(losses)
            if unknown:
                raise ValueError(f"weights for unknown losses: {sorted(unknown)}")
            self.weights.update(weights)

    def forward(self, *args, **kwargs) -> LossOutput:
        total: torch.Tensor | float = 0.0
        components: dict[str, torch.Tensor] = {}
        for name, loss in self.losses.items():
            result = loss(*args, **kwargs)
            weight = self.weights[name]
            components[name] = result.loss
            for key, value in result.components.items():
                components[f"{name}/{key}"] = value
            total = total + weight * result.loss
        return LossOutput(loss=total, components=components)
