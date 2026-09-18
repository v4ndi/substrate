"""The bridge between the two metric worlds.

``Trainer`` ranks with :mod:`fmlib.metrics` objects; AutoML ranks with its own
registry (``resolve_metric(...).compute(MetricInput)``). :class:`AutoMLMetric`
joins them by **composition**: it holds the resolved AutoML metric, accumulates
targets and scores in ``update``, and returns ``{name: value}`` for the
configured ``optimization_metric`` in ``compute``.

Composition, not inheritance, because the AutoML metric classes must not learn
about torch: ``Metric`` there is a structural ``Protocol``, the registry holds
instances, and ``fmlib/automl/**`` outside this package does not import torch.
This module does -- it runs inside the training loop -- which is why the task
layer loads it only when the backend family is actually TabNN.

Early stopping is configured with the same name, so the number the loop stops
on is the number AutoML later reports.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from fmlib.automl.metrics import MetricInput, resolve_metric
from fmlib.metrics.base import ScalarMetric

__all__ = ["AutoMLMetric", "to_scores"]

ScoreTransform = Literal["sigmoid", "softmax", "identity"]


def to_scores(logits: torch.Tensor, transform: ScoreTransform) -> np.ndarray:
    """Turn a head's logits into the scores AutoML metrics are defined on.

    Three branches, one per task shape:

    * ``sigmoid`` -- binary and response: ``(B, 1)`` logits become ``(B,)``
      probabilities;
    * ``softmax`` -- multiclass: ``(B, K)`` logits become a ``(B, K)``
      probability matrix, columns in training class order;
    * ``identity`` -- regression: ``(B, 1)`` predictions become ``(B,)``.

    Args:
        logits: Raw head output.
        transform: Which of the three shapes this task has.

    Returns:
        A detached float64 numpy array on the host.

    Raises:
        ValueError: If ``transform`` is not one of the three.
    """
    values = logits.detach().float()
    if transform == "sigmoid":
        return torch.sigmoid(values).reshape(-1).cpu().numpy().astype(np.float64)
    if transform == "softmax":
        return torch.softmax(values, dim=-1).cpu().numpy().astype(np.float64)
    if transform == "identity":
        return values.reshape(-1).cpu().numpy().astype(np.float64)
    msg = (
        f"Unknown score transform {transform!r}; expected sigmoid, softmax or identity"
    )
    raise ValueError(msg)


class AutoMLMetric(ScalarMetric):
    """Score a validation pass with one registered AutoML metric.

    Args:
        metric_name: Registered AutoML metric, the task's ``optimization_metric``.
        task_name: AutoML task the metric has to be valid for.
        score_transform: How this task's logits become scores.
        class_order: Multiclass label order, in training id order.
        target_key: Key the collate function puts the target under.

    Raises:
        ConfigError: If the metric is unknown, or not usable for optimization
            on this task. Raised by the registry, at construction time, which
            is where a misconfigured run should fail.
    """

    #: ROC-AUC and Qini are not averages over batches, so a distributed run has
    #: to gather the whole population before computing anything.
    needs_full_population = True

    #: Declaring both sides is what lets a distributed run gather two tensors
    #: instead of whole batches.
    required_outputs = ("logits",)

    def __init__(
        self,
        metric_name: str,
        task_name: str,
        score_transform: ScoreTransform = "sigmoid",
        class_order: tuple[Any, ...] | None = None,
        target_key: str = "targets",
    ):
        self.metric = resolve_metric(metric_name, task_name, "optimization")
        self.task_name = task_name
        self.score_transform = score_transform
        self.class_order = tuple(class_order) if class_order is not None else None
        self.target_key = target_key
        self.required_inputs = (target_key,)
        self._targets: list[np.ndarray] = []
        self._scores: list[np.ndarray] = []

    @property
    def name(self) -> str:
        """The metric name, which is also what early stopping is told to watch."""
        return self.metric.name

    @property
    def direction(self) -> str:
        """``max`` or ``min``, in the vocabulary :class:`EarlyStopping` speaks."""
        return "max" if self.metric.optimization_direction == "maximize" else "min"

    def update(self, inputs, outputs) -> None:
        target = inputs[self.target_key]
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        self._targets.append(np.asarray(target).reshape(-1))
        self._scores.append(to_scores(outputs.logits, self.score_transform))

    def compute(self) -> dict[str, float]:
        if not self._targets:
            return {}
        value = self.metric.compute(
            MetricInput(
                target=np.concatenate(self._targets),
                scores=np.concatenate(self._scores, axis=0),
                class_order=self.class_order,
            )
        )
        return {} if value is None else {self.name: float(value)}

    def reset(self) -> None:
        self._targets = []
        self._scores = []
