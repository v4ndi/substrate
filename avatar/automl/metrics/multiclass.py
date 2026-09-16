"""Metrics for multiclass probability models."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from avatar.automl.exceptions import ConfigError, SchemaError

from .base import MetricInput
from .binary import binary_top_k_metrics

logger = logging.getLogger(__name__)


def multiclass_objective(
    name: str,
    target: np.ndarray,
    probabilities: np.ndarray,
    class_order: Sequence[object],
) -> float:
    """Return one configured multiclass optimization metric."""
    labels = np.arange(probabilities.shape[1])
    if name == "roc_auc_ovr_macro":
        try:
            return float(roc_auc_score(target, probabilities, labels=labels, multi_class="ovr", average="macro"))
        except ValueError as exc:
            msg = f"roc_auc_ovr_macro is undefined for the validation split: {exc}"
            raise SchemaError(msg) from exc
    predicted = probabilities.argmax(axis=1)
    if name == "accuracy":
        return float(accuracy_score(target, predicted))
    if name == "f1_macro":
        return float(f1_score(target, predicted, average="macro", labels=labels))
    if name == "log_loss":
        return float(log_loss(target, probabilities, labels=labels))
    msg = f"Unsupported multiclass metric: {name!r}"
    raise ConfigError(msg)


def multiclass_class_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    class_order: Sequence[object],
    *,
    grouped: bool,
) -> dict[str, float | int | None]:
    """Return class-vs-rest ROC AUC and top-k metrics."""
    metrics: dict[str, float | int | None] = {}
    for index in range(probabilities.shape[1]):
        binary_target = (target == index).astype(np.int8)
        if np.unique(binary_target).size != 2:
            if grouped:
                logger.warning(
                    "Grouped class-wise ROC AUC is undefined and will be null; class_index=%d, class_label=%r",
                    index,
                    class_order[index],
                )
            class_auc = None
        else:
            class_auc = float(roc_auc_score(binary_target, probabilities[:, index]))
        metrics[f"roc_auc_class_{index}"] = class_auc
        metrics[f"n_positives_class_{index}"] = int(binary_target.sum())
        metrics.update(
            {
                f"{name}_class_{index}": value
                for name, value in binary_top_k_metrics(binary_target, probabilities[:, index]).items()
            }
        )
    return metrics


def multiclass_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    class_order: Sequence[object],
    *,
    grouped: bool,
) -> dict[str, float | int | None]:
    """Return aggregate and class-wise multiclass evaluation metrics."""
    labels = np.arange(probabilities.shape[1])
    predicted = probabilities.argmax(axis=1)
    missing_classes = sorted(set(labels.tolist()) - set(np.unique(target).tolist()))
    if missing_classes and not grouped:
        missing_labels = [class_order[index] for index in missing_classes]
        msg = f"Overall multiclass ROC AUC is undefined because target classes are missing: {missing_labels!r}"
        raise SchemaError(msg)
    if missing_classes:
        missing_labels = [class_order[index] for index in missing_classes]
        logger.warning("Grouped multiclass ROC AUC is undefined and will be null; missing classes: %s", missing_labels)
        auc = None
    else:
        try:
            auc = float(roc_auc_score(target, probabilities, labels=labels, multi_class="ovr", average="macro"))
        except ValueError as exc:
            if not grouped:
                msg = f"Overall multiclass ROC AUC is undefined: {exc}"
                raise SchemaError(msg) from exc
            logger.warning("Grouped multiclass ROC AUC is undefined and will be null: %s", exc)
            auc = None
    return {
        "roc_auc_ovr_macro": auc,
        "log_loss": float(log_loss(target, probabilities, labels=labels)),
        "accuracy": float(accuracy_score(target, predicted)),
        "f1_macro": float(f1_score(target, predicted, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(target, predicted, average="weighted", labels=labels, zero_division=0)),
    } | multiclass_class_metrics(target, probabilities, class_order, grouped=grouped)


@dataclass(frozen=True)
class MulticlassMetric:
    """One registered aggregate Multiclass metric."""

    name: Literal["roc_auc_ovr_macro", "log_loss", "accuracy", "f1_macro", "f1_weighted"]
    optimization_direction: Literal["maximize", "minimize"] | None
    supported_tasks: frozenset[str] = frozenset({"multiclass"})

    def compute(self, data: MetricInput) -> float:
        class_order = data.class_order or tuple(range(data.scores.shape[1]))
        if self.name == "f1_weighted":
            labels = np.arange(data.scores.shape[1])
            predicted = data.scores.argmax(axis=1)
            return float(f1_score(data.target, predicted, average="weighted", labels=labels, zero_division=0))
        return multiclass_objective(self.name, data.target, data.scores, class_order)
