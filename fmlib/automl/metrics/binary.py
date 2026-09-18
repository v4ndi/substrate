"""Metrics for binary response models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score

from .base import MetricInput

DEFAULT_TOP_K: tuple[int, ...] = (5, 10, 20, 25, 50)


def binary_roc_auc(target: np.ndarray, scores: np.ndarray) -> float:
    """Return the area under the binary receiver-operating curve."""
    return float(roc_auc_score(target, scores))


def binary_top_k_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    ks: Sequence[int] = DEFAULT_TOP_K,
) -> dict[str, float]:
    """Return precision and recall within the highest-scored percentages."""
    order = np.argsort(-scores, kind="stable")
    positive_count = int(target.sum())
    metrics: dict[str, float] = {}
    for k in ks:
        cutoff = max(1, int(np.ceil(target.size * k / 100)))
        true_positives = int(target[order[:cutoff]].sum())
        metrics[f"precision@{k}"] = true_positives / cutoff
        metrics[f"recall@{k}"] = (
            true_positives / positive_count if positive_count else float("nan")
        )
    return metrics


@dataclass(frozen=True)
class BinaryRocAuc:
    """ROC AUC for Binary and Response optimization and evaluation."""

    name: str = "roc_auc"
    supported_tasks: frozenset[str] = frozenset({"binary", "response"})
    optimization_direction: Literal["maximize"] = "maximize"

    def compute(self, data: MetricInput) -> float:
        return binary_roc_auc(data.target, data.scores)


@dataclass(frozen=True)
class BinaryTopK:
    """One fixed precision@k or recall@k evaluation metric."""

    kind: Literal["precision", "recall"]
    k: int
    supported_tasks: frozenset[str] = frozenset({"binary", "response", "multiclass"})
    optimization_direction: None = None

    @property
    def name(self) -> str:
        return f"{self.kind}@{self.k}"

    def compute(self, data: MetricInput) -> float:
        return binary_top_k_metrics(data.target, data.scores, (self.k,))[self.name]


@dataclass(frozen=True)
class ClassWiseRocAuc:
    """One-vs-rest ROC AUC selected for every Multiclass class."""

    name: str = "class_wise_roc_auc"
    supported_tasks: frozenset[str] = frozenset({"multiclass"})
    optimization_direction: None = None

    def compute(self, data: MetricInput) -> float:
        return binary_roc_auc(data.target, data.scores)
