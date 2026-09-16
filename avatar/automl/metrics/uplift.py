"""Stable, dependency-local uplift and Qini metrics for binary outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import auc, roc_auc_score
from sklearn.utils.validation import check_consistent_length

from .base import MetricInput


def _binary(values, name: str) -> np.ndarray:
    result = np.asarray(values).reshape(-1)
    unique = np.unique(result)
    if not set(unique.tolist()) <= {0, 1} or len(unique) != 2:
        msg = f"{name} must contain both binary values 0 and 1; got {unique.tolist()}"
        raise ValueError(msg)
    return result.astype(np.int8, copy=False)


def _inputs(y_true, uplift, treatment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    check_consistent_length(y_true, uplift, treatment)
    target = _binary(y_true, "y_true")
    arm = _binary(treatment, "treatment")
    score = np.asarray(uplift, dtype=float).reshape(-1)
    if not np.isfinite(score).all():
        msg = "uplift must contain only finite values"
        raise ValueError(msg)
    return target, score, arm


def uplift_curve(y_true, uplift, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative uplift-curve coordinates using stable score ordering."""
    target, score, arm = _inputs(y_true, uplift, treatment)
    order = np.argsort(-score, kind="stable")
    target, score, arm = target[order], score[order], arm[order]
    distinct = np.r_[np.flatnonzero(np.diff(score)), score.size - 1]
    n = distinct + 1
    treated_n = np.cumsum(arm, dtype=np.float64)[distinct]
    control_n = n - treated_n
    treated_y = np.cumsum(target * arm, dtype=np.float64)[distinct]
    control_y = np.cumsum(target * (1 - arm), dtype=np.float64)[distinct]
    values = (
        np.divide(
            treated_y, treated_n, out=np.zeros_like(treated_y), where=treated_n != 0
        )
        - np.divide(
            control_y, control_n, out=np.zeros_like(control_y), where=control_n != 0
        )
    ) * n
    return np.r_[0, n], np.r_[0.0, values]


def perfect_uplift_curve(y_true, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Return the attainable binary-outcome uplift curve."""
    target = _binary(y_true, "y_true")
    arm = _binary(treatment, "treatment")
    control_responders = np.sum((target == 1) & (arm == 0))
    treated_nonresponders = np.sum((target == 0) & (arm == 1))
    summand = target if control_responders > treated_nonresponders else arm
    return uplift_curve(target, 2 * (target == arm) + summand, arm)


def _normalized_area(actual, perfect) -> float:
    x_actual, y_actual = actual
    x_perfect, y_perfect = perfect
    baseline = auc([0, x_perfect[-1]], [0, y_perfect[-1]])
    denominator = auc(x_perfect, y_perfect) - baseline
    if denominator == 0:
        msg = "Normalized uplift metric is undefined because the perfect-curve area is zero"
        raise ValueError(msg)
    return float((auc(x_actual, y_actual) - baseline) / denominator)


def uplift_auc_score(y_true, uplift, treatment) -> float:
    """Return normalized area under the uplift curve."""
    return _normalized_area(
        uplift_curve(y_true, uplift, treatment), perfect_uplift_curve(y_true, treatment)
    )


def qini_curve(y_true, uplift, treatment) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative Qini-curve coordinates using stable score ordering."""
    target, score, arm = _inputs(y_true, uplift, treatment)
    order = np.argsort(-score, kind="stable")
    target, score, arm = target[order], score[order], arm[order]
    distinct = np.r_[np.flatnonzero(np.diff(score)), score.size - 1]
    n = distinct + 1
    treated_n = np.cumsum(arm, dtype=np.float64)[distinct]
    control_n = n - treated_n
    treated_y = np.cumsum(target * arm, dtype=np.float64)[distinct]
    control_y = np.cumsum(target * (1 - arm), dtype=np.float64)[distinct]
    values = treated_y - control_y * np.divide(
        treated_n, control_n, out=np.zeros_like(treated_n), where=control_n != 0
    )
    return np.r_[0, n], np.r_[0.0, values]


def perfect_qini_curve(
    y_true, treatment, negative_effect: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return the attainable binary-outcome Qini curve."""
    if not isinstance(negative_effect, bool):
        msg = f"negative_effect must be bool, got {type(negative_effect).__name__}"
        raise TypeError(msg)
    target = _binary(y_true, "y_true")
    arm = _binary(treatment, "treatment")
    if negative_effect:
        return qini_curve(target, target * arm - target * (1 - arm), arm)
    random_effect = target[arm == 1].sum() - arm.sum() * target[arm == 0].mean()
    return np.array([0.0, random_effect, len(target)]), np.array([
        0.0,
        random_effect,
        random_effect,
    ])


def qini_auc_score(y_true, uplift, treatment, negative_effect: bool = True) -> float:
    """Return normalized area under the Qini curve."""
    return _normalized_area(
        qini_curve(y_true, uplift, treatment),
        perfect_qini_curve(y_true, treatment, negative_effect),
    )


def uplift_at_k(
    y_true, uplift, treatment, strategy: str = "overall", k: float = 0.3
) -> float:
    """Return observed response-rate uplift among the highest-scored rows."""
    target, score, arm = _inputs(y_true, uplift, treatment)
    if strategy not in {"overall", "by_group"}:
        msg = "strategy must be 'overall' or 'by_group'"
        raise ValueError(msg)
    order = np.argsort(-score, kind="stable")
    if isinstance(k, float | np.floating):
        if not 0 < k < 1:
            msg = "float k must lie in (0, 1)"
            raise ValueError(msg)
        if strategy == "overall":
            selected = order[: max(1, int(len(target) * k))]
            control = target[selected][arm[selected] == 0]
            treated = target[selected][arm[selected] == 1]
        else:
            control = target[order][arm[order] == 0][
                : max(1, int((arm == 0).sum() * k))
            ]
            treated = target[order][arm[order] == 1][
                : max(1, int((arm == 1).sum() * k))
            ]
    elif isinstance(k, int | np.integer) and 0 < k < len(target):
        if strategy == "overall":
            selected = order[: int(k)]
            control = target[selected][arm[selected] == 0]
            treated = target[selected][arm[selected] == 1]
        else:
            control = target[order][arm[order] == 0][: int(k)]
            treated = target[order][arm[order] == 1][: int(k)]
    else:
        msg = "integer k must be positive and smaller than the sample size"
        raise ValueError(msg)
    if not len(control) or not len(treated):
        msg = "uplift@k is undefined because the selected rows do not contain both treatment arms"
        raise ValueError(msg)
    return float(treated.mean() - control.mean())


@dataclass(frozen=True)
class UpliftMetric:
    """One registered treatment-aware uplift metric."""

    name: Literal[
        "qini_auc", "uplift_auc", "uplift_at_10", "uplift_at_20", "uplift_at_50"
    ]
    optimization_direction: Literal["maximize"] | None
    supported_tasks: frozenset[str] = frozenset({"uplift"})

    def compute(self, data: MetricInput) -> float:
        if data.treatment is None:
            msg = f"Metric {self.name!r} requires factual treatment"
            raise ValueError(msg)
        if self.name == "qini_auc":
            return qini_auc_score(data.target, data.scores, data.treatment)
        if self.name == "uplift_auc":
            return uplift_auc_score(data.target, data.scores, data.treatment)
        k = int(self.name.rsplit("_", 1)[1])
        return uplift_at_k(data.target, data.scores, data.treatment, "overall", k / 100)


@dataclass(frozen=True)
class UpliftArmRocAuc:
    """ROC AUC for one factual arm's potential-outcome scores."""

    name: Literal["treatment_roc_auc", "control_roc_auc"]
    supported_tasks: frozenset[str] = frozenset({"uplift"})
    optimization_direction: None = None

    def compute(self, data: MetricInput) -> float:
        return float(roc_auc_score(data.target, data.scores))
