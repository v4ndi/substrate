"""Uplift metrics: qini, uplift@k, and beta calibration of the two heads.

Uplift is scored on the *difference* between the treatment and control
probabilities, so both heads' outputs are needed and neither is meaningful
alone. Calibration is optional but usually wanted: uplift@k is sensitive to how
well the two probabilities are scaled against each other, not just to their
ranking.
"""

import logging

import numpy as np
from betacal import BetaCalibration
from sklearn.metrics import roc_auc_score
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

from avatar.metrics.grouped import (
    CALIB_SPLIT,
    TEST_SPLIT,
    GroupedPredictionMetric,
    group_prefixed,
)

logger = logging.getLogger(__name__)


def calculate_uplift_metrics(
    y_true, uplift, treatment, calibrated=False
) -> dict[str, float]:
    """Computes uplift metrics."""
    values = {"y_true": y_true, "uplift": uplift, "treatment": treatment}
    if y_true.sum() == 0:
        y_true[0] += 1
    metrics = {
        f"uplift_at_{k}": uplift_at_k(**values, strategy="overall", k=k / 100)
        for k in [5, 10, 15, 20, 50]
    } | {
        "uplift_auc_score": uplift_auc_score(**values),
        "qini_auc_score": qini_auc_score(**values),
    }

    if calibrated:
        metrics = {f"calibrated_{k}": v for k, v in metrics.items()}

    return metrics


def fit_calibrator(scores, y_true):
    """Fit a beta calibrator mapping raw scores to calibrated probabilities.

    A single positive is injected when the slice has none, so calibration on a
    degenerate validation slice does not raise.
    """
    calibrator = BetaCalibration()
    if y_true.sum() == 0:
        y_true[0] += 1
    calibrator.fit(scores.reshape(-1, 1), y_true)
    return calibrator


#: A monotone calibrator must not reorder records, so per-head ROC AUC should
#: barely move. Above this the calibrator is distorting the scores.
RANK_SHIFT_TOLERANCE = 0.01

#: On the slice the calibrator was fitted on, the mean calibrated probability
#: should reproduce the observed conversion rate.
CALIBRATION_GAP_TOLERANCE = 1e-3


def apply_calibration_calculate_metrics(
    y_true, t_probs, c_probs, treatment, t_calibrator, c_calibrator, is_calib
):
    """Report uplift metrics both raw and calibrated, and the per-head ROC AUC.

    Calibration is monotone, so it must not change the ranking, and on the slice
    it was fitted on the calibrated mean probability should reproduce the
    observed conversion rate. Both are **reported as metrics**, not asserted:
    ``calibration_rank_shift_*`` and ``calibration_gap_*`` come back in the
    result and a warning is logged when either exceeds its tolerance.

    They used to be assertions, and an under-trained model — whose near-constant
    scores make ROC AUC noisy — killed the whole training run from inside
    validation. A metric has no business stopping training: its job is to report
    a number, and a number that says "the calibrator is off" is more useful in
    MLflow than a traceback.

    Returns:
        ``(metrics, calibrated_uplift)``.
    """
    no_calib_metrics = calculate_uplift_metrics(
        y_true=y_true, uplift=(t_probs - c_probs), treatment=treatment, calibrated=False
    )

    c_roc_auc = roc_auc_score(y_true[treatment == 0], c_probs[treatment == 0])
    t_roc_auc = roc_auc_score(y_true[treatment == 1], t_probs[treatment == 1])

    t_probs_ = t_calibrator.predict(t_probs.reshape(-1, 1))
    c_probs_ = c_calibrator.predict(c_probs.reshape(-1, 1))

    calib_c_roc_auc = roc_auc_score(y_true[treatment == 0], c_probs_[treatment == 0])
    calib_t_roc_auc = roc_auc_score(y_true[treatment == 1], t_probs_[treatment == 1])

    diagnostics = {
        "calibration_rank_shift_control": float(abs(calib_c_roc_auc - c_roc_auc)),
        "calibration_rank_shift_treatment": float(abs(calib_t_roc_auc - t_roc_auc)),
    }
    if is_calib:
        diagnostics["calibration_gap_control"] = float(
            abs(c_probs_[treatment == 0].mean() - y_true[treatment == 0].mean())
        )
        diagnostics["calibration_gap_treatment"] = float(
            abs(t_probs_[treatment == 1].mean() - y_true[treatment == 1].mean())
        )

    for name, value in diagnostics.items():
        tolerance = (
            RANK_SHIFT_TOLERANCE
            if name.startswith("calibration_rank_shift")
            else CALIBRATION_GAP_TOLERANCE
        )
        if value >= tolerance:
            logger.warning(
                "%s is %.4g, above the tolerance of %.4g — the calibrated uplift "
                "metrics from this slice are not trustworthy",
                name,
                value,
                tolerance,
            )

    calibrated_uplift = t_probs_ - c_probs_

    calib_metrics = calculate_uplift_metrics(
        y_true=y_true, uplift=calibrated_uplift, treatment=treatment, calibrated=True
    )
    metrics = {**no_calib_metrics, **calib_metrics, **diagnostics}
    metrics["treatment_roc_auc_score"] = calib_t_roc_auc
    metrics["control_roc_auc_score"] = calib_c_roc_auc
    return metrics, calibrated_uplift


class UpliftMetrics(GroupedPredictionMetric):
    """Campaign uplift metrics, optionally over beta-calibrated probabilities.

    Consumes ``outputs.uplift`` (or ``treatment_probs - control_probs`` when
    ``uplift`` is None), ``outputs.treatment``, ``outputs.conversion`` and
    ``outputs.group``. The population is split by ``group`` and by
    ``split_type``: calibrators are fitted on the ``calib`` slice and the
    ``test`` slice, when present, is what ``mean_{main_metric}`` reports.

    Args:
        require_calibration: Which of the two metric families feeds
            ``mean_{main_metric}`` — the calibrated one or the raw one.
            Calibrators are fitted either way.
        save_submit_path: Directory to write per-record predictions into.
            ``None`` (the default) writes nothing.
        main_metric: Which metric counts as *the* number for this run. Must be
            one of ``uplift_at_{5,10,15,20,50}``, ``uplift_auc_score`` or
            ``qini_auc_score``.

    Returns from :meth:`compute`:
        Per group, the raw and calibrated uplift metrics, the per-head ROC AUC,
        and the calibration diagnostics
        ``calibration_rank_shift_{control,treatment}`` and
        ``calibration_gap_{control,treatment}``. A rank shift far from zero
        means the calibrated numbers should not be read.

    Note:
        A validation slice with no conversions at all would make the uplift
        metrics undefined, so a single positive is injected rather than
        raising. Metrics from such a slice are meaningless — check the data
        before reading them.
    """

    required_inputs = ("epk_id", "split_type")
    required_outputs = (
        "uplift",
        "treatment",
        "conversion",
        "control_probs",
        "treatment_probs",
        "group",
    )

    def __init__(
        self,
        require_calibration: bool = False,
        save_submit_path: str | None = None,
        main_metric: str = "qini_auc_score",
    ):
        super().__init__(save_submit_path=save_submit_path, main_metric=main_metric)
        self.require_calibration = require_calibration

    def collect(self, inputs, outputs) -> dict:
        """Keep both heads' probabilities, the assignment and the conversion."""
        if outputs.uplift is None:
            uplift = (
                outputs.treatment_probs.detach().contiguous().cpu().numpy()
                - outputs.control_probs.detach().contiguous().cpu().numpy()
            )
        else:
            uplift = outputs.uplift.detach().contiguous().cpu().numpy()

        return {
            "epk_id": inputs["epk_id"] if "epk_id" in inputs else None,
            "uplift": uplift,
            "group": outputs.group.detach().contiguous().cpu().numpy()
            if outputs.group is not None
            else None,
            "split_type": inputs["split_type"] if "split_type" in inputs else None,
            "treatment": outputs.treatment.detach().contiguous().cpu().numpy(),
            "y_true": outputs.conversion.detach().contiguous().cpu().numpy(),
            "c_probs": outputs.control_probs.detach().contiguous().cpu().numpy(),
            "t_probs": outputs.treatment_probs.detach().contiguous().cpu().numpy(),
        }

    def prepare(self, merged: dict) -> None:
        """Make room for the calibrated uplift the group loop fills in."""
        merged["calibrated_uplift"] = np.full_like(
            merged["y_true"], np.nan, dtype=float
        )

    def score_group(self, merged, group, calib_mask, test_mask):
        """Fit the two calibrators on ``calib``, then score both slices."""
        t_calibrator = fit_calibrator(
            scores=merged["t_probs"][calib_mask & (merged["treatment"] == 1)],
            y_true=merged["y_true"][calib_mask & (merged["treatment"] == 1)],
        )
        c_calibrator = fit_calibrator(
            scores=merged["c_probs"][calib_mask & (merged["treatment"] == 0)],
            y_true=merged["y_true"][calib_mask & (merged["treatment"] == 0)],
        )

        main_metric = (
            f"calibrated_{self.main_metric}"
            if self.require_calibration
            else self.main_metric
        )

        scores: dict[str, float] = {}
        main_value = None
        for split, mask in ((CALIB_SPLIT, calib_mask), (TEST_SPLIT, test_mask)):
            if not mask.any():
                continue
            slice_scores, calibrated_uplift = apply_calibration_calculate_metrics(
                y_true=merged["y_true"][mask],
                t_probs=merged["t_probs"][mask],
                c_probs=merged["c_probs"][mask],
                treatment=merged["treatment"][mask],
                t_calibrator=t_calibrator,
                c_calibrator=c_calibrator,
                is_calib=split == CALIB_SPLIT,
            )
            merged["calibrated_uplift"][mask] = calibrated_uplift
            scores.update(group_prefixed(slice_scores, split, group))
            # ``test`` comes second, so a held-out slice overrides the
            # in-sample one as the group's contribution to the mean.
            main_value = slice_scores[main_metric]
        return scores, main_value

    def aggregate(self, result: dict, main_values: list[float]) -> None:
        """Report the mean under both names when calibration is required.

        ``mean_calibrated_{main_metric}`` is an alias, not a second number: the
        per-group values were already taken from the calibrated family.
        """
        super().aggregate(result, main_values)
        if self.require_calibration and main_values:
            result[f"mean_calibrated_{self.main_metric}"] = result[
                f"mean_{self.main_metric}"
            ]
