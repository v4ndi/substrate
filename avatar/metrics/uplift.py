"""Uplift metrics: qini, uplift@k, and beta calibration of the two heads.

Uplift is scored on the *difference* between the treatment and control
probabilities, so both heads' outputs are needed and neither is meaningful
alone.

Calibration is optional and, when asked for, honest: a calibrator is fitted on
the ``calib`` slice and the calibrated numbers are reported only for the
held-out ``test`` slice. A population without such a slice gets no calibrated
metrics at all, because the only ones it could produce would be measured on the
rows the calibrator was fitted on.
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

#: A monotone calibrator must not reorder records, so per-head ROC AUC should
#: barely move. Above this the calibrator is distorting the scores.
RANK_SHIFT_TOLERANCE = 0.01

#: On the slice the calibrator was fitted on, the mean calibrated probability
#: should reproduce the observed conversion rate.
CALIBRATION_GAP_TOLERANCE = 1e-3


def unscorable_reason(y_true, treatment) -> str | None:
    """Why uplift metrics cannot be computed for this slice, or ``None``.

    Uplift compares the conversion of the treated against the control, so a
    slice missing either arm, or missing conversions altogether, has nothing to
    compare.
    """
    if y_true.size == 0:
        return "the slice is empty"
    if not (treatment == 1).any():
        return "the slice has no treated records"
    if not (treatment == 0).any():
        return "the slice has no control records"
    if y_true.sum() == 0:
        return "the slice has no conversions"
    return None


def calculate_uplift_metrics(
    y_true, uplift, treatment, calibrated=False
) -> dict[str, float]:
    """Uplift metrics for one slice, or an empty dict when they are undefined.

    A degenerate slice used to be given a conversion — ``y_true[0] += 1`` —
    so that the call would not raise. That turned a broken slice into a
    plausible-looking number and wrote the invented label into the data the
    later calls read. Now the slice is skipped and the reason logged: a missing
    metric is a question the reader can ask, a fabricated one is not.
    """
    reason = unscorable_reason(y_true, treatment)
    if reason is not None:
        logger.warning("uplift metrics are undefined here: %s", reason)
        return {}

    values = {"y_true": y_true, "uplift": uplift, "treatment": treatment}
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
    """Fit a beta calibrator, or return ``None`` when the slice cannot carry one.

    A calibrator maps scores onto an observed conversion rate, so a slice
    without conversions has nothing to map onto. That case used to be papered
    over by adding a conversion to the data; now it simply yields no
    calibrator, and the caller reports no calibrated metrics.
    """
    if y_true.size == 0 or y_true.sum() == 0:
        return None
    calibrator = BetaCalibration()
    calibrator.fit(scores.reshape(-1, 1), y_true)
    return calibrator


def warn_outside_tolerance(diagnostics: dict[str, float]) -> None:
    """Log the calibration diagnostics that are too far from zero."""
    for name, value in diagnostics.items():
        if name.startswith("calibration_rank_shift"):
            tolerance = RANK_SHIFT_TOLERANCE
        elif name.startswith("calibration_gap"):
            tolerance = CALIBRATION_GAP_TOLERANCE
        else:
            continue
        if value >= tolerance:
            logger.warning(
                "%s is %.4g, above the tolerance of %.4g — the calibrated uplift "
                "metrics from this slice are not trustworthy",
                name,
                value,
                tolerance,
            )


def calibration_diagnostics(
    y_true, treatment, arms, in_sample: bool
) -> dict[str, float]:
    """Report what the calibrator did to each head.

    Calibration is monotone, so it must not change the ranking, and on the
    slice it was fitted on the mean calibrated probability should reproduce the
    observed conversion rate. Both come back as **metrics**, not assertions: a
    number that says "the calibrator is off" is more useful in MLflow than a
    traceback, and a metric has no business stopping training.

    Args:
        y_true: Conversion of every record in the slice.
        treatment: Arm of every record in the slice.
        arms: ``(name, arm value, raw probabilities, calibrated probabilities)``
            per head.
        in_sample: Whether this is the slice the calibrator was fitted on. The
            conversion-rate gap is only meaningful there.
    """
    diagnostics: dict[str, float] = {}
    for name, arm, raw_probs, calibrated_probs in arms:
        in_arm = treatment == arm
        labels = y_true[in_arm]
        if labels.size == 0 or labels.min() == labels.max():
            logger.warning(
                "calibration diagnostics skipped for the %s head: the slice has "
                "only one class",
                name,
            )
            continue
        raw_auc = roc_auc_score(labels, raw_probs[in_arm])
        calibrated_auc = roc_auc_score(labels, calibrated_probs[in_arm])
        diagnostics[f"{name}_roc_auc_score"] = float(calibrated_auc)
        diagnostics[f"calibration_rank_shift_{name}"] = float(
            abs(calibrated_auc - raw_auc)
        )
        if in_sample:
            diagnostics[f"calibration_gap_{name}"] = float(
                abs(calibrated_probs[in_arm].mean() - labels.mean())
            )

    warn_outside_tolerance(diagnostics)
    return diagnostics


class UpliftMetrics(GroupedPredictionMetric):
    """Campaign uplift metrics, optionally over beta-calibrated probabilities.

    Consumes ``outputs.uplift`` (or ``treatment_probs - control_probs`` when
    ``uplift`` is None), ``outputs.treatment``, ``outputs.conversion`` and
    ``outputs.group``. The population is split by ``group`` and by
    ``split_type``.

    Calibration, when requested, is fitted on ``calib`` and **reported only on
    ``test``**. Fitting and scoring on the same rows is what in-sample means,
    and the numbers it produces flatter the model; a population with no
    ``test`` slice — which includes every batch that carries no ``split_type``
    column at all — therefore gets the raw metrics only, and a warning.

    Args:
        require_calibration: Fit the calibrators and report the calibrated
            metrics. With no held-out slice to report them on, nothing
            calibrated is produced and a warning is logged.
        save_submit_path: Directory to write per-record predictions into.
            ``None`` (the default) writes nothing.
        main_metric: Which metric counts as *the* number for this run. Must be
            one of ``uplift_at_{5,10,15,20,50}``, ``uplift_auc_score`` or
            ``qini_auc_score``.

    Returns from :meth:`compute`:
        Per group and split, the raw uplift metrics; on ``test``, additionally
        the calibrated ones; and, when calibration ran, the per-head ROC AUC
        with the diagnostics ``calibration_rank_shift_{control,treatment}``
        and — on ``calib`` only — ``calibration_gap_{control,treatment}``. A
        rank shift far from zero means the calibrated numbers should not be
        read.

        The two summaries are independent: ``mean_{main_metric}`` averages the
        raw metric over the groups, ``mean_calibrated_{main_metric}`` the
        calibrated one. They used to be the same number written twice.

    Note:
        A slice with no conversions, or with only one arm, produces no metrics
        for that slice and logs why. Nothing is invented to keep the keys
        present.
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
        self.calibrated_mains: list[float] = []

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
        """Start a fresh calibrated summary, and make room for its column."""
        self.calibrated_mains = []
        if self.require_calibration:
            merged["calibrated_uplift"] = np.full_like(
                merged["y_true"], np.nan, dtype=float
            )

    def _fit_calibrators(self, merged: dict, group, calib_mask: np.ndarray):
        """Fit one calibrator per head on the ``calib`` slice of this group."""
        calibrators = []
        for name, arm, column in (
            ("treatment", 1, "t_probs"),
            ("control", 0, "c_probs"),
        ):
            in_arm = calib_mask & (merged["treatment"] == arm)
            calibrator = fit_calibrator(
                scores=merged[column][in_arm], y_true=merged["y_true"][in_arm]
            )
            if calibrator is None:
                logger.warning(
                    "group %s: no calibrator could be fitted for the %s head — its "
                    "calib slice has no conversions; this group reports raw uplift "
                    "metrics only",
                    group,
                    name,
                )
                return None, None
            calibrators.append(calibrator)
        return calibrators[0], calibrators[1]

    def score_group(self, merged, group, calib_mask, test_mask):
        """Score both slices; calibrated numbers only where they are honest."""
        t_calibrator, c_calibrator = (None, None)
        if self.require_calibration:
            t_calibrator, c_calibrator = self._fit_calibrators(
                merged, group, calib_mask
            )
        calibrated = t_calibrator is not None and c_calibrator is not None

        scores: dict[str, float] = {}
        main_value = None
        for split, mask in ((CALIB_SPLIT, calib_mask), (TEST_SPLIT, test_mask)):
            if not mask.any():
                continue
            y_true = merged["y_true"][mask]
            treatment = merged["treatment"][mask]
            t_probs = merged["t_probs"][mask]
            c_probs = merged["c_probs"][mask]

            slice_scores = calculate_uplift_metrics(
                y_true=y_true, uplift=t_probs - c_probs, treatment=treatment
            )
            if slice_scores:
                # ``test`` comes second, so a held-out slice overrides the
                # in-sample one as the group's contribution to the mean.
                main_value = slice_scores[self.main_metric]

            if calibrated:
                t_calibrated = t_calibrator.predict(t_probs.reshape(-1, 1))
                c_calibrated = c_calibrator.predict(c_probs.reshape(-1, 1))
                merged["calibrated_uplift"][mask] = t_calibrated - c_calibrated
                in_sample = split == CALIB_SPLIT

                slice_scores.update(
                    calibration_diagnostics(
                        y_true,
                        treatment,
                        [
                            ("treatment", 1, t_probs, t_calibrated),
                            ("control", 0, c_probs, c_calibrated),
                        ],
                        in_sample=in_sample,
                    )
                )

                if not in_sample:
                    calibrated_scores = calculate_uplift_metrics(
                        y_true=y_true,
                        uplift=t_calibrated - c_calibrated,
                        treatment=treatment,
                        calibrated=True,
                    )
                    slice_scores.update(calibrated_scores)
                    if calibrated_scores:
                        self.calibrated_mains.append(
                            calibrated_scores[f"calibrated_{self.main_metric}"]
                        )

            scores.update(group_prefixed(slice_scores, split, group))
        return scores, main_value

    def aggregate(self, result: dict, main_values: list[float]) -> None:
        """Summarise the raw and the calibrated families separately."""
        super().aggregate(result, main_values)
        if self.calibrated_mains:
            result[f"mean_calibrated_{self.main_metric}"] = float(
                np.mean(self.calibrated_mains)
            )
        elif self.require_calibration:
            logger.warning(
                "require_calibration is set, but no calibrated uplift metric could "
                "be produced. They are reported only on a held-out `test` slice, "
                "and this population has none — add a split_type column marking "
                "the rows the calibrator must not be scored on."
            )
