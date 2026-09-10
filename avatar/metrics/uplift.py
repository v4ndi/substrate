"""Uplift metrics: qini, uplift@k, and beta calibration of the two heads.

Uplift is scored on the *difference* between the treatment and control
probabilities, so both heads' outputs are needed and neither is meaningful
alone. Calibration is optional but usually wanted: uplift@k is sensitive to how
well the two probabilities are scaled against each other, not just to their
ranking.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
from betacal import BetaCalibration
from sklearn.metrics import roc_auc_score
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

from avatar.metrics.base import ScalarMetric

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


class UpliftMetrics(ScalarMetric):
    """Campaign uplift metrics, optionally over beta-calibrated probabilities.

    Consumes ``outputs.uplift`` (or ``treatment_probs - control_probs`` when
    ``uplift`` is None), ``outputs.treatment``, ``outputs.conversion`` and
    ``outputs.group``. When the batch carries a task or ``product`` column,
    metrics are additionally reported per task.

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
        Per task and group, the raw and calibrated uplift metrics, the per-head
        ROC AUC, and the calibration diagnostics
        ``calibration_rank_shift_{control,treatment}`` and
        ``calibration_gap_{control,treatment}``. A rank shift far from zero
        means the calibrated numbers should not be read.

    Note:
        A validation slice with no conversions at all would make the uplift
        metrics undefined, so a single positive is injected rather than
        raising. Metrics from such a slice are meaningless — check the data
        before reading them.
    """

    required_inputs = ("epk_id", "split_type", "product")
    required_outputs = (
        "uplift",
        "treatment",
        "conversion",
        "control_probs",
        "treatment_probs",
        "group",
        "task_name",
    )

    def __init__(
        self,
        require_calibration: bool = False,
        save_submit_path: str | None = None,
        main_metric: str = "qini_auc_score",
    ):
        self.preds = []
        self.require_calibration = require_calibration
        self.save_submit = save_submit_path
        self.main_metric = main_metric

    def update(self, inputs, outputs):
        """Stores model predictions for later computation.

        Args:
            inputs: Dict[str, any] - epk_id - optional field
            outputs: Dict[str, torch.Tensor] - uplift, treatment, conversion,
                control_probs, treatment_probs, group required fields;
                each field is a torch.Tensor with dim = 1
        """
        if outputs.uplift is None:
            uplift = (
                outputs.treatment_probs.detach().contiguous().cpu().numpy()
                - outputs.control_probs.detach().contiguous().cpu().numpy()
            )
        else:
            uplift = None

        if hasattr(outputs, "task_name") and outputs.task_name is not None:
            if torch.is_tensor(outputs.task_name):
                task_name = outputs.task_name.detach().contiguous().cpu().numpy()
            else:
                task_name = outputs.task_name
        elif "product" in inputs:
            task_name = inputs["product"]
        else:
            task_name = None

        self.preds.append({
            "epk_id": inputs["epk_id"] if "epk_id" in inputs else None,
            "uplift": outputs.uplift.detach().contiguous().cpu().numpy()
            if outputs.uplift is not None
            else uplift,
            "group": outputs.group.detach().contiguous().cpu().numpy()
            if outputs.group is not None
            else None,
            "task_name": task_name,
            # "task_name": outputs.task_name.detach().contiguous().cpu().numpy()
            # if torch.is_tensor(outputs.task_name)
            # else outputs.task_name
            # if (hasattr(outputs, "task_name") and outputs.task_name is not None)
            # else inputs["product"]
            # if ("product" in inputs)
            # else None,
            "split_type": inputs["split_type"] if "split_type" in inputs else None,
            "treatment": outputs.treatment.detach().contiguous().cpu().numpy(),
            "y_true": outputs.conversion.detach().contiguous().cpu().numpy(),
            "c_probs": outputs.control_probs.detach().contiguous().cpu().numpy(),
            "t_probs": outputs.treatment_probs.detach().contiguous().cpu().numpy(),
        })

    def compute(self) -> dict[str, float]:
        """Computes uplift and classification metrics."""
        merged_preds = {
            key: np.concatenate([p[key] for p in self.preds])
            for key in self.preds[0]
            if self.preds[0][key] is not None
        }

        result = {}
        mean_main_metric = []
        merged_preds["calibrated_uplift"] = np.full_like(
            merged_preds["y_true"], np.nan, dtype=float
        )

        if "group" not in merged_preds:
            merged_preds["group"] = -1 * np.zeros(
                merged_preds["y_true"].shape[0]
            ).astype(np.int32)

        if "task_name" not in merged_preds:
            task_names = [""]
            merged_preds["task_name"] = np.array(
                task_names * merged_preds["y_true"].shape[0]
            )
        else:
            task_names = np.unique(merged_preds["task_name"])

        if "split_type" not in merged_preds:
            merged_preds["split_type"] = np.array(
                ["calib"] * merged_preds["y_true"].shape[0]
            )

        for task in task_names:
            task_mask = merged_preds["task_name"] == task
            task_groups = np.unique(merged_preds["group"][task_mask])
            for group in task_groups:
                t_group_mask = (
                    (merged_preds["group"] == group)
                    & (merged_preds["treatment"] == 1)
                    & (merged_preds["split_type"] == "calib")
                    & task_mask
                )
                c_group_mask = (
                    (merged_preds["group"] == group)
                    & (merged_preds["treatment"] == 0)
                    & (merged_preds["split_type"] == "calib")
                    & task_mask
                )

                t_calibrator = fit_calibrator(
                    scores=merged_preds["t_probs"][t_group_mask],
                    y_true=merged_preds["y_true"][t_group_mask],
                )

                c_calibrator = fit_calibrator(
                    scores=merged_preds["c_probs"][c_group_mask],
                    y_true=merged_preds["y_true"][c_group_mask],
                )

                calib_group_mask = (
                    (merged_preds["group"] == group)
                    & (merged_preds["split_type"] == "calib")
                    & task_mask
                )
                test_group_mask = (
                    (merged_preds["group"] == group)
                    & (merged_preds["split_type"] == "test")
                    & task_mask
                )

                calib_scores, calib_calibrated_uplift = (
                    apply_calibration_calculate_metrics(
                        y_true=merged_preds["y_true"][calib_group_mask],
                        t_probs=merged_preds["t_probs"][calib_group_mask],
                        c_probs=merged_preds["c_probs"][calib_group_mask],
                        treatment=merged_preds["treatment"][calib_group_mask],
                        t_calibrator=t_calibrator,
                        c_calibrator=c_calibrator,
                        is_calib=True,
                    )
                )

                merged_preds["calibrated_uplift"][calib_group_mask] = (
                    calib_calibrated_uplift
                )

                if merged_preds["y_true"][test_group_mask].shape[0] > 0:
                    test_scores, test_calibrated_uplift = (
                        apply_calibration_calculate_metrics(
                            y_true=merged_preds["y_true"][test_group_mask],
                            t_probs=merged_preds["t_probs"][test_group_mask],
                            c_probs=merged_preds["c_probs"][test_group_mask],
                            treatment=merged_preds["treatment"][test_group_mask],
                            t_calibrator=t_calibrator,
                            c_calibrator=c_calibrator,
                            is_calib=False,
                        )
                    )
                    merged_preds["calibrated_uplift"][test_group_mask] = (
                        test_calibrated_uplift
                    )
                    test_scores = {
                        f"test_group_{group}_{key}": val
                        for key, val in test_scores.items()
                    }
                else:
                    test_scores = {}

                calib_scores = {
                    f"calib_group_{group}_{key}": val
                    for key, val in calib_scores.items()
                }

                if merged_preds["y_true"][test_group_mask].shape[0] > 0:
                    if self.require_calibration:
                        mean_main_metric.append(
                            test_scores[
                                f"test_group_{group}_calibrated_{self.main_metric}"
                            ]
                        )
                    else:
                        mean_main_metric.append(
                            test_scores[f"test_group_{group}_{self.main_metric}"]
                        )
                else:
                    if self.require_calibration:
                        mean_main_metric.append(
                            calib_scores[
                                f"calib_group_{group}_calibrated_{self.main_metric}"
                            ]
                        )
                    else:
                        mean_main_metric.append(
                            calib_scores[f"calib_group_{group}_{self.main_metric}"]
                        )
                if len(task_names) > 0:

                    def task_name_prefix(scores, task_name):
                        return {
                            (
                                f"task_{task_name!s}_{key}"
                                if len(str(task_name)) != 0
                                else key
                            ): val
                            for key, val in scores.items()
                        }

                    result = {
                        **result,
                        **task_name_prefix(scores=calib_scores, task_name=task),
                        **task_name_prefix(scores=test_scores, task_name=task),
                    }
                else:
                    result = {**result, **calib_scores, **test_scores}
        if self.require_calibration:
            result[f"mean_calibrated_{self.main_metric}"] = np.mean(mean_main_metric)
            result[f"mean_{self.main_metric}"] = np.mean(mean_main_metric)
        else:
            result[f"mean_{self.main_metric}"] = np.mean(mean_main_metric)

        if self.save_submit is not None:
            os.makedirs(self.save_submit, exist_ok=True)
            submit_path = os.path.join(self.save_submit, "predict.parquet")
            pd.DataFrame(merged_preds).to_parquet(submit_path, index=False)
            logger.info("submit written to %s", submit_path)
        return result

    def reset(self):
        """Clears stored predictions."""
        self.preds = []
