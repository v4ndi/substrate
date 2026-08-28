from typing import Dict

import numpy as np
import pandas as pd
import torch
from betacal import BetaCalibration
from sklearn.metrics import roc_auc_score
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

from avatar.metrics.base import BaseMetric


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
    calibrator = BetaCalibration()
    if y_true.sum() == 0:
        y_true[0] += 1
    calibrator.fit(scores.reshape(-1, 1), y_true)
    return calibrator


def apply_calibration_calculate_metrics(
    y_true, t_probs, c_probs, treatment, t_calibrator, c_calibrator, is_calib
):
    no_calib_metrics = calculate_uplift_metrics(
        y_true=y_true, uplift=(t_probs - c_probs), treatment=treatment, calibrated=False
    )

    c_roc_auc = roc_auc_score(y_true[treatment == 0], c_probs[treatment == 0])
    t_roc_auc = roc_auc_score(y_true[treatment == 1], t_probs[treatment == 1])

    _t_probs = t_calibrator.predict(t_probs.reshape(-1, 1))
    _c_probs = c_calibrator.predict(c_probs.reshape(-1, 1))

    calib_c_roc_auc = roc_auc_score(y_true[treatment == 0], _c_probs[treatment == 0])
    calib_t_roc_auc = roc_auc_score(y_true[treatment == 1], _t_probs[treatment == 1])

    assert abs(calib_c_roc_auc - c_roc_auc) < 0.01
    assert abs(calib_t_roc_auc - t_roc_auc) < 0.01
    if is_calib:
        assert (
            abs(_c_probs[treatment == 0].mean() - y_true[treatment == 0].mean()) < 1e-03
        )
        assert (
            abs(_t_probs[treatment == 1].mean() - y_true[treatment == 1].mean()) < 1e-03
        )

    calibrated_uplift = _t_probs - _c_probs

    calib_metrics = calculate_uplift_metrics(
        y_true=y_true, uplift=calibrated_uplift, treatment=treatment, calibrated=True
    )
    metrics = {**no_calib_metrics, **calib_metrics}
    metrics["treatment_roc_auc_score"] = calib_t_roc_auc
    metrics["control_roc_auc_score"] = calib_c_roc_auc
    return metrics, calibrated_uplift


class UpliftMetrics(BaseMetric):
    """Class for computing campaign uplift metrics.
    Args:
        require_calibration: bool = False
            Whether to apply BetaCalibration for uplift and classification metrics.
        save_submit_path: str = None
            Path to save predictions.
        main_metric: str = "qini_auc_score"
            Available metrics:
                uplift_at_5
                uplift_at_10
                uplift_at_15
                uplift_at_20
                uplift_auc_score
                qini_auc_score
    }

    """

    def __init__(
        self,
        require_calibration: bool = False,
        save_submit_path: str = None,
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

    def compute(self) -> Dict[str, float]:
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
                                f"task_{str(task_name)}_{key}"
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
            import os

            df = pd.DataFrame(merged_preds)
            if not os.path.exists(self.save_submit):
                os.makedirs(self.save_submit, exist_ok=True)
            df.to_parquet(
                os.path.join(self.save_submit, "predict.parquet"), index=False
            )
            print(
                f"Predict was saved: {os.path.join(self.save_submit, 'predict.parquet')}"
            )
        return result

    def reset(self):
        """Clears stored predictions."""
        self.preds = []
