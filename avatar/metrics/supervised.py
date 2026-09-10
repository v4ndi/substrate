"""Response and regression metrics, plus the inference-time collector.

"Response" here means the ordinary supervised setting — one probability per
record, scored with ROC AUC and precision/recall at the top k percent, which is
how campaign quality is judged.
"""

import logging
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
)

from avatar.metrics.base import ArtifactMetric, ScalarMetric

logger = logging.getLogger(__name__)


def precision_at_k(y_true, y_pred, k_pnt):
    """Share of positives among the top ``k_pnt`` **percent** by score."""
    assert y_true.ndim == 1
    k = int(y_true.shape[0] * k_pnt / 100)
    top_k_indices = np.argsort(-y_pred)[:k]
    relevant = np.take(y_true, top_k_indices)
    return relevant.sum() / k


def recall_at_k(y_true, y_pred, k_pnt):
    """Share of all positives captured by the top ``k_pnt`` percent by score."""
    assert y_true.ndim == 1
    k = int(y_true.shape[0] * k_pnt / 100)
    top_k_indices = np.argsort(-y_pred)[:k]
    relevant = np.take(y_true, top_k_indices)
    return relevant.sum() / y_true.sum()


def calculate_response_metrics(y_true, y_pred):
    """ROC AUC plus precision and recall at 5, 10, 15 and 20 percent."""
    metrics = (
        {"roc_auc_score": roc_auc_score(y_true, y_pred)}
        | {f"recall_at_{k}": recall_at_k(y_true, y_pred, k) for k in [5, 10, 15, 20]}
        | {
            f"precision_at_{k}": precision_at_k(y_true, y_pred, k)
            for k in [5, 10, 15, 20]
        }
    )
    return metrics


def calculate_regression_metrics(y_true, y_pred):
    """MSE, MAE and MAPE."""
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
    return metrics


def apply_calculate_metrics(
    y_true, y_pred, task_type: Literal["binary_clf", "reg"] = "binary_clf"
):
    """Dispatch to the response or regression metric set.

    Raises:
        ValueError: ``task_type`` is neither ``binary_clf`` nor ``reg``.
    """
    if task_type == "binary_clf":
        no_calib_metrics = calculate_response_metrics(y_true=y_true, y_pred=y_pred)
    elif task_type == "reg":
        no_calib_metrics = calculate_regression_metrics(y_true=y_true, y_pred=y_pred)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    metrics = {**no_calib_metrics}
    return metrics


class ResponseMetrics(ScalarMetric):
    """Ranking quality of a single-head response model.

    Args:
        save_submit_path: Directory to write per-record predictions into.
            ``None`` writes nothing.
        main_metric: Which metric counts as *the* number for this run.

    Returns from :meth:`compute`:
        ``roc_auc_score``, ``recall_at_{5,10,15,20}`` and
        ``precision_at_{5,10,15,20}``. The ``k`` is a **percentage** of the
        population, not a record count, so ``precision_at_5`` is precision in
        the top 5% by score.
    """

    required_inputs = (
        "epk_id",
        "group",
        "is_treat",
        "targets",
        "product",
        "split_type",
    )
    required_outputs = ("logits", "task_name")

    def __init__(
        self,
        save_submit_path: str | None = None,
        main_metric: str | None = "roc_auc_score",
    ):
        self.preds = []
        self.save_submit = save_submit_path
        self.main_metric = main_metric

    def update(self, inputs, outputs):
        """Stores model predictions for later computation.

        Args:
            inputs: Dict[str, any]
            outputs: Dict[str, torch.Tensor]
        """
        self.preds.append({
            "epk_id": inputs["epk_id"] if "epk_id" in inputs else None,
            "group": inputs["group"].detach().contiguous().cpu().numpy()
            if "group" in inputs
            else None,
            "treatment": inputs["is_treat"].detach().contiguous().cpu().numpy()
            if "is_treat" in inputs
            else None,
            "y_true": inputs["targets"].detach().contiguous().cpu().numpy()
            if "targets" in inputs
            else None,
            "y_pred": torch.nn.functional
            .sigmoid(outputs.logits)
            .squeeze(1)
            .detach()
            .contiguous()
            .cpu()
            .numpy(),
            # backward compatibility with uplift metrics
            "task_name": outputs.task_name.detach().contiguous().cpu().numpy()
            if torch.is_tensor(outputs.task_name)
            else outputs.task_name
            if (hasattr(outputs, "task_name") and outputs.task_name is not None)
            else inputs["product"]
            if ("product" in inputs)
            else None,
            "split_type": inputs["split_type"] if "split_type" in inputs else None,
        })

    def compute(self) -> dict[str, float]:
        """Computes metrics."""
        merged_preds = {
            key: np.concatenate([p[key] for p in self.preds])
            for key in self.preds[0]
            if self.preds[0][key] is not None
        }
        result = {}
        mean_main_metric = []

        if "group" not in merged_preds:
            merged_preds["group"] = -1 * np.ones(
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
                calib_scores = {}
                test_scores = {}

                if merged_preds["y_true"][calib_group_mask].shape[0] > 0:
                    calib_scores = apply_calculate_metrics(
                        y_true=merged_preds["y_true"][calib_group_mask],
                        y_pred=merged_preds["y_pred"][calib_group_mask],
                        task_type="binary_clf",
                    )

                    calib_scores = {
                        f"calib_group_{group}_{key}": val
                        for key, val in calib_scores.items()
                    }

                if merged_preds["y_true"][test_group_mask].shape[0] > 0:
                    test_scores = apply_calculate_metrics(
                        y_true=merged_preds["y_true"][test_group_mask],
                        y_pred=merged_preds["y_pred"][test_group_mask],
                        task_type="binary_clf",
                    )

                    test_scores = {
                        f"test_group_{group}_{key}": val
                        for key, val in test_scores.items()
                    }

                if merged_preds["y_true"][test_group_mask].shape[0] > 0:
                    mean_main_metric.append(
                        test_scores[f"test_group_{group}_{self.main_metric}"]
                    )
                elif merged_preds["y_true"][calib_group_mask].shape[0] > 0:
                    mean_main_metric.append(
                        calib_scores[f"calib_group_{group}_{self.main_metric}"]
                    )
                else:
                    continue

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

        if len(mean_main_metric) > 0:
            result[f"mean_{self.main_metric}"] = np.mean(mean_main_metric)

        if self.save_submit is not None:
            os.makedirs(self.save_submit, exist_ok=True)
            submit_path = os.path.join(self.save_submit, "predict.parquet")
            pd.DataFrame(merged_preds).to_parquet(submit_path, index=False)
            logger.info("submit written to %s", submit_path)
        return result

    def reset(self) -> None:
        """Clears stored predictions."""
        self.preds = []


class RegressionMetrics(ScalarMetric):
    """Error metrics for a regression head.

    Args:
        save_submit_path: str = None
            Path to save predictions.
    Available metrics:
        mse
        mae
        mape
    """

    required_inputs = (
        "epk_id",
        "group",
        "is_treat",
        "targets",
        "product",
        "split_type",
    )
    required_outputs = ("logits", "task_name")

    def __init__(
        self,
        save_submit_path: str | None = None,
        main_metric: str | None = "mae",
    ):
        self.preds = []
        self.save_submit = save_submit_path
        self.main_metric = main_metric

    def update(self, inputs, outputs):
        """Stores model predictions for later computation.

        Args:
            inputs: Dict[str, any]
            outputs: Dict[str, torch.Tensor]
        """
        self.preds.append({
            "epk_id": inputs["epk_id"] if "epk_id" in inputs else None,
            "group": inputs["group"].detach().contiguous().cpu().numpy()
            if "group" in inputs
            else None,
            "treatment": inputs["is_treat"].detach().contiguous().cpu().numpy()
            if "is_treat" in inputs
            else None,
            "y_true": inputs["targets"].detach().contiguous().cpu().numpy()
            if "targets" in inputs
            else None,
            "y_pred": outputs.logits.squeeze(1).detach().contiguous().cpu().numpy(),
            # backward compatibility with uplift metrics
            "task_name": outputs.task_name
            if (hasattr(outputs, "task_name") and outputs.task_name is not None)
            else len(inputs["epk_id"]) * ["unk"]
            if ("product" in inputs)
            else None,
            "split_type": inputs["split_type"] if "split_type" in inputs else None,
        })

    def compute(self) -> dict[str, float]:
        """Computes metrics."""
        merged_preds = {
            key: np.concatenate([p[key] for p in self.preds])
            for key in self.preds[0]
            if self.preds[0][key] is not None
        }
        result = {}
        mean_main_metric = []

        if "group" not in merged_preds:
            merged_preds["group"] = -1 * np.ones(
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

                calib_scores = apply_calculate_metrics(
                    y_true=merged_preds["y_true"][calib_group_mask],
                    y_pred=merged_preds["y_pred"][calib_group_mask],
                    task_type="reg",
                )

                calib_scores = {
                    f"calib_group_{group}_{key}": val
                    for key, val in calib_scores.items()
                }

                if merged_preds["y_true"][test_group_mask].shape[0] > 0:
                    test_scores = apply_calculate_metrics(
                        y_true=merged_preds["y_true"][test_group_mask],
                        y_pred=merged_preds["y_pred"][test_group_mask],
                        task_type="reg",
                    )

                    test_scores = {
                        f"test_group_{group}_{key}": val
                        for key, val in test_scores.items()
                    }
                else:
                    test_scores = {}

                if merged_preds["y_true"][test_group_mask].shape[0] > 0:
                    mean_main_metric.append(
                        test_scores[f"test_group_{group}_{self.main_metric}"]
                    )
                else:
                    mean_main_metric.append(
                        calib_scores[f"calib_group_{group}_{self.main_metric}"]
                    )

                if len(task_names) > 0:

                    def task_name_prefix(scores, task_name):
                        return {
                            f"{task_name}_" if len(task_name) != 0 else "" + key: val
                            for key, val in scores.items()
                        }

                    result = {
                        **result,
                        **task_name_prefix(scores=calib_scores, task_name=task),
                        **task_name_prefix(scores=test_scores, task_name=task),
                    }
                else:
                    result = {**result, **calib_scores, **test_scores}

        result[f"mean_{self.main_metric}"] = np.mean(mean_main_metric)

        if self.save_submit is not None:
            os.makedirs(self.save_submit, exist_ok=True)
            submit_path = os.path.join(self.save_submit, "predict.parquet")
            pd.DataFrame(merged_preds).to_parquet(submit_path, index=False)
            logger.info("submit written to %s", submit_path)

        return result

    def reset(self) -> None:
        """Clears stored predictions."""
        self.preds = []


class InferenceSupervisedMetrics(ArtifactMetric):
    """Write per-record predictions to parquet during inference.

    The product here is a file, not a number: predictions are flushed every
    ``save_steps`` batches so a long inference run does not hold the whole
    population in memory, and :meth:`compute` writes the tail and returns an
    empty dict.

    Args:
        path_to_save: Directory for the parquet parts; created if missing.
        save_steps: Flush every this many batches.
        task_type: ``binary_clf`` or ``reg`` — selects how logits become the
            saved prediction.
        prefix: Optional tag in the filename, to tell runs apart.

    Raises:
        ValueError: ``task_type`` is neither ``binary_clf`` nor ``reg``.
    """

    required_inputs = ("epk_id", "task_type", "target_attr_2", "report_month")
    required_outputs = ("logits",)

    def __init__(
        self,
        path_to_save: str,
        save_steps: int,
        task_type: Literal["binary_clf", "reg"],
        prefix: str | None = None,
    ):
        super().__init__(path_to_save=path_to_save, prefix=prefix)
        self.preds: list[dict] = []
        self.save_steps = save_steps
        if task_type not in ("binary_clf", "reg"):
            raise ValueError(f"Unknown task_type: {task_type}")
        self.task_type = task_type

    def update(self, inputs, outputs):
        """Accumulate one batch of predictions, flushing when the buffer is full."""
        epk_id = inputs["epk_id"]
        task_name = inputs.get("task_type", len(epk_id) * ["unk"])
        target_attr_2 = inputs.get("target_attr_2", len(epk_id) * [-1])

        logits = outputs.logits.squeeze(1)
        if self.task_type == "binary_clf":
            logits = torch.nn.functional.sigmoid(logits)
        prediction = logits.detach().contiguous().cpu().numpy()

        pred_dict = {
            "epk_id": epk_id,
            "target_attr_2": target_attr_2,
            "prediction": prediction,
            "task_name": task_name,
        }
        if "report_month" in inputs:
            pred_dict["report_month"] = inputs["report_month"]

        self.preds.append(pred_dict)

        if len(self.preds) >= self.save_steps:
            self.flush()

    def flush(self) -> None:
        """Write the buffered predictions to a parquet part and forget them."""
        if not self.preds:
            return

        columns = ["epk_id", "target_attr_2", "prediction", "task_name"]
        predict: dict[str, list] = {column: [] for column in columns}

        has_report_month = any("report_month" in item for item in self.preds)
        if has_report_month:
            predict["report_month"] = []

        for item in self.preds:
            for column in columns:
                predict[column].extend(item[column])
            if has_report_month:
                predict["report_month"].extend(
                    item.get("report_month", [None] * len(item["epk_id"]))
                )

        frame = pd.DataFrame.from_dict(predict)
        if has_report_month:
            frame["report_month"] = frame["report_month"].apply(lambda x: str(x.date()))
        self.save_dataframe(frame)
        self.reset()

    def reset(self) -> None:
        """Drop the buffered batches."""
        self.preds = []
