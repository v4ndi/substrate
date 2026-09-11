"""Response and regression metrics, plus the inference-time collector.

"Response" here means the ordinary supervised setting — one probability per
record, scored with ROC AUC and precision/recall at the top k percent, which is
how campaign quality is judged.
"""

import abc
import logging
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

from avatar.metrics.base import ArtifactMetric
from avatar.metrics.grouped import (
    CALIB_SPLIT,
    TEST_SPLIT,
    GroupedPredictionMetric,
    group_prefixed,
)

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


class SupervisedMetric(GroupedPredictionMetric):
    """One head, one target column, scored per group and per split.

    The two concrete metrics below differ in exactly two things: which metric
    set :func:`apply_calculate_metrics` should use, and how a logit becomes a
    prediction. Everything else — buffering, merging, the group loop and the
    submit file — comes from :class:`~avatar.metrics.grouped.GroupedPredictionMetric`.
    """

    #: Which metric family :func:`apply_calculate_metrics` computes.
    task_type: Literal["binary_clf", "reg"]

    required_inputs = ("epk_id", "group", "is_treat", "targets", "split_type")
    required_outputs = ("logits",)

    @abc.abstractmethod
    def predictions(self, outputs) -> np.ndarray:
        """Turn the model's logits into the number that gets scored."""

    def collect(self, inputs, outputs) -> dict:
        """Keep the target, the prediction and the columns that slice them."""
        return {
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
            "y_pred": self.predictions(outputs),
            "split_type": inputs["split_type"] if "split_type" in inputs else None,
        }

    def score_group(self, merged, group, calib_mask, test_mask):
        """Score both slices of one group; the held-out one feeds the mean."""
        scores: dict[str, float] = {}
        main_value = None
        for split, mask in ((CALIB_SPLIT, calib_mask), (TEST_SPLIT, test_mask)):
            if not mask.any():
                continue
            slice_scores = apply_calculate_metrics(
                y_true=merged["y_true"][mask],
                y_pred=merged["y_pred"][mask],
                task_type=self.task_type,
            )
            scores.update(group_prefixed(slice_scores, split, group))
            # ``test`` comes second, so when the group has a held-out slice its
            # number is the one that survives into the mean.
            main_value = slice_scores[self.main_metric]
        return scores, main_value


class ResponseMetrics(SupervisedMetric):
    """Ranking quality of a single-head response model.

    Args:
        save_submit_path: Directory to write per-record predictions into.
            ``None`` writes nothing.
        main_metric: Which metric counts as *the* number for this run.

    Returns from :meth:`compute`:
        ``{calib,test}_group_{group}_{metric}`` for ``roc_auc_score``,
        ``recall_at_{5,10,15,20}`` and ``precision_at_{5,10,15,20}``, plus
        ``mean_{main_metric}`` over the groups. The ``k`` is a **percentage**
        of the population, not a record count, so ``precision_at_5`` is
        precision in the top 5% by score.
    """

    task_type = "binary_clf"

    def __init__(
        self,
        save_submit_path: str | None = None,
        main_metric: str | None = "roc_auc_score",
    ):
        super().__init__(save_submit_path=save_submit_path, main_metric=main_metric)

    def predictions(self, outputs) -> np.ndarray:
        """A probability per record."""
        return (
            torch.nn.functional
            .sigmoid(outputs.logits)
            .squeeze(1)
            .detach()
            .contiguous()
            .cpu()
            .numpy()
        )


class RegressionMetrics(SupervisedMetric):
    """Error metrics for a regression head.

    Args:
        save_submit_path: Directory to write per-record predictions into.
            ``None`` writes nothing.
        main_metric: Which metric counts as *the* number for this run.

    Returns from :meth:`compute`:
        ``{calib,test}_group_{group}_{mse,mae,mape}`` plus
        ``mean_{main_metric}`` over the groups.
    """

    task_type = "reg"

    def __init__(
        self,
        save_submit_path: str | None = None,
        main_metric: str | None = "mae",
    ):
        super().__init__(save_submit_path=save_submit_path, main_metric=main_metric)

    def predictions(self, outputs) -> np.ndarray:
        """The raw head output, scored against the target as-is."""
        return outputs.logits.squeeze(1).detach().contiguous().cpu().numpy()


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

    required_inputs = ("epk_id", "target_attr_2", "report_month")
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
        target_attr_2 = inputs.get("target_attr_2", len(epk_id) * [-1])

        logits = outputs.logits.squeeze(1)
        if self.task_type == "binary_clf":
            logits = torch.nn.functional.sigmoid(logits)
        prediction = logits.detach().contiguous().cpu().numpy()

        pred_dict = {
            "epk_id": epk_id,
            "target_attr_2": target_attr_2,
            "prediction": prediction,
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

        columns = ["epk_id", "target_attr_2", "prediction"]
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
