import os
from datetime import datetime
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
)

from avatar.metrics.base import BaseMetric


def precision_at_k(y_true, y_pred, k_pnt):
    assert y_true.ndim == 1
    k = int(y_true.shape[0] * k_pnt / 100)
    top_k_indices = np.argsort(-y_pred)[:k]
    relevant = np.take(y_true, top_k_indices)
    return relevant.sum() / k


def recall_at_k(y_true, y_pred, k_pnt):
    assert y_true.ndim == 1
    k = int(y_true.shape[0] * k_pnt / 100)
    top_k_indices = np.argsort(-y_pred)[:k]
    relevant = np.take(y_true, top_k_indices)
    return relevant.sum() / y_true.sum()


def calculate_response_metrics(y_true, y_pred):
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
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
    return metrics


def apply_calculate_metrics(
    y_true, y_pred, task_type: Literal["binary_clf", "reg"] = "binary_clf"
):
    if task_type == "binary_clf":
        no_calib_metrics = calculate_response_metrics(y_true=y_true, y_pred=y_pred)
    elif task_type == "reg":
        no_calib_metrics = calculate_regression_metrics(y_true=y_true, y_pred=y_pred)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    metrics = {**no_calib_metrics}
    return metrics


def save_to_parquet(df: pd.DataFrame, path_to_save: str, prefix: str = None) -> str:
    """
    Save a DataFrame as a parquet file in the specified directory.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path_to_save : str
        Directory where the parquet file will be saved.
    prefix : str, optional
        Optional prefix for the filename.

    Returns
    -------
    str
        The full path to the saved parquet file.
    """
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save, exist_ok=True)
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if prefix is not None:
        filename = f"{time_now}_{prefix}.parquet"
    else:
        filename = f"{time_now}.parquet"
    full_path = os.path.join(path_to_save, filename)
    # for campatibility with spark cast datetime64[ns] to datetime64[us]
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    df.to_parquet(full_path, index=False)
    print(f"predict_saved: {full_path}")
    return full_path


class ResponseMetrics(BaseMetric):
    """Class for computing supervised metrics.
    Args:
        save_submit_path: str = None
            Path to save predictions.
    Available metrics:
        recall_at_5/10/15/20
        precision_at_5/10/15/20
        roc_auc_score
    """

    def __init__(
        self,
        save_submit_path: Optional[str] = None,
        main_metric: Optional[str] = "roc_auc_score",
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
            "y_pred": torch.nn.functional.sigmoid(outputs.logits)
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

    def compute(self) -> Dict[str, float]:
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

        if len(mean_main_metric) > 0:
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

    def reset(self) -> None:
        """Clears stored predictions."""
        self.preds = []


class RegressionMetrics(BaseMetric):
    """Class for computing regression metrics.
    Args:
        save_submit_path: str = None
            Path to save predictions.
    Available metrics:
        mse
        mae
        mape
    """

    def __init__(
        self,
        save_submit_path: Optional[str] = None,
        main_metric: Optional[str] = "mae",
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

    def compute(self) -> Dict[str, float]:
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

    def reset(self) -> None:
        """Clears stored predictions."""
        self.preds = []


class InferenceSupervisedMetrics(BaseMetric):
    def __init__(
        self,
        path_to_save,
        save_steps,
        task_type: Literal["binary_clf", "reg"],
        prefix=None,
    ):
        self.preds = []
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        self.prefix = prefix
        self.task_type = task_type

        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=False)

    def update(self, inputs, outputs):
        epk_id = inputs["epk_id"]
        if "task_type" in inputs:
            task_name = inputs["task_type"]
        else:
            task_name = len(inputs["epk_id"]) * ["unk"]

        if "target_attr_2" in inputs:
            target_attr_2 = inputs["target_attr_2"]
        else:
            target_attr_2 = len(inputs["epk_id"]) * [-1]

        if self.task_type == "binary_clf":
            prediction = (
                torch.nn.functional.sigmoid(outputs.logits)
                .squeeze(1)
                .detach()
                .contiguous()
                .cpu()
                .numpy()
            )
        elif self.task_type == "reg":
            prediction = outputs.logits.squeeze(1).detach().contiguous().cpu().numpy()
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        # Create the prediction dictionary
        pred_dict = {
            "epk_id": epk_id,
            "target_attr_2": target_attr_2,
            "prediction": prediction,
            "task_name": task_name,
        }

        # Add report_month if it exists in inputs
        if "report_month" in inputs:
            pred_dict["report_month"] = inputs["report_month"]

        self.preds.append(pred_dict)

        if len(self.preds) >= self.save_steps:
            self.compute()

    def compute(self):
        """return Dict(metric_name: value)"""
        predict = {
            "epk_id": [],
            "target_attr_2": [],
            "prediction": [],
            "task_name": [],
        }

        has_report_month = any("report_month" in item for item in self.preds)
        if has_report_month:
            predict["report_month"] = []

        for item in self.preds:
            predict["epk_id"].extend(item["epk_id"])
            predict["target_attr_2"].extend(item["target_attr_2"])
            predict["prediction"].extend(item["prediction"])
            predict["task_name"].extend(item["task_name"])
            if has_report_month:
                predict["report_month"].extend(
                    item.get("report_month", [None] * len(item["epk_id"]))
                )

        predict_df = pd.DataFrame().from_dict(predict)
        if has_report_month:
            predict_df["report_month"] = predict_df["report_month"].apply(
                lambda x: str(x.date())
            )
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.prefix is not None:
            cur_path = (
                os.path.join(self.path_to_save, time_now) + f"_{self.prefix}.parquet"
            )
        else:
            cur_path = os.path.join(self.path_to_save, time_now) + ".parquet"
        predict_df.to_parquet(cur_path, index=False)
        self.reset()
        print(f"predict_saved: {cur_path}")

    def reset(self):
        self.preds = []


class MMoEResponseMetrics(ResponseMetrics):
    """
    Response metrics + MMoE gate utilization metrics.

    Adds metrics like:
    gate_product_{product}_expert_{i}_mean
    gate_product_{product}_entropy_mean
    gate_product_{product}_max_mean
    gate_product_{product}_top1_expert_{i}_rate
    """

    def __init__(
        self,
        save_submit_path: Optional[str] = None,
        main_metric: Optional[str] = "roc_auc_score",
        num_experts: int = None,
    ):
        super().__init__(
            save_submit_path=save_submit_path,
            main_metric=main_metric,
        )
        self.num_experts = num_experts
        self.gate_preds = []

    def update(self, inputs, outputs):
        super().update(inputs, outputs)

        if not hasattr(outputs, "aux") or outputs.aux is None:
            return

        if "gate" not in outputs.aux:
            return

        if "product" in inputs:
            product = inputs["product"]
            if torch.is_tensor(product):
                product = product.detach().contiguous().cpu().numpy()
        elif hasattr(outputs, "task_name") and outputs.task_name is not None:
            product = outputs.task_name
            if torch.is_tensor(product):
                product = product.detach().contiguous().cpu().numpy()
        else:
            product = None

        gate = outputs.aux["gate"].detach().contiguous().cpu().numpy()
        gate_entropy = outputs.aux["gate_entropy"].detach().contiguous().cpu().numpy()
        gate_max = outputs.aux["gate_max"].detach().contiguous().cpu().numpy()
        gate_top1 = outputs.aux["gate_top1"].detach().contiguous().cpu().numpy()

        task_name = outputs.task_name
        if torch.is_tensor(task_name):
            task_name = task_name.detach().contiguous().cpu().numpy()

        group = outputs.group
        if torch.is_tensor(group):
            group = group.detach().contiguous().cpu().numpy()

        split_type = inputs["split_type"] if "split_type" in inputs else None

        self.gate_preds.append({
            "product": product,
            "task_name": task_name,
            "group": group,
            "split_type": split_type,
            "gate": gate,
            "gate_entropy": gate_entropy,
            "gate_max": gate_max,
            "gate_top1": gate_top1,
        })

    def _compute_gate_metrics(self) -> dict[str, float]:
        if len(self.gate_preds) == 0:
            return {}

        gate = np.concatenate([p["gate"] for p in self.gate_preds], axis=0)
        gate_entropy = np.concatenate(
            [p["gate_entropy"] for p in self.gate_preds],
            axis=0,
        )
        gate_max = np.concatenate([p["gate_max"] for p in self.gate_preds], axis=0)
        gate_top1 = np.concatenate([p["gate_top1"] for p in self.gate_preds], axis=0)

        product_values = [
            p["product"] for p in self.gate_preds if p["product"] is not None
        ]

        if len(product_values) > 0:
            product = np.concatenate(product_values, axis=0)
        else:
            product = np.array(["all"] * gate.shape[0])

        split_values = [
            p["split_type"] for p in self.gate_preds if p["split_type"] is not None
        ]

        if len(split_values) > 0:
            split_type = np.concatenate(split_values, axis=0)
        else:
            split_type = np.array(["calib"] * gate.shape[0])

        num_experts = self.num_experts or gate.shape[1]
        result = {}

        def add_gate_states(prefix: str, mask: np.ndarray):
            if mask.sum() == 0:
                return

            for expert_idx in range(num_experts):
                result[f"{prefix}_expert_{expert_idx}_mean"] = float(
                    gate[mask, expert_idx].mean()
                )

            result[f"{prefix}_entropy_mean"] = np.nan_to_num(
                float(gate_entropy[mask].mean()),
                nan=0.0,
            )
            result[f"{prefix}_max_mean"] = float(gate_max[mask].mean())

            for expert_idx in range(num_experts):
                result[f"{prefix}_top1_expert_{expert_idx}_rate"] = float(
                    (gate_top1[mask] == expert_idx).mean()
                )

        # Global gate stats
        add_gate_states("gate_global", np.ones(gate.shape[0], dtype=bool))

        # Product-level gate stats
        for split in np.unique(split_type):
            split_mask = split_type == split
            add_gate_states(f"{split}_gate_global", split_mask)
            for product_name in np.unique(product):
                mask = split_mask & (product == product_name)
                add_gate_states(f"{split}_gate_task_{product_name}", mask)
        return result

    def compute(self) -> Dict[str, float]:
        result = super().compute()
        result.update(self._compute_gate_metrics())
        return result

    def reset(self) -> None:
        super().reset()
        self.gate_preds = []


class PLEResponseMetrics(MMoEResponseMetrics):
    """
    Response metrics + PLE (CGC) gate utilization metrics.

    In PLE, a task's gate evaluates [Shared Experts] + [Task-Specific Experts].
    This class correctly maps gate indices to global/shared vs specific metrics
    so that aggregating across tasks produces mathematically valid statistics.

    Adds metrics like:
    gate_{task/global}_{shared/specific}_expert_{i}_mean
    gate_{task/global}_{product}_entropy_mean
    gate_{task/global}_{product}_max_mean
    gate_{task/global}_{product}_top1_{shared/specific}_expert_{i}_rate
    """

    def __init__(
        self,
        num_shared_experts: int,
        num_specific_experts: int,
        save_submit_path: Optional[str] = None,
        main_metric: Optional[str] = "roc_auc_score",
    ):
        # The gate for each task will output (num_shared + num_specific) values
        num_experts_per_task = num_shared_experts + num_specific_experts
        super().__init__(
            save_submit_path=save_submit_path,
            main_metric=main_metric,
            num_experts=num_experts_per_task,
        )
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts

    def _compute_gate_metrics(self) -> dict[str, float]:
        if len(self.gate_preds) == 0:
            return {}

        gate = np.concatenate([p["gate"] for p in self.gate_preds], axis=0)
        gate_entropy = np.concatenate(
            [p["gate_entropy"] for p in self.gate_preds], axis=0
        )
        gate_max = np.concatenate([p["gate_max"] for p in self.gate_preds], axis=0)
        gate_top1 = np.concatenate([p["gate_top1"] for p in self.gate_preds], axis=0)

        product_values = [
            p["product"] for p in self.gate_preds if p["product"] is not None
        ]
        product = (
            np.concatenate(product_values, axis=0)
            if len(product_values) > 0
            else np.array(["all"] * gate.shape[0])
        )

        split_values = [
            p["split_type"] for p in self.gate_preds if p["split_type"] is not None
        ]
        split_type = (
            np.concatenate(split_values, axis=0)
            if len(split_values) > 0
            else np.array(["calib"] * gate.shape[0])
        )

        result = {}

        def add_ple_gate_states(prefix: str, mask: np.ndarray, is_global: bool = False):
            if mask.sum() == 0:
                return

            # 1. Entropy & Max stats (always valid)
            result[f"{prefix}_entropy_mean"] = np.nan_to_num(
                float(gate_entropy[mask].mean()),
                nan=0.0,
            )
            result[f"{prefix}_max_mean"] = float(gate_max[mask].mean())

            # 2. Shared Experts (Indices 0 to num_shared_experts - 1)

            if self.num_shared_experts > 0:
                for idx in range(self.num_shared_experts):
                    result[f"{prefix}_shared_expert_{idx}_mean"] = float(
                        gate[mask, idx].mean()
                    )
                    result[f"{prefix}_top1_shared_expert_{idx}_rate"] = float(
                        (gate_top1[mask] == idx).mean()
                    )

            # 3. Task-Specific Experts (Indices num_shared_experts to end)
            for specific_idx in range(self.num_specific_experts):
                actual_idx = self.num_shared_experts + specific_idx

                # If we are looking at a specific task, we label it as specific_expert_{i}
                if not is_global:
                    result[f"{prefix}_specific_expert_{specific_idx}_mean"] = float(
                        gate[mask, actual_idx].mean()
                    )
                    result[f"{prefix}_top1_specific_expert_{specific_idx}_rate"] = (
                        float((gate_top1[mask] == actual_idx).mean())
                    )
                else:
                    # If global, "actual_idx" points to completely different experts for different tasks.
                    # We aggregate it as a general "utilization of task-specific experts" metric.
                    result[f"{prefix}_any_specific_expert_mean"] = float(
                        gate[mask, actual_idx:].sum(axis=1).mean()
                    )
                    result[f"{prefix}_top1_any_specific_expert_rate"] = float(
                        (gate_top1[mask] >= self.num_shared_experts).mean()
                    )

        # Global gate stats
        add_ple_gate_states(
            "gate_global", np.ones(gate.shape[0], dtype=bool), is_global=True
        )

        # Split & Product level gate stats
        for split in np.unique(split_type):
            split_mask = split_type == split
            add_ple_gate_states(f"{split}_gate_global", split_mask, is_global=True)

            for product_name in np.unique(product):
                mask = split_mask & (product == product_name)
                add_ple_gate_states(
                    f"{split}_gate_task_{product_name}", mask, is_global=False
                )

        return result
