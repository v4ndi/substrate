import datetime
import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .base import BaseMetric

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


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
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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


class CollectEmbeddings(BaseMetric):
    """
    Metric class for collecting and saving sequence embeddings and optional columns.

    Parameters
    ----------
    path_to_save : str
        Directory where the parquet files will be saved.
    save_steps : int
        Number of updates after which data is saved.
    prefix : str, optional
        Optional prefix for saved file names.
    additional_columns : list of str, optional
        List of additional input columns to collect and save if present.
    """

    def __init__(
        self,
        path_to_save: str,
        save_steps: int,
        prefix: Optional[str] = None,
        additional_columns: list[str] = None,
        month_part_value: str = None,
    ):
        """
        Initialize the CollectEmbeddings metric.

        See class docstring for parameter descriptions.
        """
        self.preds = []
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        self.prefix = prefix
        self.additional_columns = additional_columns or []
        self.month_part_value = month_part_value

        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=True)

    def update(self, inputs, outputs):
        """
        Update the metric state with new batch data.

        Parameters
        ----------
        inputs : dict
            Dictionary containing input data. Must include 'epk_id'.
            May include additional columns specified in `additional_columns`.
        outputs : torch.Tensor
            Sequence hidden states for the batch.
        """
        if hasattr(outputs, "aggregated_hidden_state"):
            seq_hidden_state = outputs.aggregated_hidden_state
        else:
            seq_hidden_state = outputs.last_hidden_state
        seq_hidden_state = seq_hidden_state.detach().contiguous().cpu().numpy()

        pred_dict = {
            "epk_id": inputs["epk_id"],
            "seq_hidden_state": seq_hidden_state,
        }
        for col in self.additional_columns:
            pred_dict[col] = [val for val in inputs[col]]

        self.preds.append(pred_dict)

        if len(self.preds) >= self.save_steps:
            self.compute()

    def compute(self):
        """
        Aggregate collected data and save to a parquet file.

        Returns
        -------
        dict
            Empty dictionary (for compatibility with metric interface).
        """
        predict = {
            "epk_id": [],
            "seq_hidden_state": [],
        }
        for col in self.additional_columns:
            if any(col in item for item in self.preds):
                predict[col] = []

        for item in self.preds:
            predict["epk_id"].extend(item["epk_id"])
            predict["seq_hidden_state"].extend(item["seq_hidden_state"])
            for col in self.additional_columns:
                if col in item:
                    predict[col].extend(item[col])

        predict_df = pd.DataFrame().from_dict(predict)
        if self.month_part_value is not None:
            predict_df["month_part"] = self.month_part_value
        save_to_parquet(predict_df, self.path_to_save, self.prefix)
        self.reset()
        return {}

    def reset(self):
        """
        Reset the internal state of the metric.
        """
        self.preds = []


class CatboostCampaignBenchmark(CollectEmbeddings):
    """
    A class for benchmarking foundation models on campaign data.
    """

    def __init__(
        self,
        path_to_save: str,
        save_steps: int,
        compute_contour: list[str],
        catboost_params: dict[str, object],
        split: dict[str, str],
        repartition_by_product: bool = False,
        channel_type: str = None,
        control_flag: str = None,
        additional_columns: list[str] = None,
        prefix: Optional[str] = None,
    ):
        """
        Initialize the CatboostCampaignBenchmark instance.

        Args:
            path_to_save (str): Path where results will be saved.
            save_steps (int): Number of steps after which to save predictions.
            compute_contour (list[str]): List of contours (products) to process.
            catboost_params (dict[str, object]): Parameters for the CatBoost model.
            split (dict[str, str]): Dict with necessary keys: "train", "valid" and "test".
            channel_type (str, optional): Column name for channel type. Defaults to None.
            control_flag (str, optional): Column name for control flag. Defaults to None.
            additional_columns (list[str], optional) : List of additional input columns to collect and save if present.
            prefix (str, optional) : Optional prefix for saved file names.

        Raises:
            ImportError: If Catboost is not installed.
        """
        if not HAS_CATBOOST:
            raise ImportError(
                "You need to install Catboost to use Campaign Catboost benchmark."
            )
        super().__init__(
            path_to_save=path_to_save,
            save_steps=save_steps,
            prefix=prefix,
            additional_columns=additional_columns,
        )
        self.repartition_by_product = repartition_by_product
        self.compute_contour = compute_contour
        self.catboost_params = catboost_params
        self.channel_type = channel_type
        self.control_flag = control_flag
        self.split = split
        self.contour_scores = {}

    def reset(self):
        self.preds = []

    def save_parquet(self):
        predict = {
            "epk_id": [],
            "seq_hidden_state": [],
            "target_attr_1": [],
            "report_month": [],
        }
        for col in self.additional_columns:
            if any(col in item for item in self.preds):
                predict[col] = []

        for item in self.preds:
            predict["epk_id"].extend(item["epk_id"])
            predict["seq_hidden_state"].extend(item["seq_hidden_state"])
            predict["target_attr_1"].extend(item["target_attr_1"])
            predict["report_month"].extend(item["report_month"])
            for col in self.additional_columns:
                if col in item:
                    predict[col].extend(item[col])
        predict_df = pd.DataFrame().from_dict(predict)
        save_to_parquet(
            df=predict_df, path_to_save=self.path_to_save, prefix=self.prefix
        )
        self.reset()

    def update(self, inputs, outputs):
        if hasattr(outputs, "aggregated_hidden_state"):
            seq_hidden_state = outputs.aggregated_hidden_state
        else:
            seq_hidden_state = outputs.last_hidden_state
        seq_hidden_state = seq_hidden_state.detach().contiguous().cpu().numpy()
        pred_dict = {
            "epk_id": inputs["epk_id"],
            "seq_hidden_state": seq_hidden_state,
            "target_attr_1": inputs["target_attr_1"],
            "report_month": inputs["report_month"],
        }
        for col in self.additional_columns:
            if col in inputs:
                pred_dict[col] = [str(val) for val in inputs[col]]
        self.preds.append(pred_dict)
        if len(self.preds) >= self.save_steps:
            self.save_parquet()

    @staticmethod
    def flatten_dict(original_dict: dict) -> dict:
        flat_dict = {
            f"{outer_key}_{inner_key}": value
            for outer_key, inner_dict in original_dict.items()
            for inner_key, value in inner_dict.items()
        }
        return flat_dict

    @staticmethod
    def calculate_auc_roc(
        ground_truth: np.array,
        predict: np.array,
        channel_type: np.array,
        control_flag: np.array,
    ):
        scores = {}

        unique_channels = np.unique(channel_type)
        unique_flags = np.unique(control_flag)
        for channel in unique_channels:
            for flag in unique_flags:
                mask = np.squeeze(channel_type == channel) & np.squeeze(
                    control_flag == flag
                )
                gt_masked = ground_truth[mask]
                predict_masked = predict[mask]
                if len(np.unique(gt_masked)) == 2:
                    scores[f"channel_type:{channel}_control_flag:{flag}"] = (
                        roc_auc_score(gt_masked, predict_masked)
                    )
                else:
                    scores[f"channel_type:{channel}_control_flag:{flag}"] = None

        return scores

    def catboost_pipeline(
        self,
        train_df: np.array,
        valid_df: np.array,
        test_df: np.array,
    ):
        x_train, y_train = (
            np.array(train_df["seq_hidden_state"].to_list()),
            np.array(train_df["target_attr_1"]),
        )
        x_valid = np.array(valid_df["seq_hidden_state"].to_list())
        x_test = np.array(test_df["seq_hidden_state"].to_list())

        model = CatBoostClassifier(**self.catboost_params)
        model.fit(x_train, y_train)

        y_valid_pred = model.predict_proba(x_valid)
        y_test_pred = model.predict_proba(x_test)

        return y_valid_pred, y_test_pred

    def compute(self):
        if len(self.preds) != 0:
            self.save_parquet()

        for contour in self.compute_contour:
            if self.repartition_by_product:
                contour_df = pd.read_parquet(
                    os.path.join(self.path_to_save, f"product_name={contour}")
                )
            else:
                contour_df = pd.read_parquet(self.path_to_save)
                contour_df = contour_df[contour_df["product_name"] == contour]

            report_month = pd.to_datetime(contour_df["report_month"])
            train_contour_df = contour_df[
                report_month <= pd.to_datetime(self.split["train"])
            ]
            valid_contour_df = contour_df[
                report_month == pd.to_datetime(self.split["valid"])
            ]
            test_contour_df = contour_df[
                report_month == pd.to_datetime(self.split["test"])
            ]

            valid_pred, test_pred = self.catboost_pipeline(
                train_df=train_contour_df,
                valid_df=valid_contour_df,
                test_df=test_contour_df,
            )

            valid_scores = self.calculate_auc_roc(
                ground_truth=np.array(valid_contour_df["target_attr_1"]),
                predict=valid_pred[:, 1],
                channel_type=np.array(valid_contour_df[self.channel_type]),
                control_flag=np.array(valid_contour_df[self.control_flag]),
            )
            test_scores = self.calculate_auc_roc(
                ground_truth=np.array(test_contour_df["target_attr_1"]),
                predict=test_pred[:, 1],
                channel_type=np.array(test_contour_df[self.channel_type]),
                control_flag=np.array(test_contour_df[self.control_flag]),
            )
            print(f"{contour}:\n{valid_scores=}\n{test_scores=}")
            self.contour_scores[f"{contour}_valid"] = valid_scores
            self.contour_scores[f"{contour}_test"] = test_scores

            del contour_df
            del train_contour_df
            del valid_contour_df
            del test_contour_df

        flatten_scores = self.flatten_dict(self.contour_scores)
        print(flatten_scores)
        return flatten_scores


# DERPRECATED
class InferenceMultiTaskCampaignMetrics(BaseMetric):
    def __init__(self, path_to_save, save_steps, prefix=None):
        self.preds = []
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        self.prefix = prefix

        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=False)

    def update(self, inputs, outputs):
        epk_id = inputs["epk_id"]
        if "task_type" in inputs:
            task_name = inputs["task_type"]
        elif "task_name" in inputs:
            task_name = inputs["task_name"].detach().contiguous().cpu().numpy()
        else:
            task_name = len(inputs["epk_id"]) * ["unk"]

        target_attr_2 = outputs.group.detach().contiguous().cpu().numpy()
        c_probs = outputs.control_probs.detach().contiguous().cpu().numpy()
        t_probs = outputs.treatment_probs.detach().contiguous().cpu().numpy()

        # Create the prediction dictionary
        pred_dict = {
            "epk_id": epk_id,
            "target_attr_2": target_attr_2,
            "control_probs": c_probs,
            "treatment_probs": t_probs,
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
            "control_probs": [],
            "treatment_probs": [],
            "task_name": [],
        }

        has_report_month = any("report_month" in item for item in self.preds)
        if has_report_month:
            predict["report_month"] = []

        for item in self.preds:
            predict["epk_id"].extend(item["epk_id"])
            predict["target_attr_2"].extend(item["target_attr_2"])
            predict["control_probs"].extend(item["control_probs"])
            predict["treatment_probs"].extend(item["treatment_probs"])
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
        time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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


class InferenceMultiTaskResponseMetrics(BaseMetric):
    def __init__(self, path_to_save, save_steps, prefix=None):
        self.preds = []
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        self.prefix = prefix

        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=False)

    def update(self, inputs, outputs):
        epk_id = inputs["epk_id"]
        if "task_type" in inputs:
            task_name = inputs["task_type"]
        elif "task_name" in inputs:
            if isinstance(inputs["task_name"], torch.Tensor):
                task_name = inputs["task_name"].detach().contiguous().cpu().numpy()
            else:
                task_name = inputs["task_name"]
        else:
            task_name = len(inputs["epk_id"]) * ["unk"]

        target_attr_2 = outputs.group.detach().contiguous().cpu().numpy()
        if "original_target_attr_2" in inputs:
            original_target_attr_2 = inputs["original_target_attr_2"]
        else:
            original_target_attr_2 = target_attr_2
        prediction = (
            torch.nn.functional.sigmoid(outputs.logits)
            .squeeze(1)
            .detach()
            .contiguous()
            .cpu()
            .numpy()
        )
        # Create the prediction dictionary
        pred_dict = {
            "epk_id": epk_id,
            "target_attr_2": target_attr_2,
            "original_target_attr_2": original_target_attr_2,
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
            "original_target_attr_2": [],
            "prediction": [],
            "task_name": [],
        }

        has_report_month = any("report_month" in item for item in self.preds)
        if has_report_month:
            predict["report_month"] = []

        for item in self.preds:
            predict["epk_id"].extend(item["epk_id"])
            predict["target_attr_2"].extend(item["target_attr_2"])
            predict["original_target_attr_2"].extend(item["original_target_attr_2"])
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
        time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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


# DEPRECATED
class InferenceCampaignMetrics(BaseMetric):
    def __init__(self, path_to_save, save_steps, prefix=None):
        self.preds = []
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        self.prefix = prefix
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=False)

    def update(self, inputs, outputs):
        epk_id = inputs["epk_id"]
        target_attr_1 = (
            inputs["targets"].cpu().numpy().tolist() if ("targets" in inputs) else None
        )

        target_attr_2 = inputs["target_attr_2"]
        target_attr_3 = inputs["target_attr_3"]

        predicted = (
            torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]
            .detach()
            .contiguous()
            .cpu()
            .tolist()
        )
        self.preds.append({
            "epk_id": epk_id,
            "target_attr_2": target_attr_2,
            "target_attr_3": target_attr_3,
            "predicted": predicted,
        })
        if target_attr_1 is not None and len(self.preds) > 0:
            self.preds[-1].update({"target_attr_1": target_attr_1})

        if len(self.preds) >= self.save_steps:
            self.compute()

    def compute(self):
        """return Dict(metric_name: value)"""
        predict = {
            "epk_id": [],
            "target_attr_2": [],
            "target_attr_3": [],
            "predicted": [],
        }
        if len(self.preds) > 0 and "target_attr_1" in self.preds[-1]:
            predict["target_attr_1"] = []

        for item in self.preds:
            predict["predicted"].extend(item["predicted"])
            predict["epk_id"].extend(item["epk_id"])
            predict["target_attr_2"].extend(item["target_attr_2"])
            predict["target_attr_3"].extend(item["target_attr_3"])
            if "target_attr_1" in self.preds[-1]:
                predict["target_attr_1"].extend(item["target_attr_1"])

        predict_df = pd.DataFrame().from_dict(predict)
        time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.prefix is not None:
            cur_path = (
                os.path.join(self.path_to_save, time_now) + f"_{self.prefix}.parquet"
            )
        else:
            cur_path = os.path.join(self.path_to_save, time_now) + ".parquet"
        predict_df.to_parquet(cur_path, index=False)
        self.reset()
        print(f"predict_saved: {cur_path}")

        return {}

    def reset(self):
        self.preds = []


class MLPCampaignBenchmark(BaseMetric):
    def __init__(
        self,
        path_to_config_dir: str,
        config_name: str,
        mlflow_run_name: str,
        mlflow_exp_name: str,
        path_to_save: str,
        save_steps: int,
        compute_contour: list[str],
        split: dict[str, str],
        additional_columns: list[str] = [],
        contour_name_column="product_name",
        prefix=None,
        cat_feature=False,
    ):
        self.config_name = config_name
        self.path_to_config_dir = path_to_config_dir
        self.mlflow_run_name = mlflow_run_name
        self.mlflow_exp_name = mlflow_exp_name
        self.compute_contour = compute_contour

        self.split = split
        self.path_to_save = (
            path_to_save + "/" + self.mlflow_run_name + "_" + self.mlflow_exp_name
        )
        self.save_steps = save_steps
        self.prefix = prefix
        self.preds = []
        self.contour_name_column = contour_name_column
        self.additional_columns = additional_columns
        self.cat_feature = cat_feature
        if self.cat_feature:
            assert (
                "target_attr_2" in self.additional_columns
                and "target_attr_3" in self.additional_columns
            ), "No columns target_attr_2 or target_attr_3 in additional_columns"

    def reset(self):
        self.preds = []

    def filter_and_save(self, df, split_type, contour):
        if split_type == "train":
            save_df = df[df["report_month"] <= pd.to_datetime(self.split[split_type])]
        else:
            save_df = df[df["report_month"] == pd.to_datetime(self.split[split_type])]

        save_to_parquet(
            df=save_df,
            path_to_save=self.path_to_save + f"/{contour}/{split_type}",
            prefix=self.prefix,
        )

    def save_parquet(self):
        predict = {
            "epk_id": [],
            "seq_hidden_state": [],
            "report_month": [],
            self.contour_name_column: [],
        }
        for col in self.additional_columns:
            if any(col in item for item in self.preds):
                predict[col] = []

        for item in self.preds:
            predict["epk_id"].extend(item["epk_id"])
            predict["seq_hidden_state"].extend(item["seq_hidden_state"])
            predict["report_month"].extend(item["report_month"])
            predict[self.contour_name_column].extend(item[self.contour_name_column])
            for col in self.additional_columns:
                if col in item:
                    predict[col].extend(item[col])
        if self.cat_feature:
            predict["cat_features"] = [
                [target_attr_2, target_attr_3]
                for target_attr_2, target_attr_3 in zip(
                    predict["target_attr_2"], predict["target_attr_3"]
                )
            ]
        predict_df = pd.DataFrame().from_dict(predict)
        for contour in self.compute_contour:
            contour_df = predict_df[predict_df[self.contour_name_column] == contour]
            self.filter_and_save(contour_df, "train", contour)
            self.filter_and_save(contour_df, "valid", contour)
            self.filter_and_save(contour_df, "test", contour)
        self.reset()

    def update(self, inputs, outputs):
        if hasattr(outputs, "aggregated_hidden_state"):
            seq_hidden_state = outputs.aggregated_hidden_state
        else:
            seq_hidden_state = outputs.last_hidden_state
        seq_hidden_state = seq_hidden_state.detach().contiguous().cpu().numpy()
        pred_dict = {
            "epk_id": inputs["epk_id"],
            "seq_hidden_state": seq_hidden_state,
            "report_month": inputs["report_month"],
            self.contour_name_column: inputs[self.contour_name_column],
        }
        for col in self.additional_columns:
            if col in inputs:
                pred_dict[col] = [val for val in inputs[col]]
        self.preds.append(pred_dict)
        if len(self.preds) >= self.save_steps:
            self.save_parquet()

    def mlp_pipeline(
        self,
        path_to_test,
        path_to_train,
        path_to_valid,
        path_to_save,
        contour,
    ):
        subprocess.run(
            [
                "accelerate",
                "launch",
                "-m",
                "avatar.train",
                "--config-dir",
                f"{self.path_to_config_dir}",
                "--config-name",
                f"{self.config_name}",
                f"mlflow.run_name={self.mlflow_run_name + '_' + contour}",
                f"mlflow.experiment_name={self.mlflow_exp_name}",
                f"train_dataloader.dataset.path={path_to_train}",
                f"valid_dataloader.dataset.path={path_to_valid}",
                f"test_dataloader.dataset.path={path_to_test}",
                f"PATH_TO_SAVE_PREDICT={path_to_save}",
            ],
            check=True,
        )

    def compute(self):
        if len(self.preds) != 0:
            self.save_parquet()

        for contour in self.compute_contour:
            self.mlp_pipeline(
                path_to_test=self.path_to_save + f"/{contour}/test",
                path_to_train=self.path_to_save + f"/{contour}/train",
                path_to_valid=self.path_to_save + f"/{contour}/valid",
                path_to_save=self.path_to_save + f"/{contour}/predict",
                contour=contour,
            )
        return {}
