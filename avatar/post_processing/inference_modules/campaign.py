import datetime
import os

import pandas as pd
import torch

from .base import BaseInference


# TODO: rewrite and remove literals and change naming mb
class InferenceCampaign(BaseInference):
    def __init__(self, path_to_save, save_steps, prefix=None):
        self.preds = []
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        self.prefix = prefix

        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=True)

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
