"""Campaign-scale collectors: dump embeddings and per-record predictions.

Both classes here are :class:`~fmlib.metrics.base.ArtifactMetric` — their
product is a parquet file, not a number. They flush every ``save_steps``
batches, so a run over tens of millions of records never holds the population in
memory.
"""

from __future__ import annotations

import pandas as pd

from .base import ArtifactMetric


class CollectEmbeddings(ArtifactMetric):
    """Collect and save sequence embeddings, plus any extra input columns.

    Args:
        path_to_save: Directory where the parquet parts will be saved.
        save_steps: Flush every this many batches.
        prefix: Optional tag in the filename, to tell runs apart.
        additional_columns: Input columns to carry into the output alongside the
            embedding.
        month_part_value: When set, written into a ``month_part`` column — the
            campaign contour the dump belongs to.
    """

    required_outputs = ("aggregated_hidden_state", "last_hidden_state")

    def __init__(
        self,
        path_to_save: str,
        save_steps: int,
        prefix: str | None = None,
        additional_columns: list[str] | None = None,
        month_part_value: str | None = None,
    ):
        super().__init__(path_to_save=path_to_save, prefix=prefix)
        self.preds: list[dict] = []
        self.save_steps = save_steps
        self.additional_columns = additional_columns or []
        self.month_part_value = month_part_value
        self.required_inputs = ("epk_id", *self.additional_columns)

    def update(self, inputs, outputs):
        """Accumulate one batch of embeddings, flushing when the buffer is full.

        Args:
            inputs: Batch dictionary; must include ``epk_id``.
            outputs: Model output carrying a pooled representation.
        """
        # ``getattr(..., None)`` rather than ``hasattr``: a narrowed output
        # keeps every field and blanks the ones no metric asked for, so the
        # attribute exists either way and only its value tells them apart.
        seq_hidden_state = getattr(outputs, "aggregated_hidden_state", None)
        if seq_hidden_state is None:
            seq_hidden_state = outputs.last_hidden_state
        seq_hidden_state = seq_hidden_state.detach().contiguous().cpu().numpy()

        pred_dict = {
            "epk_id": inputs["epk_id"],
            "seq_hidden_state": seq_hidden_state,
        }
        for column in self.additional_columns:
            pred_dict[column] = list(inputs[column])

        self.preds.append(pred_dict)

        if len(self.preds) >= self.save_steps:
            self.flush()

    def flush(self) -> None:
        """Write the buffered embeddings to a parquet part and forget them."""
        if not self.preds:
            return

        predict: dict[str, list] = {"epk_id": [], "seq_hidden_state": []}
        for column in self.additional_columns:
            if any(column in item for item in self.preds):
                predict[column] = []

        for item in self.preds:
            predict["epk_id"].extend(item["epk_id"])
            predict["seq_hidden_state"].extend(item["seq_hidden_state"])
            for column in self.additional_columns:
                if column in item:
                    predict[column].extend(item[column])

        frame = pd.DataFrame.from_dict(predict)
        if self.month_part_value is not None:
            frame["month_part"] = self.month_part_value
        self.save_dataframe(frame)
        self.reset()

    def reset(self):
        """Drop the buffered batches."""
        self.preds = []


class InferenceMultiTaskCampaignMetrics(ArtifactMetric):
    """Write per-record uplift predictions for a multi-task campaign model.

    Collects ``epk_id``, the task name, the group and both head probabilities,
    flushing to parquet every ``save_steps`` batches.

    Args:
        path_to_save: Directory for the parquet parts; created if missing.
        save_steps: Flush every this many batches.
        prefix: Optional tag in the filename, to tell runs apart.
    """

    required_inputs = ("epk_id", "task_type", "task_name", "report_month")
    required_outputs = ("group", "control_probs", "treatment_probs")

    def __init__(self, path_to_save: str, save_steps: int, prefix: str | None = None):
        super().__init__(path_to_save=path_to_save, prefix=prefix)
        self.preds: list[dict] = []
        self.save_steps = save_steps

    def update(self, inputs, outputs):
        """Accumulate one batch of predictions, flushing when the buffer is full."""
        if "task_type" in inputs:
            task_name = inputs["task_type"]
        elif "task_name" in inputs:
            task_name = inputs["task_name"].detach().contiguous().cpu().numpy()
        else:
            task_name = len(inputs["epk_id"]) * ["unk"]

        pred_dict = {
            "epk_id": inputs["epk_id"],
            "target_attr_2": outputs.group.detach().contiguous().cpu().numpy(),
            "control_probs": outputs.control_probs.detach().contiguous().cpu().numpy(),
            "treatment_probs": outputs.treatment_probs.detach()
            .contiguous()
            .cpu()
            .numpy(),
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

        columns = [
            "epk_id",
            "target_attr_2",
            "control_probs",
            "treatment_probs",
            "task_name",
        ]
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

    def reset(self):
        """Drop the buffered batches."""
        self.preds = []
