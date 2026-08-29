import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .base import BaseMetric


class RocAucScore(BaseMetric):
    """Calculate RocAucScore with group support

    Args:
        group_column: str - Specifies the column used to partition the data
        The RocAuc score will be calculated separately for each unique value
        in this column and also averaged across groups
    """

    def __init__(self, group_column: str | None = None):
        super().__init__()
        self.group_column = group_column
        self.preds = []
        self.groups = []

    def update(self, inputs, outputs):
        targets = inputs["targets"].detach().contiguous().cpu().tolist()

        if outputs.logits.dim() == 1:
            predicted = (
                torch.nn.functional.sigmoid(outputs.logits)
                .detach()
                .contiguous()
                .cpu()
                .tolist()
            )
        else:
            predicted = (
                torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]
                .detach()
                .contiguous()
                .cpu()
                .tolist()
            )

        self.preds.append({"targets": targets, "probability": predicted})

        # Store group information if group_column is specified
        if self.group_column is not None and self.group_column in inputs:
            if isinstance(inputs[self.group_column], torch.Tensor):
                groups = inputs[self.group_column].detach().contiguous().cpu().tolist()
            elif isinstance(inputs[self.group_column], list):
                groups = inputs[self.group_column]
            else:
                group_type = type(inputs[self.group_column])
                raise ValueError(
                    f"inputs[self.group_column] must be a list or torch.Tensor, but got {group_type}"
                )
            self.groups.extend(groups)

    def compute(self) -> dict[str, float]:
        """return Dict(metric_name: value)"""
        predicted = []
        targets = []

        for item in self.preds:
            predicted.extend(item["probability"])
            targets.extend(item["targets"])

        result = {}
        # Standard ROC AUC calculation without groups
        result["roc_auc_score"] = (
            roc_auc_score(targets, predicted)
            if len(targets) != 0 and sum(targets) > 0
            else 0.5
        )

        if self.group_column is not None:
            # Calculate ROC AUC for each group and their average
            if len(self.groups) != len(targets):
                raise ValueError(
                    f"Length of groups ({len(self.groups)}) doesn't match "
                    f"length of targets ({len(targets)})"
                )

            # Convert to numpy arrays for easier manipulation
            groups_array = np.array(self.groups)
            targets_array = np.array(targets)
            predicted_array = np.array(predicted)

            unique_groups = np.unique(groups_array)
            group_scores = []

            for group in unique_groups:
                mask = groups_array == group
                group_targets = targets_array[mask]
                group_predicted = predicted_array[mask]

                if len(np.unique(group_targets)) < 2:
                    # Skip groups with only one class
                    continue

                score = roc_auc_score(group_targets, group_predicted)
                result[f"roc_auc_score_group_{group}"] = score
                group_scores.append(score)

            # Calculate average score across groups
            if group_scores:
                result["roc_auc_score_mean"] = np.mean(group_scores)
                result["roc_auc_score_std"] = np.std(group_scores)

            # Also include overall score
            if len(np.unique(targets_array)) >= 2:
                result["roc_auc_score_overall"] = roc_auc_score(
                    targets_array, predicted_array
                )

        return result

    def reset(self):
        self.preds = []
        self.groups = []
