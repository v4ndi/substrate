from typing import Any

import torch

from avatar.data.dataset.collate_fn.sequence_collate_fn import EventSequenceCollateFn
from avatar.data.dataset.collate_fn.tabular_collate_fn import TabularCollateFn


class FixedHorizonCollateFn(EventSequenceCollateFn):
    """Collate Function for EventSequenceDataset
    Args:
        sequence_columns: List[str] - list of sequence columns
        create_attention_mask: bool - whether to create attention mask
        target_column: Optional[str] - target column
        is_regression: bool - whether the target is regression
        has_tabular: bool - whether the dataset has tabular data
    """

    def __init__(
        self,
        sequence_columns: list[str],
        create_attention_mask: bool = True,
        target_column: str | None = None,
        is_regression: bool = False,
        has_tabular: bool = False,
        ctx_columns: list[str] | None = None,
        target_columns: list[str] | None = None,
    ) -> None:
        if target_columns is None:
            target_columns = []
        if ctx_columns is None:
            ctx_columns = []
        super().__init__(
            sequence_columns=sequence_columns,
            create_attention_mask=create_attention_mask,
            target_column=target_column,
            is_regression=is_regression,
            has_tabular=has_tabular,
        )
        self.ctx_columns = ctx_columns
        self.target_columns = target_columns

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        seq_batch = []
        if len(self.target_columns) > 0:
            timestamps_columns = ["_timestamps", "ctx_timestamps", "targets_timestamps"]
        else:
            timestamps_columns = ["_timestamps", "ctx_timestamps"]
        all_columns = []
        for column in [self.sequence_columns, self.ctx_columns, self.target_columns]:
            if len(column) > 0:
                all_columns.append(column)
        for i, columns in enumerate(all_columns):
            seq_batch.append(
                super().collate_sequence(
                    batch=batch,
                    pad_token=self.pad_token,
                    create_attention_mask=self.create_attention_mask,
                    sequence_columns=columns,
                    timestamps_column=timestamps_columns[i],
                )
            )
        if len(self.target_columns) > 0:
            output = {
                "seq_features": seq_batch[0],
                "ctx_features": seq_batch[1],
                "targets_features": seq_batch[2],
            }
        else:
            output = {"seq_features": seq_batch[0], "ctx_features": seq_batch[1]}
        if self.not_sequence_columns is None:
            self.not_sequence_columns = [
                key
                for key in batch[0].keys()
                if key
                not in [
                    *self.sequence_columns,
                    "event_ids",
                    "evt_dttm",
                    *timestamps_columns,
                    *self.ctx_columns,
                    *self.target_columns,
                ]
            ]
        for key in self.not_sequence_columns:
            if key not in self.sequence_columns:
                output[key] = self.extract_values(batch=batch, column_name=key)

        if self.target_column is not None:
            targets = self.extract_values(batch=batch, column_name=self.target_column)

            if self.is_regression:
                output["targets"] = torch.FloatTensor(targets)
            else:
                output["targets"] = torch.LongTensor(targets)

        if self.has_tabular:
            output["tab_features"] = TabularCollateFn.collate_tabular(batch=batch)

        return output
