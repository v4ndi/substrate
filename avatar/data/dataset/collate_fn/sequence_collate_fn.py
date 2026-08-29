from functools import partial, reduce
from operator import iadd
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from avatar.data.dataset.collate_fn.base_collate_fn import BaseCollateFn
from avatar.data.dataset.collate_fn.tabular_collate_fn import TabularCollateFn
from avatar.data.event_seq_batch import EventSequenceBatch


class EventSequenceCollateFn(BaseCollateFn):
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
        length_to_pad: int | None = None,
    ) -> None:
        self.sequence_columns = sequence_columns
        self.create_attention_mask = create_attention_mask
        self.pad_token = 0
        self.target_column = target_column
        self.is_regression = is_regression
        self.has_tabular = has_tabular
        self.not_sequence_columns = None
        self.length_to_pad = length_to_pad

    @staticmethod
    def collate_sequence(
        batch,
        pad_token: int,
        create_attention_mask: bool,
        sequence_columns: list[str],
        timestamps_column: str = "_timestamps",
        length_to_pad: int | None = None,
    ):
        events = {}

        if length_to_pad is not None:
            pad_fn = partial(
                EventSequenceCollateFn.pad_to_max_len, max_length=length_to_pad
            )
        else:
            pad_fn = partial(pad_sequence, batch_first=True)

        for column in sequence_columns:
            items = [item[column] for item in batch]
            if len(items[0]) == 0:
                items[0] = F.pad(items[0], pad=(0, 1), value=pad_token)
            if items[0].dtype == torch.int64:
                events[column] = pad_fn(items, padding_value=pad_token)
            elif len(items[0].shape) == 0:
                events[column] = torch.stack(items)
            elif batch[0][column].dtype == torch.float32:
                events[column] = pad_fn(items, padding_value=0.0)

        if timestamps_column in batch[0]:
            items = [item[timestamps_column] for item in batch]
            if len(items[0]) == 0:
                items[0] = F.pad(items[0], pad=(0, 1), value=0.0)
            timestamps = pad_sequence(items, batch_first=True, padding_value=0.0)
        else:
            timestamps = None

        if create_attention_mask:
            attn_msk = [
                torch.ones(item[sequence_columns[0]].shape[0]) for item in batch
            ]
            if attn_msk[0].size(0) == 0:
                attn_msk[0] = torch.zeros(1)
            attention_mask = pad_fn(attn_msk, padding_value=0).long()
        else:
            attention_mask = None

        if "_event_ids" in batch[0]:
            event_ids = []
            for item in batch:
                event_ids.append(item["_event_ids"])
            if len(event_ids[0]) == 0:
                event_ids[0] = F.pad(event_ids[0], pad=(0, 1), value=-100)
            event_ids = pad_sequence(
                event_ids,
                batch_first=True,
                padding_value=-100,  # to ignore in CE loss
            ).long()
        else:
            event_ids = None
        return EventSequenceBatch(
            events=events,
            timestamps=timestamps,
            attention_mask=attention_mask,
            event_ids=event_ids,
        )

    @staticmethod
    def pad_to_max_len(sequences, max_length, padding_value):
        dummy_tensor = torch.full(
            (max_length,), padding_value, dtype=sequences[0].dtype
        )
        seq_with_dummy = [*sequences, dummy_tensor]
        padded = pad_sequence(
            seq_with_dummy, batch_first=True, padding_value=padding_value
        )

        return padded[:-1]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if isinstance(batch[0], partial):
            batch = [process_func() for process_func in batch]
        seq_batch = EventSequenceCollateFn.collate_sequence(
            batch=batch,
            pad_token=self.pad_token,
            create_attention_mask=self.create_attention_mask,
            sequence_columns=self.sequence_columns,
            length_to_pad=self.length_to_pad,
        )
        output = {"seq_features": seq_batch}
        if self.not_sequence_columns is None:
            self.not_sequence_columns = [
                key
                for key in batch[0].keys()
                if key not in [*self.sequence_columns, "_event_ids", "_timestamps"]
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


class ColesCollateFn(EventSequenceCollateFn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, batch: list[list[dict[str, Any]]]
    ) -> tuple[dict[str, Any], torch.LongTensor]:
        class_labels = None
        if isinstance(batch[0], list):
            class_labels = [
                i for i, class_samples in enumerate(batch) for _ in class_samples
            ]
            batch = reduce(iadd, batch)
            class_labels = torch.LongTensor(class_labels)
        output = super().__call__(batch)
        output["seq_features"]._targets = class_labels
        return output
