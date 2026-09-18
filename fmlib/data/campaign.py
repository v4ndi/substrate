"""Fan one batch out over a campaign's (task x communication channel) matrix.

A campaign model is trained once and then scored for every combination the
campaign offers: each task it can be asked about, and each channel the offer
could go out through. The data does not carry those columns — they are what the
inference run is sweeping over — so they are written onto copies of the batch
here, one copy per combination.

This used to be an entrypoint of its own (``fmlib/cam_inference.py``), which
differed from :mod:`fmlib.inference` by exactly this loop. It is not an
entrypoint but an argument: :func:`fmlib.train.evaluate.predict` already takes
a ``prepare_batch`` callable, so the campaign sweep is configuration.

    prepare_batch:
      _target_: fmlib.data.campaign.CampaignTaskChannelBatches
      tasks: ${campaign_meta.tasks}
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import torch

__all__ = ["NO_CHANNEL", "CampaignTaskChannelBatches"]

#: Channel id standing in for "this task has no channel dimension".
NO_CHANNEL = -1


class CampaignTaskChannelBatches:
    """Turn one batch into one batch per (task, channel) pair.

    Args:
        tasks: The campaign's task map — task name to a mapping with a
            ``comm_type`` list of channel ids. A ``None`` channel means the task
            has no channel dimension; it is scored once, with ``group`` left
            unset and the channel column written as :data:`NO_CHANNEL`.
    """

    def __init__(self, tasks: Mapping[str, Any]):
        self.tasks = tasks

    def __call__(self, batch: dict) -> Iterator[dict]:
        """Yield one shallow copy of ``batch`` per combination."""
        device = batch["tab_features"].cat_features.device
        size = len(batch["epk_id"])

        for task_name, task_meta in self.tasks.items():
            try:
                task_values: Any = torch.LongTensor([int(task_name)] * size).to(device)
            except (TypeError, ValueError):
                # Some campaigns key tasks by name rather than by id.
                task_values = [task_name] * size

            for comm_type in task_meta["comm_type"]:
                channel_values = [NO_CHANNEL if comm_type is None else comm_type] * size
                variant = dict(batch)
                variant["task_name"] = task_values
                variant["target_attr_2"] = channel_values
                variant["group"] = (
                    None
                    if comm_type is None
                    else torch.LongTensor(channel_values).to(device)
                )
                # A campaign scores the treated arm; control comes out of the
                # same forward pass as the other head.
                if "is_treat" not in variant:
                    variant["is_treat"] = torch.ones(size, device=device).long()
                yield variant
