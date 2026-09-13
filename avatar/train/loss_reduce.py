"""Cross-rank loss reduction for multi-head outputs.

Lifted from ``avatar.train_utils.calculate_output_loss`` with the
``Accelerator`` argument replaced by :class:`~avatar.train.dist.DistEnv`.
"""

from __future__ import annotations

from typing import Any

import torch

from avatar.train.dist import DistEnv


def calculate_output_loss(
    output: Any, env: DistEnv, distributed: bool = True
) -> torch.Tensor:
    """Return the scalar the loop should call ``.backward()`` on.

    Outputs with a plain ``loss`` pass straight through. Multi-head outputs
    carry ``losses`` (per head) and ``num_items`` (valid tokens per head): those
    are averaged with a *global* token count, so a rank that happened to get
    shorter sequences does not weigh more per token than its neighbours.

        loss_head = loss_head_local * world_size / sum_over_ranks(num_items_head)

    The ``world_size`` factor cancels DDP's mean gradient reduction, leaving a
    true token-weighted sum.
    """
    if not hasattr(output, "num_items") or output.num_items is None:
        return output.loss

    local_num_items = output.num_items
    if distributed and env.distributed:
        gathered = env.gather_objects({
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in local_num_items.items()
        })
        num_items = {
            key: sum(float(part[key].sum()) for part in gathered)
            for key in local_num_items
        }
        num_proc = env.world_size
    else:
        num_items = {
            key: float(value.sum()) if torch.is_tensor(value) else float(value)
            for key, value in local_num_items.items()
        }
        num_proc = 1

    loss = 0.0
    for key, value in num_items.items():
        if value != 0:
            loss = loss + output.losses[key] * num_proc / value
    return loss / len(num_items)
