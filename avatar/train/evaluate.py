"""Evaluation loop on plain ``torch.distributed``.

Two reduction modes, picked from whether the dataset shards itself:

* **sharded** — every rank walks its own slice, then ``all_gather_object``
  brings ``(batch, output)`` to rank 0, which feeds the metrics. Non-additive
  metrics (ROC-AUC and friends) need the whole population in one place.
* **replicated** — every rank sees the same records, so gathering would count
  each sample ``world_size`` times. Every rank still runs the forward pass (so
  nobody blocks at a collective the others never reach), but only rank 0 scores.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from avatar.metrics import BaseMetric
from avatar.train.dist import DistEnv, unwrap_model
from avatar.train.loss_reduce import calculate_output_loss
from avatar.train.utils import move_to_device, wrap_metrics


def dataloader_is_sharded(dataloader: DataLoader) -> bool:
    """True when the dataset already split the record stream by rank."""
    return bool(getattr(dataloader.dataset, "shard", False))


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    env: DistEnv,
    metrics: BaseMetric | list[BaseMetric] | None = None,
    description: str = "Validation step",
    progress: bool = True,
) -> dict[str, Any] | None:
    """Run the model over ``dataloader`` and compute metrics.

    Args:
        model: Module to evaluate. May be DDP- or ``AveragedModel``-wrapped.
        dataloader: Validation or test loader.
        env: Distributed environment; decides the reduction mode.
        metrics: One metric, a list of them, or None for loss only.
        description: Progress-bar label.
        progress: Whether to draw a progress bar on the local main process.

    Returns:
        The score dict on rank 0, ``None`` on every other rank.
    """
    # DDP's forward only exists to sync gradients; under inference_mode it adds
    # nothing but a chance to deadlock, so evaluate the bare module.
    model = unwrap_model(model)
    model.eval()
    sharded = dataloader_is_sharded(dataloader)
    gather = env.distributed and sharded

    metrics = wrap_metrics(metrics)
    bar = tqdm(
        dataloader,
        desc=description,
        disable=not (progress and env.is_local_main),
    )

    local_loss = 0.0
    num_steps = 0
    for batch in bar:
        num_steps += 1
        batch = move_to_device(batch, env.device)
        output = model(**batch)
        loss = calculate_output_loss(output, env, distributed=gather)
        local_loss += float(loss.detach())

        if metrics is None:
            continue
        pairs = env.gather_objects((batch, output)) if gather else [(batch, output)]
        if not env.is_main:
            continue
        for metric in metrics:
            for inputs, outputs in pairs:
                metric.update(inputs=inputs, outputs=outputs)

    # Sharded ranks each saw a different slice, so the mean loss is a global
    # reduction; replicated ranks all computed the same number, so rank 0's is
    # already the answer.
    if gather:
        total_loss = env.all_reduce_sum(local_loss)
        total_steps = env.all_reduce_sum(float(num_steps))
    else:
        total_loss, total_steps = local_loss, float(num_steps)

    if not env.is_main:
        return None

    scores: dict[str, Any] = {"loss": total_loss / total_steps if total_steps else 0.0}
    if metrics is not None:
        for metric in metrics:
            scores.update(metric.compute())
            metric.reset()
    return scores


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    env: DistEnv,
    metrics: BaseMetric | list[BaseMetric] | None = None,
    description: str = "Inference",
    prepare_batch=None,
) -> dict[str, Any] | None:
    """Inference variant of :func:`evaluate` — no loss, optional batch rewriting.

    ``prepare_batch`` receives the on-device batch and yields one or more
    batches to score; campaign inference uses it to fan a batch out over
    (task, channel) combinations.
    """
    model = unwrap_model(model)
    model.eval()
    sharded = dataloader_is_sharded(dataloader)
    gather = env.distributed and sharded

    metrics = wrap_metrics(metrics)
    bar = tqdm(dataloader, desc=description, disable=not env.is_local_main)

    for batch in bar:
        batch = move_to_device(batch, env.device)
        variants = [batch] if prepare_batch is None else prepare_batch(batch)
        for variant in variants:
            output = model(**variant)
            if metrics is None:
                continue
            pairs = (
                env.gather_objects((variant, output)) if gather else [(variant, output)]
            )
            if not env.is_main:
                continue
            for metric in metrics:
                for inputs, outputs in pairs:
                    metric.update(inputs=inputs, outputs=outputs)

    env.barrier()
    if not env.is_main or metrics is None:
        return None
    scores: dict[str, Any] = {}
    for metric in metrics:
        scores.update(metric.compute())
        metric.reset()
    return scores
