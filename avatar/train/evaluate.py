"""Evaluation loop on plain ``torch.distributed``.

Three reduction modes, picked from what the metrics need and from whether the
dataset shards itself:

* **gather** — the dataset is sharded and at least one metric needs the whole
  population on one rank (ROC-AUC, Qini, calibration are not sums). Every rank
  walks its own slice and ``gather_object`` brings ``(batch, output)`` to rank
  0, which feeds the metrics.
* **replicated** — every rank sees the same records, so gathering would count
  each sample ``world_size`` times. Every rank still runs the forward pass (so
  nobody blocks at a collective the others never reach), but only rank 0 scores.
* **local** — the dataset is sharded and no metric needs the population, which
  is the case for every collector whose product is a parquet directory. Each
  rank feeds its own metrics with its own rows and writes its own part files.
  No payload crosses the wire, and — because nothing collective happens inside
  the loop — the ranks do not have to agree on how many batches they have.

Only :func:`predict` has all three. :func:`evaluate` runs during training, where
``calculate_output_loss`` is itself a collective and DDP requires equal batch
counts anyway, so it keeps the original two.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from avatar.metrics import BaseMetric
from avatar.train.dist import DistEnv, unwrap_model
from avatar.train.loss_reduce import calculate_output_loss
from avatar.train.utils import (
    metric_field_selection,
    move_to_device,
    narrow_for_metrics,
    wrap_metrics,
)


def dataloader_is_sharded(dataloader: DataLoader) -> bool:
    """True when the dataset already split the record stream by rank."""
    return bool(getattr(dataloader.dataset, "shard", False))


def warn_about_dropped_tail(dataloader: DataLoader, env: DistEnv) -> None:
    """Say out loud how many records this run will not score.

    ``drop_tail`` exists for training, where every rank must produce the same
    number of gradient steps. At inference it means ``total % world_size``
    records get no prediction at all, and the only sign of it was a line in the
    shard scan that reads like a statistic.
    """
    dataset = dataloader.dataset
    if not env.distributed or not getattr(dataset, "drop_tail", False):
        return
    planner = getattr(dataset, "_planner", None)
    if planner is None or not planner.tail:
        return
    warnings.warn(
        f"{planner.tail} of {planner.total} records will not be scored: the "
        f"dataset shards with drop_tail=True, which drops the remainder that "
        f"does not divide by world_size={env.world_size}. Inference does not "
        f"need it — set drop_tail: false on the dataset.",
        stacklevel=2,
    )


def metrics_need_population(metrics: list[BaseMetric] | None) -> bool:
    """True when any metric has to see every record on one rank.

    A metric that says nothing is assumed to need it: the expensive answer is
    the safe one, and a metric that silently scored a slice of the population
    would report a number nobody could tell was wrong.
    """
    return any(
        getattr(metric, "needs_full_population", True) for metric in metrics or []
    )


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
    selection = metric_field_selection(metrics)
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
        payload = narrow_for_metrics(batch, output, selection)
        pairs = env.gather_objects(payload) if gather else [payload]
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

    Returns:
        The score dict on rank 0, ``None`` on every other rank. Every rank
        still flushes its own metrics, which is what writes the part files in
        ``local`` mode.
    """
    model = unwrap_model(model)
    model.eval()

    metrics = wrap_metrics(metrics)
    selection = metric_field_selection(metrics)
    sharded = dataloader_is_sharded(dataloader)
    gather = env.distributed and sharded and metrics_need_population(metrics)
    # A replicated dataset gives every rank the same rows, so only rank 0 may
    # score them; a sharded one in ``local`` mode gives every rank rows of its
    # own, which every rank scores itself.
    scoring = env.is_main or (sharded and not gather)

    warn_about_dropped_tail(dataloader, env)
    bar = tqdm(dataloader, desc=description, disable=not env.is_local_main)

    def payloads():
        """Everything this rank has to hand to the metrics, in order."""
        for batch in bar:
            batch = move_to_device(batch, env.device)
            variants = [batch] if prepare_batch is None else prepare_batch(batch)
            for variant in variants:
                output = model(**variant)
                if metrics is not None:
                    yield narrow_for_metrics(variant, output, selection)

    def feed(pairs) -> None:
        for metric in metrics:
            for inputs, outputs in pairs:
                metric.update(inputs=inputs, outputs=outputs)

    if not gather:
        for payload in payloads():
            if scoring:
                feed([payload])
    else:
        # ``gather_object`` is collective: every rank has to reach it the same
        # number of times. The ranks are not obliged to hold the same number of
        # batches — an evaluation shard keeps its tail — so the loop runs until
        # the last rank is done and the ones that finished early send nothing.
        local = payloads()
        while True:
            payload = next(local, None)
            if not env.all_reduce_any(payload is not None):
                break
            pairs = env.gather_objects(payload)
            if env.is_main:
                feed([pair for pair in pairs if pair is not None])

    env.barrier()
    if metrics is None:
        return None
    # Every rank computes, because for a collector ``compute`` is what writes
    # the tail of its buffer. Only rank 0's numbers are returned.
    scores: dict[str, Any] = {}
    for metric in metrics:
        produced = metric.compute() if scoring else {}
        metric.reset()
        scores.update(produced)
    return scores if env.is_main else None
