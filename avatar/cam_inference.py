"""Campaign inference: score every (task, communication channel) combination.

Same loop as :mod:`avatar.infer`, but each batch is fanned out over the
campaign's tasks and channels before being scored, so one pass over the data
yields a prediction per combination.
"""

from __future__ import annotations

import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from avatar.train.config import resolve_run_config
from avatar.train.dist import DistEnv, seed_everything
from avatar.train.evaluate import predict

os.environ.setdefault("HYDRA_FULL_ERROR", "1")

# Channel id standing in for "this task has no channel dimension".
NO_CHANNEL = -1


def make_campaign_batches(campaign_meta: DictConfig):
    """Build the ``prepare_batch`` hook for a campaign's task/channel matrix.

    Args:
        campaign_meta: The ``campaign_meta`` config block; its ``tasks`` mapping
            gives the ``comm_type`` list for each task.

    Returns:
        A callable turning one batch into one batch per (task, channel) pair.
    """

    def prepare(batch: dict):
        device = batch["tab_features"].cat_features.device
        size = len(batch["epk_id"])
        for task_name, task_meta in campaign_meta["tasks"].items():
            try:
                task_values = torch.LongTensor([int(task_name)] * size).to(device)
            except (TypeError, ValueError):
                # Some campaigns key tasks by name rather than id.
                task_values = [task_name] * size
            for comm_type in task_meta["comm_type"]:
                channel = NO_CHANNEL if comm_type is None else comm_type
                channel_values = [channel] * size
                variant = dict(batch)
                variant["task_name"] = task_values
                variant["target_attr_2"] = channel_values
                variant["group"] = (
                    None
                    if comm_type is None
                    else torch.LongTensor(channel_values).to(device)
                )
                if "is_treat" not in variant:
                    variant["is_treat"] = torch.ones(size, device=device).long()
                yield variant

    return prepare


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    """Load weights and score the campaign's task/channel matrix."""
    run_config = resolve_run_config(config)
    env = DistEnv.from_env(
        backend=run_config.distributed.backend,
        timeout_sec=run_config.distributed.timeout_sec,
    )
    seed_everything(42, rank=env.rank)
    env.print(OmegaConf.to_yaml(config))

    model = instantiate(config["model"])
    test_dataloader = instantiate(config["test_dataloader"])

    load_state = config.get("load_state")
    if load_state is None:
        raise ValueError("load state is None")
    model.load_state_dict(
        torch.load(load_state, map_location="cpu", weights_only=False), strict=True
    )
    env.print(model)
    model = model.to(env.device)

    metrics = instantiate(config["metrics"]["test_metrics"])
    try:
        predict(
            model,
            test_dataloader,
            env,
            metrics,
            description="Campaign inference",
            prepare_batch=make_campaign_batches(config["campaign_meta"]),
        )
    finally:
        env.destroy()


if __name__ == "__main__":
    main()
