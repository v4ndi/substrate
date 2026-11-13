"""Inference entrypoint.

    python -m avatar.inference --config-dir=configs --config-name=inference

Multi-rank inference works the same way training does — launch it under
``torchrun`` and give it a dataset that shards itself. A replicated dataset
would have every rank score the same rows, so the metrics are computed on
rank 0 only in that case.

A config may set ``prepare_batch`` to a callable that rewrites each batch before
it is scored; campaign inference uses it to sweep the (task x channel) matrix.
"""

from __future__ import annotations

import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from avatar.train.config import resolve_run_config
from avatar.train.dist import DistEnv, seed_everything
from avatar.train.evaluate import predict
from avatar.train.utils import prefix_metrics

os.environ.setdefault("HYDRA_FULL_ERROR", "1")


def log_scores(mlflow_arguments: dict | None, scores: dict, prefix: str = "") -> None:
    """Record inference scores as a standalone MLflow run."""
    if not mlflow_arguments or not scores:
        return
    import mlflow

    if mlflow_arguments.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_arguments["tracking_uri"])
    if mlflow_arguments.get("experiment_name"):
        mlflow.set_experiment(mlflow_arguments["experiment_name"])
    with mlflow.start_run(run_name=mlflow_arguments.get("run_name")):
        mlflow.log_metrics({
            key: float(value)
            for key, value in prefix_metrics(scores, prefix).items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        })


@hydra.main(version_base=None, config_path=".", config_name="inference")
def main(config: DictConfig) -> None:
    """Load weights, score ``test_dataloader`` and report the metrics."""
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
    if load_state is not None:
        model.load_state_dict(
            torch.load(load_state, map_location="cpu", weights_only=False), strict=True
        )
    else:
        warnings.warn("\n\nNo load_state provided\n\n", stacklevel=2)

    env.print(model)
    model = model.to(env.device)

    metrics = None
    if "metrics" in config and config["metrics"] is not None:
        metrics = instantiate(config["metrics"]["test_metrics"])

    # Optional batch rewriting, the campaign sweep being the one that exists:
    # see :class:`avatar.data.campaign.CampaignTaskChannelBatches`.
    prepare_batch = None
    if config.get("prepare_batch") is not None:
        prepare_batch = instantiate(config["prepare_batch"])

    try:
        scores = predict(
            model, test_dataloader, env, metrics, prepare_batch=prepare_batch
        )
        if env.is_main and scores:
            env.print(scores)
            mlflow_arguments = (
                OmegaConf.to_container(config["mlflow"])
                if "mlflow" in config and config["mlflow"] is not None
                else None
            )
            log_scores(mlflow_arguments, scores, prefix="val_")
    finally:
        env.destroy()


if __name__ == "__main__":
    main()
