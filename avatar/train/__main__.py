"""Training entrypoint.

    torchrun --standalone --nproc_per_node=<gpus> -m avatar.train \
        --config-dir=configs --config-name=<name>

Single-process runs need no launcher: ``python -m avatar.train ...`` sees no
``WORLD_SIZE`` and runs with world size 1 and no process group.

Multi-node is the same command per node with ``--nnodes`` / ``--node_rank`` /
``--rdzv_*``.

NCCL tuning is deliberately *not* set here. Import-time ``os.environ`` writes
are invisible at the call site and one of the old ones
(``CUDA_LAUNCH_BLOCKING=1``) serialises every CUDA call. Set what you need in
the launcher script instead.
"""

from __future__ import annotations

import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from avatar.train.config import resolve_run_config
from avatar.train.dist import DistEnv, seed_everything
from avatar.train.factory import build_callbacks, resolve_performance_config
from avatar.train.loop import Trainer
from avatar.training_arguments import TrainingArguments
from avatar.utils.init_modules import (
    init_dataloaders,
    init_early_stopping,
    init_exp_run_name,
    init_metrics,
    init_optimizer,
    init_scheduler,
)
from avatar.utils.logging import print_meta, print_train_info
from avatar.utils.performance_metrics import configure_mlflow_system_metrics

os.environ.setdefault("HYDRA_FULL_ERROR", "1")


@hydra.main(version_base=None, config_path="..", config_name="config-td")
def main(config: DictConfig) -> None:
    """Build the run from config and hand it to :class:`~avatar.train.loop.Trainer`.

    Args:
        config: Hydra config with ``model``, ``train_dataloader``, ``optimizer``,
            ``scheduler``, ``train`` and (optionally) ``mlflow``, ``metrics``,
            ``logging``, ``distributed``/``amp``/``ddp`` sections.
    """
    startup_begin_time = time.perf_counter()
    run_config = resolve_run_config(config)
    env = DistEnv.from_env(
        backend=run_config.distributed.backend,
        timeout_sec=run_config.distributed.timeout_sec,
    )

    train_config = OmegaConf.to_container(config["train"])
    early_stopping = init_early_stopping(train_config)
    training_arguments = TrainingArguments(**train_config)
    seed_everything(training_arguments.seed, rank=env.rank)

    experiment_name, run_name = init_exp_run_name(config)
    if "root_dir" in config:
        os.chdir(config["root_dir"])

    model = hydra.utils.instantiate(config["model"])
    train_dataloader, valid_dataloader, test_dataloader = init_dataloaders(config)
    train_metrics, valid_metrics, test_metrics = init_metrics(config)

    optimizer = init_optimizer(config, model=model)
    scheduler = init_scheduler(
        config,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        gradient_accumulation_steps=run_config.gradient_accumulation_steps,
    )

    performance_config = resolve_performance_config(config)
    configure_mlflow_system_metrics(
        enabled=bool(
            performance_config.get("enabled")
            and performance_config.get("native_system_metrics", True)
        ),
        sampling_interval_sec=float(
            performance_config.get("sampling_interval_sec", 2.0)
        ),
    )

    checkpoint_dir = f"best_models/{experiment_name}/{run_name}"
    callbacks = build_callbacks(
        config,
        model=model,
        train_dataloader=train_dataloader,
        checkpoint_dir=checkpoint_dir,
        early_stopping=early_stopping,
        train_metrics=train_metrics,
        startup_begin_time=startup_begin_time,
    )

    if config.get("logging", {}).get("enable", True):
        print_train_info(
            env, model, train_dataloader, valid_dataloader, test_dataloader
        )
        print_meta(env, model)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        training_arguments=training_arguments,
        run_config=run_config,
        env=env,
        callbacks=callbacks,
        checkpoint_dir=checkpoint_dir,
        config=config,
    )
    try:
        trainer.train()
    finally:
        env.destroy()


if __name__ == "__main__":
    main()
