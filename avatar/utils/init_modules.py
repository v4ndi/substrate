import os
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
import torch.profiler
from hydra.utils import instantiate
from omegaconf import DictConfig

from avatar.metrics import BaseMetric


def get_params_group(model, named_params_to_group=None, need_logs=True):
    if named_params_to_group is None:
        return model.parameters()
    assert all(
        sorted(d.keys()) == sorted(["name", "opt_params"])
        for d in named_params_to_group
    ), "named_params_to_group must be a list of tuples with 2 keys: name and opt_params"
    named_params_to_group.append({"name": [""], "opt_params": {}})
    param_groups = [{"params": [], **d["opt_params"]} for d in named_params_to_group]
    for module_name, param in model.named_parameters():
        for i, d in enumerate(named_params_to_group):
            names, opt_params = d["name"], d["opt_params"]
            if any(module_name.find(name) != -1 for name in names):
                param_groups[i]["params"].append(param)
                if not need_logs:
                    break
                print(f"for {module_name} optimizer params: {opt_params}")
                break
    param_groups = list(filter(lambda group: len(group["params"]) > 0, param_groups))
    return param_groups


def init_optimizer(
    config: Union[DictConfig, dict[str, Any]],
    model: nn.Module,
) -> torch.optim.Optimizer:
    """Initializes and configures a PyTorch optimizer based on the provided configuration.

    This function handles:
    - Learning rate scaling for multi-GPU training when requested
    - Clean instantiation of the optimizer with proper parameter groups
    - Special case for scale_lr_multigpu configuration flag

    Args:
        config: Configuration dictionary or DictConfig containing:
            - optimizer: Parameters for optimizer instantiation
                - Must include '_target_' specifying optimizer class
                - May include 'scale_lr_multigpu' boolean flag
        model: Model to optimize. Can be only nn.Module

    Returns:
        Initialized optimizer instance
    """
    instantiate_dict = dict(config["optimizer"])
    if "named_params_to_group" not in instantiate_dict.keys():
        named_params_to_group = None
    else:
        named_params_to_group = instantiate_dict["named_params_to_group"]
        instantiate_dict.pop("named_params_to_group")
    params = get_params_group(model, named_params_to_group)

    if "scale_lr_multigpu" in instantiate_dict:
        if isinstance(instantiate_dict["scale_lr_multigpu"], bool):
            instantiate_dict["scale_lr_multigpu"] = torch.cuda.device_count()
        instantiate_dict["lr"] *= instantiate_dict["scale_lr_multigpu"]
        instantiate_dict.pop("scale_lr_multigpu")
    return instantiate(instantiate_dict)(params)


def init_profiler(mlflow_arguments):
    def trace_handler(profiler):
        save_dir = f"/home/datalab/nfs/profile_traces/{mlflow_arguments['experiment_name']}/{mlflow_arguments['run_name']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        save_file = f"{save_dir}/trace.json"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(save_file)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=3, warmup=1, active=5, repeat=2),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        use_cuda=torch.cuda.is_available(),
    )
    return profiler


def init_scheduler(
    config: Union[DictConfig, dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Initializes and configures a learning rate scheduler based on the provided configuration.

    This function handles:
    - Gradient accumulation step calculation
    - Automatic determination of total training steps when not specified
    - Special cases for infinite/invalid training step values
    - Instantiation of the scheduler with proper parameters

    Args:
        config: Configuration dictionary or DictConfig containing:
            - scheduler: Parameters for scheduler instantiation
            - accelerator: Gradient accumulation settings
            - train: Training duration settings
        optimizer: The optimizer whose learning rate should be scheduled
        train_dataloader: Training dataloader used to calculate total steps

    Returns:
        Initialized learning rate scheduler instance
    """

    scheduler_params_dict = dict(config["scheduler"])
    grad_accumulation_steps = (
        config["accelerator"]["gradient_accumulation_steps"]
        if "gradient_accumulation_steps" in config["accelerator"].keys()
        else 1
    )
    if "num_training_steps" not in scheduler_params_dict.keys():
        scheduler_params_dict["num_training_steps"] = (
            config["train"]["num_epochs"] * len(train_dataloader) * 100
        )  # infty
    elif (
        scheduler_params_dict["num_training_steps"] is None
        or scheduler_params_dict["num_training_steps"] == -1
    ):
        scheduler_params_dict["num_training_steps"] = (
            config["train"]["num_epochs"]
            * len(train_dataloader)
            // grad_accumulation_steps
        )
    scheduler = instantiate(scheduler_params_dict)(optimizer=optimizer)
    return scheduler


def init_dataloaders(config: DictConfig):
    assert (
        "train_dataloader" in config.keys() and config["train_dataloader"] is not None
    )
    train_dataloader = instantiate(config["train_dataloader"])
    valid_dataloader = None
    if "valid_dataloader" in config.keys() and config["valid_dataloader"] is not None:
        valid_dataloader = instantiate(config["valid_dataloader"])
    test_dataloader = None
    if "test_dataloader" in config.keys() and config["test_dataloader"] is not None:
        test_dataloader = instantiate(config["test_dataloader"])
    return train_dataloader, valid_dataloader, test_dataloader


def init_metrics(config: DictConfig):
    if "metrics" not in config.keys() or config["metrics"] is None:
        return None, None, None

    def init_current_metrics(metrics_config: DictConfig, key_metric: str) -> BaseMetric:
        if key_metric in config["metrics"].keys():
            metrics = instantiate(config["metrics"][key_metric])
        else:
            metrics = None
        return metrics

    train_metrics = init_current_metrics(config["metrics"], "train_metrics")
    valid_metrics = init_current_metrics(config["metrics"], "valid_metrics")
    test_metrics = init_current_metrics(config["metrics"], "test_metrics")

    return train_metrics, valid_metrics, test_metrics


def init_early_stopping(train_config: DictConfig):
    early_stopping = None
    if (
        "early_stopping" in train_config.keys()
        and train_config["early_stopping"] is not None
    ):
        early_stopping = instantiate(train_config["early_stopping"])
    if "early_stopping" in train_config.keys():
        train_config.pop("early_stopping")
    return early_stopping


def init_exp_run_name(config: DictConfig):
    experiment_name = config["mlflow"]["experiment_name"]
    run_name = config["mlflow"]["run_name"]
    if run_name != "debug":
        assert not os.path.exists(
            f"best_models/{experiment_name}/{run_name}"
        ), f"directory with run_name: {run_name} and experiment_name: \
            {experiment_name} already exists"
    return experiment_name, run_name


def init_swa_model(
    config: Union[DictConfig, dict[str, Any]],
    model,
) -> torch.optim.swa_utils.AveragedModel:
    if "swa_model" not in config:
        return None, None, None
    swa_params_dict = dict(config["swa_model"])
    if not swa_params_dict["usage"]:
        return None, None, None

    min_num_steps = swa_params_dict["min_num_steps"]
    min_epoch = swa_params_dict["min_epoch"]
    min_num_steps = 128 if min_num_steps is None else min_num_steps
    min_epoch = 0 if min_epoch is None else min_epoch

    params_list = ["_target_", "_partial_", "avg_fn"]
    swa_params_dict = {key: swa_params_dict[key] for key in params_list}
    alpha = swa_params_dict["avg_fn"]
    swa_params_dict["avg_fn"] = (
        lambda avg_param, new_param, num_avg: alpha * avg_param
        + (1 - alpha) * new_param
    )
    swa_model = instantiate(swa_params_dict)(model)
    return swa_model, min_num_steps, min_epoch
