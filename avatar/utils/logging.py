import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.profiler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary


def print_meta(accelerator, logging_info, model):
    num_params = sum(p.numel() for p in model.parameters())
    accelerator.print(OmegaConf.to_yaml(logging_info["params_to_log"]))
    accelerator.print(f"Total model parameters: {num_params}")
    # Logging context information


def log_time(accelerator, begin_time: int, metric_name: str, step: int):
    accelerator.wait_for_everyone()
    time_duration = time.time() - begin_time
    time_in_minutes = time_duration / 60
    if accelerator.is_main_process:
        accelerator.log({metric_name + "_min": time_in_minutes}, step=step)


def print_trainable_layers(
    accelerator, model: torch.nn.Module, indent: int = 0, max_depth: int = None
):
    """
    Recursively prints the hierarchical structure of a PyTorch model with trainable status.

    This function traverses through all layers of a neural network model and displays
    them in a tree-like structure, highlighting which layers are trainable (green) or
    frozen (gray). It's particularly useful for debugging model architectures and
    verifying which parameters will be updated during training.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze. Can be any nn.Module
            instance including complete models or individual layers.
        indent (int, optional): Current indentation level for formatting the output.
            Defaults to 0. Used internally for recursive calls to maintain proper
            tree structure visualization.
        max_depth (int, optional): Maximum depth to traverse in the model hierarchy.
            If None, traverses the entire model structure. Defaults to None.
            Useful for limiting output when dealing with very deep models.

    Returns:
        None: This function prints directly to stdout and doesn't return any value.
    """
    if max_depth is not None and indent // 2 >= max_depth:
        return
    for name, module in model.named_children():
        params = list(module.parameters())
        trainable = any(p.requires_grad for p in params)
        layer_info = f"{' ' * indent}- {name} ({module.__class__.__name__})"
        if trainable:
            layer_info = f"\033[92m{layer_info} [TRAINABLE]\033[0m"
        else:
            layer_info = f"\033[90m{layer_info} [FROZEN]\033[0m"
        accelerator.print(layer_info)
        if isinstance(module, nn.Module) and len(list(module.children())) > 0:
            print_trainable_layers(accelerator, module, indent + 2, max_depth)


def print_train_info(
    accelerator,
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None,
):
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or int(local_rank) == 0:
        accelerator.print("=" * 100)
        accelerator.print("train dataset size: ", len(train_dataloader.dataset))
        accelerator.print("num batches in train dataloader:", len(train_dataloader))
        accelerator.print("=" * 100)
        if valid_dataloader is not None:
            accelerator.print("valid dataset size: ", len(valid_dataloader.dataset))
            accelerator.print(
                "num batches in valid dataloader::", len(valid_dataloader)
            )
        else:
            accelerator.print("valid dataset not available!")
        accelerator.print("=" * 100)
        if test_dataloader is not None:
            accelerator.print("test dataset size: ", len(test_dataloader.dataset))
            accelerator.print("num batches in test dataloader::", len(test_dataloader))
        else:
            accelerator.print("test dataset not available!")
        accelerator.print("=" * 100)
        accelerator.print()
        accelerator.print("num GPUs used:", torch.cuda.device_count())
        accelerator.print()
        accelerator.print(summary(model, depth=3))
        accelerator.print()
    print_trainable_layers(accelerator, model)
