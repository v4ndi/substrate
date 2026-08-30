import os
from typing import Any

import torch
import torch.nn as nn
import torch.profiler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary


def print_meta(env, model: nn.Module, params: dict[str, Any] | None = None) -> None:
    """Print the run's logged parameters and model size on the main process."""
    num_params = sum(p.numel() for p in model.parameters())
    if params is not None:
        env.print(OmegaConf.to_yaml(params))
    env.print(f"Total model parameters: {num_params}")


def print_trainable_layers(
    env, model: torch.nn.Module, indent: int = 0, max_depth: int | None = None
):
    """
    Recursively prints the hierarchical structure of a PyTorch model with trainable status.

    This function traverses through all layers of a neural network model and displays
    them in a tree-like structure, highlighting which layers are trainable (green) or
    frozen (gray). It's particularly useful for debugging model architectures and
    verifying which parameters will be updated during training.

    Args:
        env (DistEnv): Distributed environment; printing happens on the main rank.
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
        env.print(layer_info)
        if isinstance(module, nn.Module) and len(list(module.children())) > 0:
            print_trainable_layers(env, module, indent + 2, max_depth)


def print_train_info(
    env,
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader | None = None,
    test_dataloader: DataLoader | None = None,
):
    """Summarise dataset sizes, device count and the model on the main process."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or int(local_rank) == 0:
        env.print("=" * 100)
        env.print("train dataset size: ", len(train_dataloader.dataset))
        env.print("num batches in train dataloader:", len(train_dataloader))
        env.print("=" * 100)
        if valid_dataloader is not None:
            env.print("valid dataset size: ", len(valid_dataloader.dataset))
            env.print("num batches in valid dataloader::", len(valid_dataloader))
        else:
            env.print("valid dataset not available!")
        env.print("=" * 100)
        if test_dataloader is not None:
            env.print("test dataset size: ", len(test_dataloader.dataset))
            env.print("num batches in test dataloader::", len(test_dataloader))
        else:
            env.print("test dataset not available!")
        env.print("=" * 100)
        env.print()
        env.print("num GPUs used:", torch.cuda.device_count())
        env.print()
        env.print(summary(model, depth=3))
        env.print()
    print_trainable_layers(env, model)
