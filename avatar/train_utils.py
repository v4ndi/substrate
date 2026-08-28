import os
import shutil
from datetime import timedelta
from glob import glob
from typing import Any, Optional, Union

import accelerate
import hydra
import mlflow
import numpy as np
import torch
import torch.profiler
from accelerate import InitProcessGroupKwargs
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from avatar.metrics import BaseMetric


def init_accelerate(
    accelerate_arguments: dict[str, Any], mlflow_arguments: dict[str, Any] = None
) -> accelerate.Accelerator:
    """Initialize and configure the Hugging Face Accelerator with optional MLflow tracking.

    This function sets up the `accelerate.Accelerator` with the provided arguments and
    optionally logs the configuration to MLflow if MLflow arguments are provided.

    Args:
        accelerate_arguments (Dict[str, Any]): A dictionary containing configuration parameters
            for the `accelerate.Accelerator`.

        mlflow_arguments (Dict[str, Any]): A dictionary containing MLflow tracking parameters.
            If provided, the Accelerator config will be logged to MLflow. Common keys include:
            - `experiment_name` (str): Name of the MLflow experiment.
            - `run_name` (str): Name of the MLflow run.
            - Other supported MLflow parameters (e.g., `tracking_uri`).

    Returns:
        accelerate.Accelerator: An initialized Accelerator object configured with given arguments.

    """
    if mlflow_arguments is not None and "tracking_uri" in mlflow_arguments:
        mlflow.set_tracking_uri(mlflow_arguments["tracking_uri"])
        mlflow_arguments = dict(mlflow_arguments)
        mlflow_arguments.pop("tracking_uri")

    if mlflow_arguments is not None:
        tracker = accelerate.tracking.MLflowTracker(**mlflow_arguments)
    else:
        tracker = None
    if "kwargs_handlers" in accelerate_arguments:
        kwargs_handlers = [
            hydra.utils.instantiate(kwarg)
            for kwarg in accelerate_arguments["kwargs_handlers"]
        ]
        del accelerate_arguments.kwargs_handlers
    else:
        kwargs_handlers = []
    kwargs_handlers.append(InitProcessGroupKwargs(timeout=timedelta(seconds=36000000)))
    accelerator = instantiate(accelerate_arguments)(
        log_with=tracker, kwargs_handlers=kwargs_handlers
    )
    if mlflow_arguments is not None:
        accelerator.init_trackers(mlflow_arguments["run_name"])

    return accelerator


class EarlyStopping:
    """Implements early stopping mechanism for training processes.

    Monitors a specified metric and stops training when the metric hasn't improved
    for a given number of consecutive evaluations (patience). Supports both
    maximization and minimization objectives.

    Attributes:
        main_metric (str): Name of the metric to monitor for early stopping.
        patience (int): Number of evaluations to wait before stopping when no improvement.
        delta (float): Minimum change in monitored metric to qualify as improvement.
        counter (int): Current count of evaluations without improvement.
        best_score (float): Best score observed so far.
        early_stop (bool): Flag indicating whether to stop training.
        strategy (str): Optimization strategy - "max" to maximize or "min" to minimize.

    Example:
        >>> early_stopping = EarlyStopping(main_metric="val_loss", patience=5, strategy="min")
        >>> for epoch in range(100):
        ...     # Training happens here
        ...     val_scores = {"val_loss": 0.2, "val_acc": 0.95}
        ...     early_stopping(val_scores)
        ...     if early_stopping.early_stop:
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(
        self,
        main_metric: str,
        patience: int = 10,
        delta: int = 0,
        strategy: str = "max",
    ):
        """Initializes the EarlyStopping instance.

        Args:
            main_metric (str): Name of the metric to monitor.
            patience (int, optional): Number of evaluations to wait before stopping when
                no improvement occurs. Defaults to 10.
            delta (float, optional): Minimum change in monitored metric to qualify as
                improvement. Defaults to 0.
            strategy (str, optional): Optimization strategy - "max" to maximize or
                "min" to minimize the metric. Defaults to "max".

        Raises:
            ValueError: If strategy is neither "max" nor "min".
        """

        assert strategy in ["min", "max"], (
            f"Unsupported value for strategy: {strategy=}"
        )
        self.main_metric = main_metric
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.strategy = strategy

    def __call__(self, scores):
        """Evaluates whether to stop training based on current metric scores.

        Args:
            scores (dict): Dictionary containing metric values from current evaluation.

        Returns:
            None: Updates internal state (counter, best_score, early_stop).

        Raises:
            AssertionError: If main_metric is not found in scores dictionary.
        """
        assert self.main_metric in scores, (
            f"Not found metric: {self.main_metric} in scores"
        )
        score = scores[self.main_metric]
        if self.strategy == "min":
            score *= -1
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def save_checkpoint(
    accelerator: accelerate.Accelerator,
    path: str,
    num_step: int,
    max_checkpoints: int = None,
):
    """Saves a training checkpoint using the Accelerator and manages checkpoint rotation.

    The function saves the complete training state (including model, optimizer, and scheduler)
    to a specified path using the Hugging Face Accelerator. Optionally maintains a maximum
    number of checkpoints by deleting the oldest ones when the limit is exceeded.

    Args:
        accelerator (Accelerator): The Hugging Face Accelerator instance containing the training state.
        path (str): Base directory path where checkpoints will be saved.
        num_step (int): Training step number, used to create a versioned subdirectory
            (e.g., path/num_step/checkpoints/).
        max_checkpoints (int, optional): Maximum number of checkpoints to retain. If None,
            keeps all checkpoints. When specified, oldest checkpoints beyond this number
            will be deleted based on modification time.

    Returns:
        None

    Example:
        >>> from accelerate import Accelerator
        >>> accelerator = Accelerator()
        >>> save_checkpoint(accelerator, 'training_checkpoints', 5000, max_checkpoints=3)
        # Saves full state to 'training_checkpoints/5000/checkpoints/'
        # Keeps only the 3 most recent checkpoints
    """
    output_dir = os.path.join(f"{path}", str(num_step))
    output_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    accelerator.save_state(output_dir, safe_serialization=False)

    if max_checkpoints is not None:
        checkpoints = glob(f"{path}/*")
        checkpoints.sort(key=os.path.getmtime)

        for checkpoint in checkpoints[:-max_checkpoints]:
            shutil.rmtree(checkpoint)


def save_model(
    model: torch.nn.Module,
    path: str,
    num_step: int,
    max_checkpoints: int = None,
):
    """Saves a PyTorch model to disk and manages checkpoints.

    The function saves the model's state dictionary to a specified path, creating the directory
    if it doesn't exist. Optionally maintains a maximum number of checkpoints by deleting
    the oldest ones when the limit is exceeded.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        path (str): Base directory path where the model will be saved.
        num_step (int): Training step number, used to create a subdirectory (e.g., path/num_step/).
        max_checkpoints (int, optional): Maximum number of checkpoints to retain. If None, keeps all.
            When specified, oldest checkpoints beyond this number will be deleted.

    Returns:
        None

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> save_model(model, 'checkpoints', 1000, max_checkpoints=5)
        # Saves model to 'checkpoints/1000/model.bin' and keeps only 5 most recent checkpoints
    """
    output_dir = os.path.join(f"{path}", str(num_step))
    output_path = os.path.join(output_dir, "model.bin")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(model, torch.optim.swa_utils.AveragedModel):
        model = model.module

    torch.save(model.state_dict(), output_path)

    if max_checkpoints is not None:
        checkpoints = glob(f"{path}/*")
        checkpoints.sort(key=os.path.getmtime)

        for checkpoint in checkpoints[:-max_checkpoints]:
            shutil.rmtree(checkpoint)


def flatten_dict(
    params_dict: Union[dict[str, Any], DictConfig],
    parent_key: str = "",
    sep: str = "_",
    ignore_keys: Optional[Union[set[str], list]] = None,
    preserve_keys: Optional[Union[set[str], list]] = None,
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary or DictConfig into a single-level dictionary.

    This function handles both regular dictionaries and OmegaConf DictConfig objects,
    creating keys by concatenating nested keys with the specified separator.

    Args:
        params_dict (Union[dict[str, Any], DictConfig]):
            Input dictionary or DictConfig to flatten. Can be arbitrarily nested.
        parent_key (str): Base key string used for recursion (leave empty for top-level call).
        sep (str): Separator to use between concatenated keys. Defaults to '_'.
        ignore_keys (Optional[Union[set[str], list]]):
            Keys to exclude from the output. Can be a set or list.
            Nested keys should be specified in their flattened form.
        preserve_keys (Optional[Union[set[str], list]]):
            Container keys to retain as one value instead of recursively flattening.

    Returns:
        A flattened dictionary where:
        - Keys are paths through the original nested structure joined by `sep`
        - Values are the leaf nodes from the original structure

    Raises:
        TypeError: If input is not a dictionary or DictConfig.
        ValueError: If separator appears in original keys (would cause ambiguity).
    """
    if ignore_keys is None:
        ignore_keys = set()
    else:
        ignore_keys = set(ignore_keys)
    if preserve_keys is None:
        preserve_keys = set()
    else:
        preserve_keys = set(preserve_keys)
    items = {}
    for k, v in params_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if k in ignore_keys:
            continue

        if k in preserve_keys:
            items[new_key] = v
        elif isinstance(v, DictConfig):
            items.update(
                flatten_dict(
                    v,
                    new_key,
                    sep=sep,
                    ignore_keys=ignore_keys,
                    preserve_keys=preserve_keys,
                )
            )
        else:
            items[new_key] = v

    return items


IGNORE_CONFIG_KEYS = ["tokens_meta", "_target_"]
PRESERVE_CONFIG_KEYS = ["filters"]


def normalize_log_param(value: Any) -> Any:
    """Convert resolved config values to logging-safe primitive containers.

    Hydra resolvers may return NumPy values such as ``np.datetime64``. Those
    values are valid filter bounds, but OmegaConf and MLflow cannot serialize
    them as run parameters without normalization.
    """
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return str(value)
    if isinstance(value, np.ndarray):
        return [normalize_log_param(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return normalize_log_param(value.item())
    if isinstance(value, dict):
        return {str(key): normalize_log_param(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_log_param(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((normalize_log_param(item) for item in value), key=repr)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def get_default_log_params(config: DictConfig):
    return normalize_log_param(
        flatten_dict(
            {
                "model": config["model"],
                "accelerator": config["accelerator"],
                "optimizer": config["optimizer"],
                "scheduler": config["scheduler"]
                if "scheduler" in config.keys()
                else None,
                "swa_model": config["swa_model"]
                if "swa_model" in config.keys()
                else None,
                "train": config["train"],
                "train_dataloader": config["train_dataloader"],
                "valid_dataloader": config["valid_dataloader"]
                if "valid_dataloader" in config.keys()
                else None,
            },
            ignore_keys=IGNORE_CONFIG_KEYS,
            preserve_keys=PRESERVE_CONFIG_KEYS,
        )
    )


def apply_basic_loader_checks(train_dataloader, valid_dataloader, test_dataloader):
    if (
        hasattr(train_dataloader.dataset, "lazy_process")
        and train_dataloader.dataset.lazy_process
    ):
        assert train_dataloader.drop_last, (
            "If you want to use lazy_process=True, set dataloader.drop_last=True"
        )
    if (
        valid_dataloader is not None
        and hasattr(valid_dataloader.dataset, "lazy_process")
        and valid_dataloader.dataset.lazy_process
    ):
        assert valid_dataloader.drop_last, (
            "If you want to use lazy_process=True, set dataloader.drop_last=True"
        )
    if (
        test_dataloader is not None
        and hasattr(test_dataloader.dataset, "lazy_process")
        and test_dataloader.dataset.lazy_process
    ):
        assert test_dataloader.drop_last, (
            "If you want to use lazy_process=True, set dataloader.drop_last=True"
        )


def wrap_metrics(metrics: list[BaseMetric]):
    if OmegaConf.is_list(metrics):
        metrics = OmegaConf.to_container(metrics)
    elif not OmegaConf.is_list(metrics) and metrics is not None:
        metrics = [metrics]
    return metrics


def calculate_output_loss(output, accelerator, distributed: bool = True):
    """
    Calculates the weighted average loss from an output object based on the number of items in each metric.

    Args:
        output (object): An object containing loss values and optionally a num_items attribute.
        accelerator (Accelerator): An instance of the Accelerator class for distributed computing operations.
        distributed (bool, optional): Whether to use distributed computing. Defaults to True.

    Returns:
        float or dict: If output has 'num_items', returns a weighted average loss as a float.
        Otherwise, returns a dictionary of gathered loss values by field name.

    Logic:
        the loss in each process is collected using the following formula:
        loss_proc = output_loss_i * num_proc / (num_item_0 + num_items_1 + ... + num_items_num_proc),
        where num_proc is the process synchronization constant will decriease, because mean all reduce occurs.
    """
    if not hasattr(output, "num_items") or output.num_items is None:
        return output.loss
    local_num_items_in_batch = output.num_items
    if distributed:
        num_items_in_batch = accelerator.gather_for_metrics(local_num_items_in_batch)
        num_proc = accelerator.num_processes
    else:
        num_items_in_batch = local_num_items_in_batch
        num_proc = 1

    num_items_in_batch = {
        key: val.sum().item() for key, val in num_items_in_batch.items()
    }

    loss = 0.0
    for key, value in num_items_in_batch.items():
        if value != 0:
            loss += output.losses[key] * num_proc / (value)
    return loss / len(num_items_in_batch)


def set_root_dir(path: Union[str, bytes]) -> None:
    """Change the current working directory to the specified path.

    This function wraps `os.chdir()` to set the root directory for the script.
    Useful for ensuring consistent file paths across different execution environments.

    Args:
        path (Union[str, bytes]): Path to the new working directory.
            Can be a string or bytes.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
        PermissionError: If the user lacks permissions to access the directory.
    """
    os.chdir(path)


def log_metrics(
    accelerator: accelerate.Accelerator,
    metrics: dict[str, Union[float, int]],
    num_step: int,
    prefix: Optional[str] = None,
) -> None:
    """Log training/evaluation metrics through the Accelerator's logging interface.

    This function handles metric logging in a distributed training environment,
    ensuring compatibility with various logging backends (e.g., TensorBoard, WandB).

    Args:
        accelerator (accelerate.Accelerator): The Accelerator object used for distributed training.
        metrics (Dict[str, Union[float, int]]): Dictionary of metrics to log.
            Keys are metric names (str), values should be numeric (float/int).
        num_step (int): Current training step number (used for x-axis in logging).
        prefix (Optional[str]): Optional prefix to add to all metric names.
            Useful for separating train/validation metrics (e.g., "train/", "val/").
    """

    for key, val in metrics.items():
        if prefix is not None:
            metrics_name = prefix + key
        else:
            metrics_name = key
        if isinstance(val, torch.Tensor):
            val = val.item()
        accelerator.log({metrics_name: val}, step=num_step)


def move_to_device(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any]],
    device: Union[str, torch.device],
) -> Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any]]:
    """Recursively move data and all nested contents to the specified device.

    This function handles multiple data types including:
    - PyTorch tensors
    - Dictionaries containing tensors
    - Lists and tuples of tensors
    - Numpy arrays (converted to torch tensors)
    - Any combination of the above nested structures

    Args:
        data: Input data to move. Can be:
            - A single torch.Tensor
            - A dictionary with values to move
            - A list/tuple of items to move
        device: Target device. Can be either:
            - String (e.g., 'cpu', 'cuda', 'cuda:0')
            - torch.device object

    Returns:
        The same data structure with all tensors moved to the target device.

    Raises:
        TypeError: If input data type is not supported.
    """
    if isinstance(data, torch.Tensor) or hasattr(data, "to"):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def update_ema_weights(
    model: torch.nn.Module,
    swa_model: torch.optim.swa_utils.AveragedModel,
    epoch: int,
    min_epoch: int,
    num_steps: int,
    min_num_steps: int,
    accelerator: accelerate.Accelerator,
) -> torch.optim.swa_utils.AveragedModel | torch.nn.Module:
    """Decides whether to use the base model or the EMA model.
    Args:
        model (torch.nn.Module): Base model
        swa_model (torch.optim.swa_utils.AveragedModel): Averaged model(Base model)
        epoch (int): current epoch
        min_epoch (int): num of epochs to start EMA
        num_steps (int): current step
        min_num_steps (int): step to update EMA
        accelerator (accelerate.Accelerator)

    Returns:
        torch.optim.swa_utils.AveragedModel | torch.nn.Module
    """
    if not swa_model or epoch < min_epoch:
        return model

    if num_steps % ((min_num_steps + 1) // accelerator.num_processes) == 0:
        with torch.inference_mode():
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()
            swa_model.update_parameters(unwrapped_model)

        model.train()

    return swa_model
