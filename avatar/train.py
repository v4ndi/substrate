import datetime
import os
import time
from typing import Any, Optional, Union

import accelerate
import hydra
import torch
from accelerate.utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from avatar.accelerate_utils import init_accelerate
from avatar.metrics import BaseMetric
from avatar.train_utils import (
    EarlyStopping,
    apply_basic_loader_checks,
    calculate_output_loss,
    get_default_log_params,
    log_metrics,
    move_to_device,
    save_checkpoint,
    save_model,
    update_ema_weights,
    wrap_metrics,
)
from avatar.training_arguments import TrainingArguments
from avatar.utils.init_modules import (
    init_dataloaders,
    init_early_stopping,
    init_exp_run_name,
    init_metrics,
    init_optimizer,
    init_profiler,
    init_scheduler,
    init_swa_model,
)
from avatar.utils.logging import log_time, print_meta, print_train_info
from avatar.utils.performance_metrics import (
    SystemMetricsCollector,
    configure_mlflow_system_metrics,
    infer_batch_size,
    reduce_epoch_performance,
    reduce_max_duration,
    reduce_system_snapshot,
)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT"] = "100_000_000"
os.environ["NCCL_TIMEOUT"] = "36_000_000"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
    f"/home/datalab/nfs/torchinductor_cache/cache_pid{os.getpid()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)


@hydra.main(version_base=None, config_path=".", config_name="config-td")
def main(config: DictConfig) -> None:
    """Main training pipeline execution function.

    Orchestrates the complete training workflow including:
    - Configuration processing
    - Model initialization
    - Data loading
    - Training loop
    - Evaluation
    - Logging and checkpointing

    Args:
        config: OmegaConf configuration object containing all parameters.
            Expected sections include:
            - model: Model architecture parameters
            - data: Dataset and data loading parameters
            - training: Optimization and training parameters
            - logging: MLflow/W&B logging parameters
            - evaluation: Validation/testing parameters

    Raises:
        ValueError: If required configuration sections are missing
        RuntimeError: If training fails or resources unavailable
    """
    startup_begin_time = time.perf_counter()
    train_config = OmegaConf.to_container(config["train"])
    early_stopping = init_early_stopping(train_config)
    training_arguments = TrainingArguments(**train_config)

    set_seed(training_arguments.seed, device_specific=False)

    experiment_name, run_name = init_exp_run_name(config)

    if "root_dir" in config:
        os.chdir(config["root_dir"])

    model = instantiate(config["model"])

    train_dataloader, valid_dataloader, test_dataloader = init_dataloaders(config)

    train_metrics, valid_metrics, test_metrics = init_metrics(config)

    checkpoint_dir = f"best_models/{experiment_name}/{run_name}"
    mlflow_arguments = config["mlflow"]
    accelerate_arguments = config["accelerator"]

    optimizer = init_optimizer(config, model=model)
    scheduler = init_scheduler(
        config, optimizer=optimizer, train_dataloader=train_dataloader
    )

    performance_config = {
        "enabled": False,
        "sampling_interval_sec": 2.0,
        "native_system_metrics": True,
    }
    if "logging" in config and "performance_metrics" in config["logging"]:
        configured_performance_metrics = OmegaConf.to_container(
            config["logging"]["performance_metrics"]
        )
        if configured_performance_metrics is not None:
            performance_config.update(configured_performance_metrics)

    logging_info = {
        "enable": config["logging"]["enable"]
        if "logging" in config and "enable" in config["logging"]
        else True,
        "enable_profiler": config["logging"]["enable_profiler"]
        if "logging" in config and "enable_profiler" in config["logging"]
        else False,
        "performance_metrics": performance_config,
        "params_to_log": get_default_log_params(config),
    }
    if performance_config["enabled"] and performance_config["native_system_metrics"]:
        logging_info["params_to_log"]["system_metrics_scope"] = "node_0"
    if performance_config["enabled"]:
        train_dataset = train_dataloader.dataset
        available_cpu_cores = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        logging_info["params_to_log"].update({
            "performance_metrics_enabled": True,
            "performance_metrics_sampling_interval_sec": performance_config[
                "sampling_interval_sec"
            ],
            "shard_by_rank": bool(getattr(train_dataset, "shard_by_rank", False)),
            "drop_tail": getattr(train_dataset, "drop_tail", None),
            "rotate_tail": getattr(train_dataset, "rotate_tail", None),
            "filter_cache": getattr(train_dataset, "filter_cache", None),
            "filter_cache_dir": getattr(train_dataset, "filter_cache_dir", None),
            "scan_workers_per_rank": getattr(train_dataset, "scan_workers", None),
            "scan_ranks": getattr(train_dataset, "scan_ranks", None),
            "effective_scan_ranks": getattr(
                train_dataset, "_effective_scan_ranks", None
            ),
            "batch_size_per_rank": train_dataloader.batch_size,
            "num_workers_per_rank": train_dataloader.num_workers,
            "parquet_file_count": len(getattr(train_dataset, "files", [])),
            "dataset_size_bytes": getattr(train_dataset, "total_parquet_bytes", None),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "available_cpu_cores": available_cpu_cores,
        })

    train(
        accelerate_arguments=accelerate_arguments,
        mlflow_arguments=mlflow_arguments,
        training_arguments=training_arguments,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        train_metrics=train_metrics,
        checkpoint_dir=checkpoint_dir,
        early_stopping=early_stopping,
        logging_info=logging_info,
        startup_begin_time=startup_begin_time,
    )


def train(
    accelerate_arguments: Union[dict[str, Any], DictConfig],
    mlflow_arguments: Union[dict[str, Any], DictConfig],
    training_arguments: TrainingArguments,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    config,
    checkpoint_dir: str,
    train_dataloader: DataLoader,
    train_metrics: Union[BaseMetric, list[BaseMetric]] = None,
    valid_metrics: Union[BaseMetric, list[BaseMetric]] = None,
    test_metrics: Union[BaseMetric, list[BaseMetric]] = None,
    valid_dataloader: Optional[DataLoader] = None,
    test_dataloader: Optional[DataLoader] = None,
    early_stopping: Optional[EarlyStopping] = None,
    logging_info: Optional[dict[str, Any]] = None,
    startup_begin_time: Optional[float] = None,
) -> dict[str, Any]:
    """Execute the complete training pipeline with logging, checkpointing and evaluation.

    Args:
        accelerate_arguments: Configuration for Accelerator initialization.
            Can be dict, DictConfig or Namespace.
        mlflow_arguments: MLflow tracking configuration.
            Can be dict, DictConfig or Namespace.
        training_arguments: Training configuration from HuggingFace transformers.
        model: The model to train (must be torch.nn.Module compatible).
        optimizer: The optimizer to use for training.
        scheduler: Learning rate scheduler.
        checkpoint_dir: Directory to save model checkpoints.
        train_metrics: Optional metric calculator for training.
        valid_metrics: Optional metric calculator for validation.
        test_metrics: Optional metric calculator for testing.
        train_dataloader: DataLoader for training data.
        valid_dataloader: DataLoader for validation data.
        test_dataloader: DataLoader for test data.
        early_stopping: Optional early stopping controller.
        params_to_log: Additional parameters to log to MLflow.

    """
    apply_basic_loader_checks(train_dataloader, valid_dataloader, test_dataloader)

    set_seed(training_arguments.seed, device_specific=False)

    train_metrics = wrap_metrics(train_metrics)
    performance_config = logging_info.get("performance_metrics", {})
    performance_enabled = bool(
        logging_info.get("enable", True) and performance_config.get("enabled", False)
    )
    sampling_interval_sec = float(performance_config.get("sampling_interval_sec", 2.0))
    configure_mlflow_system_metrics(
        enabled=(
            performance_enabled
            and performance_config.get("native_system_metrics", True)
        ),
        sampling_interval_sec=sampling_interval_sec,
    )
    accelerator = init_accelerate(
        accelerate_arguments=accelerate_arguments, mlflow_arguments=mlflow_arguments
    )
    train_dataset = train_dataloader.dataset
    shard_metrics_enabled = bool(
        performance_enabled and getattr(train_dataset, "shard_by_rank", False)
    )
    if shard_metrics_enabled and hasattr(train_dataset, "configure_epoch_metrics"):
        train_dataset.configure_epoch_metrics(train_dataloader.num_workers)
    system_collector = (
        SystemMetricsCollector(
            is_local_main_process=accelerator.is_local_main_process,
            device=accelerator.device,
            sampling_interval_sec=sampling_interval_sec,
        )
        if performance_enabled
        else None
    )
    if logging_info["enable"]:
        print_train_info(
            accelerator, model, train_dataloader, valid_dataloader, test_dataloader
        )

    if accelerator.is_main_process:
        accelerator.trackers[0].store_init_configuration(logging_info["params_to_log"])

    if training_arguments.steps_before_evaluation is not None:
        training_arguments.steps_before_evaluation = (
            training_arguments.steps_before_evaluation // accelerator.num_processes
        )

    if training_arguments.model_state is not None:
        print("LOADED MODEL WEIGHTS")
        state = torch.load(training_arguments.model_state)
        model.load_state_dict(state)

    if accelerator.is_main_process and logging_info["enable_profiler"]:
        profiler = init_profiler(mlflow_arguments)
        profiler.start()

    swa_model, min_num_steps, min_epoch = init_swa_model(config, model)

    if getattr(train_dataloader.dataset, "shard_by_rank", False):
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader
        )

    if training_arguments.checkpoint_state is not None:
        accelerator.load_state(training_arguments.checkpoint_state)
    accelerator.wait_for_everyone()
    num_total_steps = 0

    if logging_info["enable"]:
        print_meta(accelerator, logging_info, model)

    # log to mlflow
    num_params = sum(p.numel() for p in model.parameters())
    accelerator.log({"num_processes": accelerator.num_processes}, step=0)
    accelerator.log({"total_number_model_parameters": num_params}, step=0)
    accelerator.log(
        {"gradient_accumulation_steps": accelerator.gradient_accumulation_steps}, step=0
    )

    if shard_metrics_enabled:
        scan_sec = reduce_max_duration(
            float(getattr(train_dataset, "_scan_duration_sec", 0.0)),
            accelerator.device,
        )
        total_raw_rows = int(getattr(train_dataset, "total_length", 0))
        file_counts = getattr(train_dataset, "_file_counts", None)
        total_valid_rows = int(file_counts.sum()) if file_counts is not None else 0
        dropped_tail_rows = (
            total_valid_rows % accelerator.num_processes
            if getattr(train_dataset, "drop_tail", True)
            else 0
        )
        if accelerator.is_main_process:
            initial_shard_metrics = {
                "shard/scan_sec": scan_sec,
                "shard/scan_rows_per_sec": (
                    total_raw_rows / scan_sec if scan_sec else 0.0
                ),
                "shard/total_raw_rows": total_raw_rows,
                "shard/total_valid_rows": total_valid_rows,
                "shard/dropped_tail_rows": dropped_tail_rows,
                "shard/parquet_file_count": len(train_dataset.files),
            }
            if getattr(train_dataset, "_use_filter_cache", False):
                initial_shard_metrics["shard/filter_cache_hit"] = float(
                    getattr(train_dataset, "_filter_cache_hit", False)
                )
                initial_shard_metrics["shard/scan_rank_count"] = float(
                    getattr(train_dataset, "_effective_scan_ranks", 0)
                )
            accelerator.log(initial_shard_metrics, step=0)

    start_epoch = (
        training_arguments.start_epoch
        if training_arguments.start_epoch is not None
        else 0
    )

    num_epochs = training_arguments.num_epochs
    previous_training_end = None
    if performance_enabled:
        accelerator.wait_for_everyone()
        startup_local = (
            time.perf_counter() - startup_begin_time
            if startup_begin_time is not None
            else 0.0
        )
        startup_max = reduce_max_duration(startup_local, accelerator.device)
        if accelerator.is_main_process:
            accelerator.log(
                {"startup/before_first_epoch_sec": startup_max},
                step=start_epoch,
            )

    for epoch in range(start_epoch, num_epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if performance_enabled:
            accelerator.wait_for_everyone()
            if previous_training_end is not None:
                between_epochs_local = time.perf_counter() - previous_training_end
                between_epochs_max = reduce_max_duration(
                    between_epochs_local, accelerator.device
                )
                if accelerator.is_main_process:
                    accelerator.log(
                        {"train_epoch/between_epochs_sec": between_epochs_max},
                        step=epoch - 1,
                    )
            begin_epoch_time = time.perf_counter()
            system_collector.reset_epoch()
            if shard_metrics_enabled:
                train_dataset.reset_epoch_metrics()
        else:
            begin_epoch_time = time.time()
        num_steps = 0
        local_samples = 0
        first_batch_sec = None
        epoch_lr = 0.0
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Training step; epoch={epoch + 1}/{num_epochs}",
            disable=(not accelerator.is_local_main_process),
        )
        end_training = early_stopping is not None and early_stopping.early_stop

        early_stop_tensor = torch.tensor(int(end_training), device=accelerator.device)
        early_stop_tensor = accelerator.gather(early_stop_tensor).max()

        if early_stop_tensor.item():
            break

        losses = []
        grad_norms = []

        for _, batch in enumerate(progress_bar):
            if performance_enabled:
                if first_batch_sec is None:
                    first_batch_sec = time.perf_counter() - begin_epoch_time
                system_collector.maybe_sample()
                batch_samples = infer_batch_size(batch)
                if batch_samples is None:
                    batch_samples = train_dataloader.batch_size or 0
                local_samples += batch_samples
            num_steps += 1
            num_total_steps += 1

            batch = move_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                if "mixed_precision" in accelerate_arguments:
                    with accelerator.autocast():
                        output = model(**batch)
                else:
                    output = model(**batch)

                loss = calculate_output_loss(output, accelerator)
                cur_lr = scheduler.get_lr()[0]
                epoch_lr += cur_lr

                detached_loss = loss.detach() if loss.requires_grad else loss
                losses.append(detached_loss)

                progress_bar.set_description(
                    f"Training; epoch={epoch + 1}/{num_epochs}, ",
                    f"loss={detached_loss.item():.4f}, lr={cur_lr:.4f}",
                )

                accelerator.backward(loss) if loss.requires_grad else None
                if (
                    training_arguments.clip_grad_norm is not None
                    and accelerator.sync_gradients
                ):
                    max_grad_norm = training_arguments.clip_grad_norm
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    grad_norms.append(grad_norm)
                if train_metrics is not None:
                    gathered_objects = accelerator.gather_for_metrics((batch, output))
                    if accelerator.is_main_process:
                        len_gather_obj = len(gathered_objects)
                        assert len_gather_obj % 2 == 0, "incorrect lenght"
                        for metric in train_metrics:
                            for i in range(0, len_gather_obj, 2):
                                metric.update(
                                    inputs=gathered_objects[i],
                                    outputs=gathered_objects[i + 1],
                                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and logging_info["enable_profiler"]:
                    profiler.step()

            model_type = update_ema_weights(
                model=model,
                swa_model=swa_model,
                epoch=epoch,
                min_epoch=min_epoch,
                num_steps=num_steps,
                min_num_steps=min_num_steps,
                accelerator=accelerator,
            )

            # per-step validation
            if (
                valid_dataloader is not None
                and training_arguments.steps_before_evaluation is not None
                and num_total_steps % training_arguments.steps_before_evaluation == 0
            ):
                begin_eval_time = time.time()
                apply_evaluation(
                    accelerator=accelerator,
                    model=model_type,
                    training_arguments=training_arguments,
                    valid_metrics=valid_metrics,
                    valid_dataloader=valid_dataloader,
                    early_stopping=early_stopping,
                    num_steps=num_total_steps,
                    checkpoint_dir=checkpoint_dir,
                )
                log_time(
                    accelerator,
                    begin_time=begin_eval_time,
                    metric_name="eval_epoch_time_duration",
                    step=num_total_steps,
                )
                model.train()

        if performance_enabled:
            epoch_wall_sec = time.perf_counter() - begin_epoch_time
            previous_training_end = time.perf_counter()
            shard_stats = (
                train_dataset.get_epoch_metrics() if shard_metrics_enabled else None
            )
            epoch_performance = reduce_epoch_performance(
                device=accelerator.device,
                epoch_wall_sec=epoch_wall_sec,
                local_samples=local_samples,
                local_batches=num_steps,
                shard_stats=shard_stats,
                total_raw_rows=int(getattr(train_dataset, "total_length", 0)),
                total_parquet_bytes=int(
                    getattr(train_dataset, "total_parquet_bytes", 0)
                ),
            )
            system_performance = reduce_system_snapshot(
                system_collector.snapshot(), accelerator.device
            )
            if epoch == start_epoch:
                first_batch_max = reduce_max_duration(
                    first_batch_sec or 0.0, accelerator.device
                )
                epoch_performance["startup/first_batch_sec"] = first_batch_max
            if accelerator.is_main_process:
                accelerator.log({**epoch_performance, **system_performance}, step=epoch)
        else:
            log_time(
                accelerator,
                begin_time=begin_epoch_time,
                metric_name="train_epoch_time_duration",
                step=epoch,
            )

        # per-epoch validation
        if (
            valid_dataloader is not None
            and training_arguments.steps_before_evaluation is None
        ):
            begin_eval_time = time.time()
            apply_evaluation(
                accelerator=accelerator,
                model=model_type,
                training_arguments=training_arguments,
                valid_metrics=valid_metrics,
                valid_dataloader=valid_dataloader,
                early_stopping=early_stopping,
                num_steps=epoch,
                checkpoint_dir=checkpoint_dir,
            )
            log_time(
                accelerator,
                begin_time=begin_eval_time,
                metric_name="eval_epoch_time_duration",
                step=num_total_steps,
            )
        # Logging
        epoch_lr /= num_steps
        accelerator.log({"mean_epoch_lr": epoch_lr}, step=epoch)

        losses = accelerator.gather(sum(losses)).sum().item() / (
            accelerator.num_processes * num_steps
        )
        accelerator.log({"train_loss": losses}, step=epoch)
        if len(grad_norms) > 0:
            grad_norms = accelerator.gather(sum(grad_norms)).sum().item() / (
                accelerator.num_processes * num_steps
            )
            accelerator.log({"grad_norm": grad_norms}, step=epoch)

        if accelerator.is_main_process:
            scores = {}
            if train_metrics is not None:
                for metric in train_metrics:
                    scores.update(metric.compute())
                    metric.reset()
                log_metrics(accelerator, scores, epoch, prefix="train_")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process and logging_info["enable_profiler"]:
        profiler.stop()

    if test_dataloader is not None:
        train_root_dir = os.getcwd()

        if isinstance(model_type, torch.optim.swa_utils.AveragedModel):
            unwrapped_model = model_type.module
        else:
            unwrapped_model = accelerator.unwrap_model(model_type)
        path_to_model_dir = f"{train_root_dir}/{checkpoint_dir}"

        best_epoch = sorted([int(direct) for direct in os.listdir(path_to_model_dir)])[
            -1
        ]
        path_to_model = (
            f"{path_to_model_dir}/{best_epoch}/checkpoints/pytorch_model.bin"
        )
        state = torch.load(path_to_model)
        with torch.inference_mode():
            unwrapped_model.load_state_dict(state)

        scores = evaluation(
            model=unwrapped_model,
            device=accelerator.device,
            valid_metrics=test_metrics,
            valid_dataloader=test_dataloader,
            accelerator=accelerator,
        )
        if accelerator.is_main_process:
            if logging_info["enable"]:
                accelerator.print(scores)
            log_metrics(accelerator, scores, epoch, prefix="test_")
    accelerator.end_training()


def apply_evaluation(
    accelerator: accelerate.Accelerator,
    model: torch.nn.Module,
    training_arguments: TrainingArguments,
    valid_metrics: Union[BaseMetric, list[BaseMetric]],
    valid_dataloader: DataLoader,
    early_stopping,
    num_steps: int,
    checkpoint_dir: str,
):
    if isinstance(model, torch.optim.swa_utils.AveragedModel):
        unwrapped_model = model
    else:
        unwrapped_model = accelerator.unwrap_model(model)

    scores = evaluation(
        model=unwrapped_model,
        device=accelerator.device,
        valid_metrics=valid_metrics,
        valid_dataloader=valid_dataloader,
        accelerator=accelerator,
        distributed_evaluate=training_arguments.distributed_evaluate,
    )
    if not accelerator.is_main_process:
        return
    log_metrics(accelerator, scores, num_steps, prefix="valid_")
    if early_stopping is not None:
        early_stopping(scores)
    if early_stopping is None or early_stopping.counter == 0:
        max_checkpoints = training_arguments.max_saved_checkpoints
        save_checkpoint(
            accelerator=accelerator,
            path=checkpoint_dir,
            num_step=num_steps,
            max_checkpoints=max_checkpoints,
        )
        save_model(
            model=unwrapped_model,
            path=checkpoint_dir,
            num_step=num_steps,
        )


@torch.inference_mode()
def evaluation(
    model: Module,
    device: Union[str, torch.device],
    valid_dataloader: DataLoader,
    valid_metrics: Union[BaseMetric, list[BaseMetric]] = None,
    accelerator=None,
    distributed_evaluate: bool = False,
) -> dict[str, Any]:
    """Evaluate the model on validation data and compute metrics.

    Args:
        model: The trained model to evaluate (must be a PyTorch Module).
        device: The device to run evaluation on ('cpu', 'cuda', or torch.device object).
        valid_dataloader: DataLoader containing the validation dataset.
        valid_metrics: Optional metric calculator. If None, only loss will be computed.
        distributed_evaluate: Whether to evaluate the model in distributed mode.
        Supports only with valid_dataloader.drop_last = True flag.

    Returns:
        Dictionary containing:
        - 'loss': Average loss across validation set
        - 'metrics': Dictionary of computed metrics (if valid_metrics provided)
        - 'samples': Number of samples processed

    Raises:
        RuntimeError: If model is not in eval mode or metrics computation fails.
        ValueError: If DataLoader is empty or device is invalid.
        assert error when use distributed_evaluate and valid_dataloader.drop_last = False
    """
    model.eval()
    if distributed_evaluate:
        assert accelerator is not None
        if getattr(valid_dataloader.dataset, "shard_by_rank", False):
            model = accelerator.prepare(model)
        else:
            model, valid_dataloader = accelerator.prepare(model, valid_dataloader)
    else:
        model.to(device)

    losses = []

    eval_num_steps = 0
    progress_bar = tqdm(
        valid_dataloader,
        desc="Validation step",
        disable=(not accelerator.is_local_main_process),
    )

    valid_metrics = wrap_metrics(valid_metrics)

    for _, batch in enumerate(progress_bar):
        eval_num_steps += 1

        if not distributed_evaluate and not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            continue

        batch = move_to_device(batch, device)
        output = model(**batch)
        loss = calculate_output_loss(
            output, accelerator, distributed=distributed_evaluate
        )
        losses.append(loss.detach())

        if valid_metrics is not None:
            if distributed_evaluate:
                gathered_objects = accelerator.gather_for_metrics((batch, output))
            else:
                gathered_objects = [batch, output]
            if accelerator.is_main_process:
                len_gather_obj = len(gathered_objects)
                assert len_gather_obj % 2 == 0, "incorrect lenght"
                for metric in valid_metrics:
                    for i in range(0, len_gather_obj, 2):
                        metric.update(
                            inputs=gathered_objects[i],
                            outputs=gathered_objects[i + 1],
                        )
        if not distributed_evaluate:
            accelerator.wait_for_everyone()

    scores = {}
    if distributed_evaluate:
        gathered_losses = accelerator.gather_for_metrics(sum(losses)).sum()
        num_proc = accelerator.num_processes
    else:
        gathered_losses = sum(losses)
        num_proc = 1

    if not accelerator.is_main_process:
        return None

    scores["loss"] = gathered_losses.item() / (num_proc * eval_num_steps)
    if valid_metrics is not None:
        for metric in valid_metrics:
            scores.update(metric.compute())
            metric.reset()
    return scores


if __name__ == "__main__":
    main()
