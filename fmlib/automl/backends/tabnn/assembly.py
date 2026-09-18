"""Turn an AutoML task plus one trial's parameters into an fmlib config.

**The config-to-objects layer already exists and a second one must not be
written.** ``fmlib/train/__main__.py`` builds a run out of
``hydra.utils.instantiate`` and the ``init_*`` helpers in
``fmlib/utils/init_modules.py``; the runner here calls exactly those. So this
module produces a ``DictConfig`` and nothing else.

Four things follow from that, and they are the reason for the choice:

1. **Acceptance criterion 21** -- AutoML equals a hand-written fmlib run to
   1e-6 -- is nearly true by construction, because it is the same assembly
   code rather than two parallel implementations to reconcile.
2. **A trial's spec is its config.** ``run_spec.json`` is this ``DictConfig``
   plus AutoML metadata: one shape for all three runners, and an Osiris job's
   entrypoint is nearly ``python -m fmlib.train``.
3. **A new architecture is a config, not code:** an MLP or an FT-transformer is
   a different ``_target_``.
4. **A trial reproduces by hand:** the config is in the artifact, and
   ``python -m fmlib.train --config-dir=... --config-name=...`` runs it.

What the generator decides itself, because Hydra cannot:

* ``steps_before_evaluation``, from the row count and the batch size -- so the
  config can only be built once the encoded dataset's size is known;
* ``amp``, from the hardware;
* the callbacks, listed explicitly and in order;
* the validation metric, as a ``_target_`` on the AutoML bridge;
* paths and column names, from the encoded directory;
* the translation of flat search names into config paths. ``hidden_size``
  lands in two places -- the embedding and the encoder -- which is the argument
  for flat names: with dotted keys the user would have to know that.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from fmlib.automl.exceptions import ConfigError
from fmlib.automl.metrics import resolve_metric

from .data import ProcessedData

__all__ = [
    "CONFIG_CONTRACT_VERSION",
    "DEFAULTS",
    "build_train_config",
    "load_template",
    "resolve_amp",
]

#: Written into the artifact beside the generated config. ``_target_`` strings
#: are paths, and paths move; a version is what lets a loader normalize them
#: instead of discovering the problem at instantiate time.
CONFIG_CONTRACT_VERSION = "1"

TEMPLATES = Path(__file__).parent / "templates"

#: Fixed on purpose and not searched (design 7.2). The epoch ceiling buys time
#: rather than metric while early stopping is live; ``amp`` is a launch
#: parameter, chosen from the hardware and recorded, because trials run in
#: different precision are not comparable.
DEFAULTS: dict[str, Any] = {
    "batch_size": 4096,
    "num_heads": 8,
    "attn_dropout": 0.15,
    "dropout_p": 0.2,
    "out_head_hidden_dim": 256,
    "aggregation": "mean",
    "weight_decay": 1e-2,
    "max_epochs": 30,
    "patience": 5,
    "num_workers": 4,
    "evaluations_per_epoch": 8,
    "hidden_size": 64,
    "num_layers": 3,
    "lr": 3e-4,
}


def load_template(task_name: str) -> dict[str, Any]:
    """Read the fragment that says how one task differs from the others.

    Args:
        task_name: AutoML task name.

    Returns:
        The parsed template.

    Raises:
        ConfigError: If no template is packaged for that task.
    """
    path = TEMPLATES / f"{task_name}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in TEMPLATES.glob("*.yaml")))
        msg = f"No TabNN template for task {task_name!r}; available: {available}"
        raise ConfigError(msg)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_amp(device: str) -> str:
    """Pick mixed precision from the hardware, never from the search space.

    Args:
        device: ``cpu`` or ``gpu``.

    Returns:
        ``"bf16"`` on an Ampere-or-newer GPU, ``"no"`` otherwise.
    """
    if device != "gpu":
        return "no"
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "no"


def _steps_before_evaluation(
    rows: int, batch_size: int, per_epoch: int, world_size: int = 1
) -> int:
    """How often to validate, so patience means hours rather than weeks.

    On three million rows one epoch is minutes and validating once an epoch is
    right. On a hundred million it is hours, and ``patience=5`` would mean
    "stop in a week". The cadence therefore follows the data: roughly
    ``per_epoch`` validations per epoch.

    The count is **per rank**, because that is what the loop counts. A cadence
    computed from the whole corpus would, on several ranks, be longer than the
    epoch itself -- and then a short run would finish having never validated,
    with no best metric and a trial that looks like a failure.
    """
    per_rank = max(1, math.ceil(rows / max(1, world_size)))
    steps_per_epoch = max(1, math.ceil(per_rank / max(1, batch_size)))
    return max(1, min(steps_per_epoch, math.ceil(steps_per_epoch / max(1, per_epoch))))


def _dataloader(
    split_path: Path,
    *,
    processed: ProcessedData,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    is_regression: bool,
    device: str,
) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "_target_": "fmlib.data.TabularDataset",
        "path": str(split_path),
        "shuffle_files": shuffle,
        "shuffle_pq": shuffle,
    }
    if processed.hidden_states:
        dataset["hidden_state_columns"] = list(processed.hidden_states)
    return {
        "_target_": "torch.utils.data.DataLoader",
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device == "gpu",
        "drop_last": False,
        "collate_fn": {
            "_target_": "fmlib.data.TabularCollateFn",
            "target_column": processed.target_column,
            "is_regression": is_regression,
        },
    }


def build_train_config(
    *,
    config: Any,
    task_name: str,
    processed: ProcessedData,
    params: Mapping[str, Any],
    trial_dir: str | Path,
    group_value: Any | None = None,
    backend_options: Mapping[str, Any] | None = None,
    device: str | None = None,
    class_order: tuple[Any, ...] | None = None,
    mlflow: Mapping[str, Any] | None = None,
    world_size: int = 1,
) -> DictConfig:
    """Compose the fmlib config one trial is trained from.

    Args:
        config: Public AutoML task configuration.
        task_name: AutoML task name, selecting the template.
        processed: The encoded data this trial reads.
        params: This trial's hyperparameters, flat names as in the search space.
        trial_dir: Where checkpoints and the result of this trial go.
        group_value: Under a partitioned layout, the group this model part
            reads. ``None`` is the flat layout a global model needs.
        backend_options: Task-specific values, such as ``num_classes``.
        device: ``cpu`` or ``gpu``; defaults to the configuration's.
        class_order: Multiclass labels in training id order, for the metric.
        mlflow: Tracking settings; omitted entirely when not configured.
        world_size: Ranks this trial will run on. Only the validation cadence
            depends on it, and getting it wrong is what makes a multi-rank
            trial finish without ever validating.

    Returns:
        A ``DictConfig`` the existing ``instantiate`` / ``init_*`` layer expands.

    Raises:
        ConfigError: If the template needs a value the task did not supply, or
            the head width does not divide by the number of attention heads.
    """
    template = load_template(task_name)
    options = dict(backend_options or {})
    device = device or config.resolved_device
    settings = DEFAULTS | dict(params)

    num_classes = template["model"].get("num_classes")
    if num_classes is None:
        num_classes = options.get("num_classes")
    if not isinstance(num_classes, int) or num_classes < 1:
        msg = (
            f"Task {task_name!r} needs num_classes from the task layer; "
            f"got {num_classes!r}"
        )
        raise ConfigError(msg)

    hidden_size = int(settings["hidden_size"])
    num_heads = int(settings["num_heads"])
    if hidden_size % num_heads:
        msg = (
            f"hidden_size={hidden_size} must divide by num_heads={num_heads}; "
            "a search space that changes hidden_size has to keep that true"
        )
        raise ConfigError(msg)

    metric = resolve_metric(config.optimization_metric, task_name, "optimization")
    direction = "max" if metric.optimization_direction == "maximize" else "min"
    batch_size = int(settings["batch_size"])
    trial_dir = str(trial_dir)

    model: dict[str, Any] = {
        "_target_": "fmlib.pipeline.tabular.SupervisedLearner",
        "embedding": {
            "_target_": "fmlib.nn.embedding.TabularEmbedding",
            "num_numerical_features": processed.num_numerical or None,
            "vocab_size": processed.vocab_size or None,
            "hidden_size": hidden_size,
        },
        "tabular_encoder": {
            "_target_": "fmlib.nn.tabular.TabularTransformer",
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_layers": int(settings["num_layers"]),
            "attn_dropout": float(settings["attn_dropout"]),
        },
        "aggregation_config": {"name": settings["aggregation"]},
        "num_classes": num_classes,
        "task_type": template["model"]["task_type"],
        "dropout_p": float(settings["dropout_p"]),
        "out_head_hidden_dim": int(settings["out_head_hidden_dim"]),
    }
    if processed.hidden_states:
        # Late fusion: the embedding arrives as a vector and is layer-normalised
        # whole, which is the geometry the producing model made.
        model["normalize_hidden_states"] = dict(processed.hidden_states)

    payload: dict[str, Any] = {
        "contract_version": CONFIG_CONTRACT_VERSION,
        "automl": {
            "task": task_name,
            "backend": "tabnn",
            "engine": config.engine,
            "metric": metric.name,
            "direction": direction,
            "score_transform": template["score_transform"],
            "processed_key": processed.key,
            "params": dict(params),
        },
        "amp": resolve_amp(device),
        "distributed": {"backend": None, "gradient_accumulation_steps": 1},
        "ddp": {"find_unused_parameters": False},
        "model": model,
        "train_dataloader": _dataloader(
            processed.split_path("train", group_value),
            processed=processed,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(settings["num_workers"]),
            is_regression=bool(template["is_regression"]),
            device=device,
        ),
        "valid_dataloader": _dataloader(
            processed.split_path("valid", group_value),
            processed=processed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(settings["num_workers"]),
            is_regression=bool(template["is_regression"]),
            device=device,
        ),
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "_partial_": True,
            "lr": float(settings["lr"]),
            "weight_decay": float(settings["weight_decay"]),
            # Explicit and recorded: a flag that silently multiplies the
            # learning rate by the device count would make trials run on one
            # card and on several incomparable.
            "scale_lr_multigpu": False,
        },
        "scheduler": {
            "_target_": "transformers.optimization.get_scheduler",
            "_partial_": True,
            "name": "constant",
        },
        "train": {
            "num_epochs": int(settings["max_epochs"]),
            "seed": int(config.random_state),
            "clip_grad_norm": settings.get("clip_grad_norm"),
            "steps_before_evaluation": _steps_before_evaluation(
                processed.train.rows,
                batch_size,
                int(settings["evaluations_per_epoch"]),
                world_size,
            ),
        },
        "metrics": {
            "valid_metrics": {
                "_target_": "fmlib.automl.backends.tabnn.metric.AutoMLMetric",
                "metric_name": metric.name,
                "task_name": task_name,
                "score_transform": template["score_transform"],
                "class_order": list(class_order) if class_order else None,
            }
        },
        # Explicit, and in this order: early stopping has to run before the
        # checkpointer, because clearing `should_save` is how "keep only the
        # best" works. `build_callbacks` only falls back to the default list --
        # profiler, progress bars, throughput -- when this key is absent.
        "callbacks": [
            {
                "_target_": "fmlib.train.callbacks.EarlyStoppingCallback",
                "early_stopping": {
                    "_target_": "fmlib.train.early_stopping.EarlyStopping",
                    "main_metric": metric.name,
                    "patience": int(settings["patience"]),
                    "strategy": direction,
                },
            },
            {
                "_target_": "fmlib.train.callbacks.CheckpointCallback",
                "directory": trial_dir,
                "max_checkpoints": 1,
            },
        ],
        "logging": {"enable": False},
    }
    if mlflow:
        payload["mlflow"] = dict(mlflow)
        payload["callbacks"].append({
            "_target_": "fmlib.train.callbacks.MLflowCallback",
            "experiment_name": mlflow.get("experiment_name"),
            "run_name": mlflow.get("run_name"),
            "tracking_uri": mlflow.get("tracking_uri"),
        })
    return OmegaConf.create(payload)
