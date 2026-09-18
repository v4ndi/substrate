"""The TabNN side of ``_fit_one``: encode, assemble, run one trial, keep a path.

This is the second of the four places the two backend families differ. The
first -- the config mapping -- is :mod:`.assembly`; this module is the one that
strings the pieces together for the task layer, so the task layer holds a
two-line dispatch rather than a second training implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from fmlib.automl.data import CanonicalColumnMapper
from fmlib.automl.exceptions import UnsupportedBackendError
from fmlib.automl.progress import log_progress
from fmlib.automl.tasks.preparation import DataPreparation
from fmlib.automl.tasks.state import ModelEntry, TrainingInput

from .assembly import build_train_config
from .base import TabNNBackend
from .data import build_schema, prepare_processed_data
from .runner import InProcessRunner, TrialSpec

__all__ = ["fit_model_part"]


def _weights_of(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read the weights of the best step out of a trial directory.

    ``max_checkpoints=1`` keeps one *step directory*, holding the full training
    state and a bare weights file next to it. Only the weights travel into the
    artifact; optimizer and scheduler state are not part of a fitted model.
    """
    import torch

    from fmlib.train.checkpoint import latest_checkpoint_step, model_path

    step = latest_checkpoint_step(str(checkpoint_dir))
    if step is None:
        msg = f"The trial wrote no checkpoint into {checkpoint_dir}"
        raise RuntimeError(msg)
    state = torch.load(model_path(str(checkpoint_dir), step), map_location="cpu")
    return {name: value.cpu() for name, value in state.items()}


def fit_model_part(
    task: Any,
    train: TrainingInput,
    valid: TrainingInput,
    *,
    layout: str,
    group_value: Any | None = None,
    params: Mapping[str, Any] | None = None,
    trial_id: str = "trial-0",
) -> ModelEntry:
    """Train one model part with the TabNN backend.

    Args:
        task: The task facade being fitted.
        train: Training split descriptor. Not materialized.
        valid: Validation split descriptor.
        layout: ``global`` or ``per_group``.
        group_value: The group this part is for, under ``per_group``.
        params: Hyperparameters for this trial.
        trial_id: Identifier of the trial, used to name its directory.

    Returns:
        The fitted model part, carrying a number and a set of weights.

    Raises:
        UnsupportedBackendError: For a layout TabNN does not implement yet.
        RuntimeError: If the trial failed; the traceback travels with it.
    """
    if layout != "global":
        msg = (
            f"backend='tabnn' does not support model_layout={layout!r} yet; "
            "only a global model is implemented"
        )
        raise UnsupportedBackendError(msg)

    started = perf_counter()
    config = task.config
    internal = task._internal_config
    mapper = CanonicalColumnMapper.from_config(config)
    categorical_roles = (
        internal.group_column if layout == "global" else None,
        getattr(internal, "treatment_column", None),
    )

    schema = build_schema(
        train.source,
        internal,
        mapper.to_external,
        categorical_role_columns=categorical_roles,
    )
    processed = prepare_processed_data(
        config=internal,
        schema=schema,
        train_source=train.source,
        valid_source=valid.source,
        train_manifest=DataPreparation.source_manifest(train.source),
        valid_manifest=DataPreparation.source_manifest(valid.source),
        to_external=mapper.to_external,
        num_workers=1,
    )

    part = task._model_name(layout, group_value)
    trial_dir = Path(config.output_dir) / "tabnn" / part / trial_id
    class_order = getattr(task, "_class_order", None)
    train_config = build_train_config(
        config=config,
        task_name=task._task_name,
        processed=processed,
        params=dict(params or config.model_params),
        trial_dir=trial_dir,
        backend_options=task._backend_options(),
        class_order=class_order,
    )
    spec = TrialSpec(
        trial_id=trial_id,
        config=train_config,
        trial_dir=str(trial_dir),
        metric_name=train_config["automl"]["metric"],
        direction=train_config["automl"]["direction"],
        seed=int(config.random_state),
        params=dict(params or config.model_params),
    )
    runner = InProcessRunner()
    result = runner.collect(runner.submit(spec))
    if not result.completed:
        msg = f"TabNN trial {trial_id} for model part {part!r} failed: {result.error}"
        raise RuntimeError(msg)
    log_progress(
        "[tabnn fit] model=%s objective=%.12g duration_seconds=%.3f",
        part,
        result.objective,
        perf_counter() - started,
    )

    backend = TabNNBackend(
        engine=config.engine,
        params=dict(spec.params),
        random_state=int(config.random_state),
        device=internal.resolved_device,
        verbose=bool(config.verbose),
        task_name=task._task_name,
        num_classes=int(train_config["model"]["num_classes"]),
        score_transform=train_config["automl"]["score_transform"],
        class_order=tuple(class_order) if class_order else None,
        train_config=_plain(train_config),
        column_names=dict(mapper.to_external),
        hidden_states=dict(processed.hidden_states),
        preprocessor_state=processed.preprocessor().dump(),
        state_dict=_weights_of(result.checkpoint_dir),
        batch_size=int(train_config["train_dataloader"]["batch_size"]),
    )
    return ModelEntry(
        backend=backend,
        schema=schema,
        hidden_dimensions=dict(schema.hidden_states),
        best_params=dict(spec.params),
        validation_metric=float(result.objective),
        layout=layout,
        group_value=group_value,
    )


def _plain(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)
