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

from fmlib.automl.backends.search import plan_trials
from fmlib.automl.data import CanonicalColumnMapper
from fmlib.automl.progress import log_progress
from fmlib.automl.tasks.preparation import DataPreparation
from fmlib.automl.tasks.state import ModelEntry, TrainingInput

from .assembly import build_train_config
from .data import build_schema, prepare_processed_data
from .runner import InProcessRunner, TrialResult, TrialSpec

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
        RuntimeError: If the trial failed; the traceback travels with it.
    """
    started = perf_counter()
    config = task.config
    internal = task._internal_config
    mapper = CanonicalColumnMapper.from_config(config)
    # The treatment column is an ordinary categorical feature of a response
    # model, and an input of its own for an uplift one -- there the S-Learner
    # feeds it in itself, so it must not also become a feature.
    is_uplift = task._task_name == "uplift"
    categorical_roles = (
        internal.group_column if layout == "global" else None,
        None if is_uplift else getattr(internal, "treatment_column", None),
    )

    schema = build_schema(
        train.source,
        internal,
        mapper.to_external,
        categorical_role_columns=categorical_roles,
    )
    # A per-group model reads one partition of the encoded data; a global model
    # reads the flat layout. Both are cut out of the *same* encoding, so every
    # part shares one vocabulary and one scaling.
    processed = prepare_processed_data(
        config=internal,
        schema=schema,
        train_source=train.source,
        valid_source=valid.source,
        train_manifest=DataPreparation.source_manifest(train.source),
        valid_manifest=DataPreparation.source_manifest(valid.source),
        to_external=mapper.to_external,
        num_workers=1,
        partition_by=internal.group_column if layout == "per_group" else None,
    )

    part = task._model_name(layout, group_value)
    class_order = getattr(task, "_class_order", None)
    root = Path(config.output_dir) / "tabnn" / part
    candidates = _trial_params(task, params)
    forced = getattr(task, "_forced_trial", None)
    if forced:
        trial_id = str(forced["trial_id"])
    runner = InProcessRunner()

    results: list[tuple[TrialResult, Mapping[str, Any], Any]] = []
    for index, trial_params in enumerate(candidates):
        name = f"{trial_id}-{index:04d}" if len(candidates) > 1 else trial_id
        try:
            train_config = build_train_config(
                config=config,
                task_name=task._task_name,
                processed=processed,
                params=dict(trial_params),
                trial_dir=root / name,
                group_value=group_value,
                backend_options=_backend_options(task),
                class_order=class_order,
            )
            spec = TrialSpec(
                trial_id=name,
                config=train_config,
                trial_dir=str(root / name),
                metric_name=train_config["automl"]["metric"],
                direction=train_config["automl"]["direction"],
                seed=int(config.random_state),
                params=dict(trial_params),
            )
            result = runner.collect(runner.submit(spec))
        except Exception as error:
            # A parameter set the assembly rejects -- a hidden size that does
            # not divide by the head count, say -- is a failed trial, not a
            # failed search. The boosting path keeps today's behaviour, where
            # the first exception ends the study.
            result = TrialResult(
                trial_id=name,
                state="FAIL",
                error=f"{type(error).__name__}: {error}",
            )
            train_config = None
        results.append((result, dict(trial_params), train_config))
        log_progress(
            "[tabnn trial %d/%d] model=%s state=%s objective=%s",
            index + 1,
            len(candidates),
            part,
            result.state,
            "n/a" if result.objective is None else f"{result.objective:.12g}",
        )

    completed = [item for item in results if item[0].completed]
    if not completed:
        errors = "; ".join(f"{item[0].trial_id}: {item[0].error}" for item in results)
        msg = f"Every TabNN trial for model part {part!r} failed: {errors}"
        raise RuntimeError(msg)
    direction = next(
        item[2]["automl"]["direction"] for item in results if item[2] is not None
    )
    best = (max if direction == "max" else min)(
        completed, key=lambda item: item[0].objective
    )
    result, best_params, train_config = best
    if len(completed) < len(results):
        log_progress(
            "[tabnn fit] model=%s trials_completed=%d/%d (failed trials do not end a search)",
            part,
            len(completed),
            len(results),
        )
    log_progress(
        "[tabnn fit] model=%s best_trial=%s objective=%.12g duration_seconds=%.3f",
        part,
        result.trial_id,
        result.objective,
        perf_counter() - started,
    )

    adapter = task._backend_class_for(config.backend)
    backend = adapter(
        engine=config.engine,
        params=dict(best_params),
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
        validation_metric=float(result.objective),
        batch_size=int(train_config["train_dataloader"]["batch_size"]),
    )
    return ModelEntry(
        backend=backend,
        schema=schema,
        hidden_dimensions=dict(schema.hidden_states),
        best_params=dict(best_params),
        validation_metric=float(result.objective),
        layout=layout,
        group_value=group_value,
    )


def _backend_options(task: Any) -> dict[str, Any]:
    """Task-specific values the assembly needs, from the hook that already exists."""
    # _backend_options is SupervisedTask's hook; uplift does not have it,
    # because its boosting adapter takes its arguments another way.
    options = dict(getattr(task, "_backend_options", dict)())
    treatment = getattr(task._internal_config, "treatment_column", None)
    if task._task_name == "uplift" and treatment:
        options["treatment_column"] = treatment
    return options


def _plain(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def _trial_params(
    task: Any, params: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    """Decide the parameter sets of this model part, before any of them runs.

    Without ``hyperopt`` there is exactly one set: whatever the user fixed.
    With it, the plan comes from :func:`plan_trials`, and how many trials it
    actually contains is logged — a budget of ten that produces six looks like
    four lost trials unless somebody says otherwise.
    """
    config = task.config
    if params is not None:
        return [dict(params)]
    forced = getattr(task, "_forced_trial", None)
    if forced:
        # One job of a fan-out: the driver already chose this trial's
        # parameters, and choosing again here would mean K jobs each running
        # the whole search.
        return [dict(forced["params"])]
    if not config.hyperopt:
        return [dict(config.model_params)]
    plan = plan_trials(
        backend=config.backend,
        engine=config.engine,
        model_params=config.model_params,
        search_space=config.search_space,
        n_trials=int(config.n_trials),
        random_state=int(config.random_state),
    )
    log_progress(
        "[tabnn search] trials=%d budget=%d mode=%s",
        len(plan),
        plan.requested,
        "full grid" if plan.exhaustive else "random sample",
    )
    if plan.short_of_budget:
        log_progress(
            "[tabnn search] fewer trials than requested: the space has %d distinct "
            "points against a budget of %d",
            len(plan),
            plan.requested,
        )
    return [dict(item) for item in plan.params]
