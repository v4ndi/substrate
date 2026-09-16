"""Training and hyperparameter search owned by the boosting backend layer."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, Mapping

import numpy as np
import polars as pl

from avatar.automl.backends.boosting.base import BaseBoostingBackend
from avatar.automl.config.boosting import default_search_space
from avatar.automl.data import FeatureSchema
from avatar.automl.exceptions import ConfigError, MissingDependencyError
from avatar.automl.progress import log_progress


@dataclass(frozen=True)
class BoostingFitResult:
    """Contain the selected fitted backend and search diagnostics.

    Attributes:
        backend: Already-fitted backend selected on validation data.
        best_params: Parameters used by the selected backend.
        validation_metric: Selected validation objective value.
    """

    backend: BaseBoostingBackend
    best_params: dict[str, Any]
    validation_metric: float


def suggest_params(
    trial: Any,
    *,
    engine: str,
    model_params: Mapping[str, Any],
    search_space: Mapping[str, Any] | None,
    n_trials: int = 3,
    train_frame: pl.DataFrame | None = None,
    schema: FeatureSchema | None = None,
) -> dict[str, Any]:
    """Materialize one validated parameter set for an Optuna trial.

    Args:
        trial: Optuna trial implementing ``suggest_*`` methods.
        engine: Boosting engine used to select the default space.
        model_params: Fixed estimator parameters.
        search_space: Explicit search definitions, or ``None`` for defaults.

    Returns:
        Fixed parameters merged with values suggested for this trial.

    Raises:
        ConfigError: If a search-space definition is malformed.
    """
    space = (
        resolve_default_search_space(engine, n_trials=n_trials, train_frame=train_frame, schema=schema)
        if search_space is None
        else search_space
    )
    suggested: dict[str, Any] = {}
    for name, raw_definition in space.items():
        if isinstance(raw_definition, (list, tuple)):
            if not raw_definition:
                msg = f"Categorical search parameter {name!r} must contain at least one choice"
                raise ConfigError(msg)
            suggested[name] = trial.suggest_categorical(name, list(raw_definition))
            continue
        if not isinstance(raw_definition, Mapping):
            msg = f"Search parameter {name!r} must be a range mapping or a non-empty categorical sequence"
            raise ConfigError(msg)

        parameter_type = raw_definition.get("type")
        if parameter_type == "categorical":
            choices = raw_definition.get("choices")
            if not isinstance(choices, (list, tuple)) or not choices:
                msg = f"Categorical search parameter {name!r} requires non-empty 'choices'"
                raise ConfigError(msg)
            suggested[name] = trial.suggest_categorical(name, list(choices))
            continue
        if parameter_type not in {"int", "float"}:
            msg = f"Unsupported search parameter for {name!r}: {raw_definition!r}"
            raise ConfigError(msg)
        if "low" not in raw_definition or "high" not in raw_definition:
            msg = f"Search parameter {name!r} requires both 'low' and 'high'"
            raise ConfigError(msg)

        low, high = raw_definition["low"], raw_definition["high"]
        numerical = (int, float)
        if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, numerical) or not isinstance(high, numerical):
            msg = f"Search parameter {name!r} bounds must be numerical; got low={low!r}, high={high!r}"
            raise ConfigError(msg)
        if low > high:
            msg = f"Search parameter {name!r} requires low <= high; got low={low!r}, high={high!r}"
            raise ConfigError(msg)
        step = raw_definition.get("step")
        log = raw_definition.get("log", False)
        if not isinstance(log, bool):
            msg = f"Search parameter {name!r} field 'log' must be boolean"
            raise ConfigError(msg)
        if step is not None and (isinstance(step, bool) or not isinstance(step, numerical) or step <= 0):
            msg = f"Search parameter {name!r} step must be positive; got {step!r}"
            raise ConfigError(msg)
        incompatible_step = (parameter_type == "float" and step is not None) or (
            parameter_type == "int" and step not in {None, 1}
        )
        if log and incompatible_step:
            msg = f"Search parameter {name!r} cannot combine log=True with step={step!r}"
            raise ConfigError(msg)
        if log and low <= 0:
            msg = f"Log-scaled search parameter {name!r} requires low > 0; got {low!r}"
            raise ConfigError(msg)

        if parameter_type == "int":
            if not isinstance(low, int) or not isinstance(high, int) or (step is not None and not isinstance(step, int)):
                msg = f"Integer search parameter {name!r} requires integer bounds and step"
                raise ConfigError(msg)
            suggested[name] = trial.suggest_int(name, low, high, step=step or 1, log=log)
        else:
            suggested[name] = trial.suggest_float(name, float(low), float(high), step=step, log=log)
    return dict(model_params) | suggested


def resolve_default_search_space(
    engine: str,
    *,
    n_trials: int,
    train_frame: pl.DataFrame | None,
    schema: FeatureSchema | None,
) -> dict[str, dict[str, Any]]:
    """Resolve default dimensions from the actual model-part training matrix."""
    has_categorical = bool(schema and schema.categorical)
    has_nan = False
    if train_frame is not None and schema is not None:
        has_nan = any(
            train_frame[column].null_count() > 0
            or (
                train_frame[column].dtype in {pl.Float32, pl.Float64}
                and train_frame[column].is_nan().fill_null(False).any()
            )
            for column in schema.numerical
        )
    return default_search_space(
        engine,
        n_trials=n_trials,
        has_nan=has_nan,
        has_categorical=has_categorical,
    )


def fit_boosting_model(
    *,
    backend_class: type[BaseBoostingBackend],
    engine: str,
    model_params: Mapping[str, Any],
    search_space: Mapping[str, Any] | None,
    hyperopt: bool,
    n_trials: int | None,
    random_state: int,
    device: str,
    verbose: bool | int,
    train_frame: pl.DataFrame,
    train_target: np.ndarray,
    valid_frame: pl.DataFrame,
    valid_target: np.ndarray,
    schema: FeatureSchema,
    objective_metric: Callable[[np.ndarray, np.ndarray], float],
    direction: Literal["minimize", "maximize"],
    backend_options: Mapping[str, Any] | None = None,
) -> BoostingFitResult:
    """Fit once or run Optuna and retain the best already-fitted model.

    Args:
        backend_class: Concrete task-specific boosting adapter.
        engine: Native boosting engine.
        model_params: Fixed estimator parameters.
        search_space: Search-space overrides, or ``None`` for defaults.
        hyperopt: Run Optuna when ``True``.
        n_trials: Number of Optuna trials.
        random_state: Sampler and estimator seed.
        device: Runtime ``cpu`` or ``gpu`` device.
        verbose: Native estimator and progress logging setting.
        train_frame: Normalized training frame.
        train_target: Training target array.
        valid_frame: Normalized validation frame.
        valid_target: Validation target array.
        schema: Ordered fitted feature schema.
        objective_metric: Function mapping validation targets and scores to a scalar.
        direction: Whether Optuna minimizes or maximizes the objective.
        backend_options: Optional task-specific backend constructor arguments.

    Returns:
        Selected fitted backend, parameters, validation value, and trial history.

    Raises:
        MissingDependencyError: If Optuna is requested but unavailable.
        RuntimeError: If hyperparameter search produces no fitted model.
    """

    def make_backend(params: Mapping[str, Any]) -> BaseBoostingBackend:
        return backend_class(
            engine=engine,
            params=params,
            random_state=random_state,
            device=device,
            verbose=verbose,
            **dict(backend_options or {}),
        )

    effective_space = (
        resolve_default_search_space(
            engine,
            n_trials=n_trials or 0,
            train_frame=train_frame,
            schema=schema,
        )
        if search_space is None and hyperopt
        else search_space
    )
    preparation_backend = make_backend(model_params)
    preparation_started = perf_counter()
    log_progress(
        "[boosting prepare 1/1] engine=%s building reusable estimator data prequantize=%s",
        engine,
        hyperopt and preparation_backend.can_prequantize(effective_space),
    )
    prepared = preparation_backend.prepare_fit_data(
        train_frame,
        train_target,
        schema,
        valid_frame=valid_frame,
        valid_target=valid_target,
        prequantize=hyperopt and preparation_backend.can_prequantize(effective_space),
    )
    log_progress(
        "[boosting prepare 1/1] engine=%s completed duration_seconds=%.3f",
        engine,
        perf_counter() - preparation_started,
    )
    if not hyperopt:
        fit_started = perf_counter()
        log_progress("[boosting fit 1/1] engine=%s started", engine)
        backend = make_backend(model_params)
        backend.fit_prepared(prepared)
        value = objective_metric(valid_target, backend.predict_prepared_score(prepared.valid_prediction_features))
        log_progress(
            "[boosting fit 1/1] engine=%s completed duration_seconds=%.3f objective_value=%.12g",
            engine,
            perf_counter() - fit_started,
            value,
        )
        return BoostingFitResult(backend, dict(model_params), value)

    if n_trials is None:
        msg = "hyperopt=True requires a resolved n_trials value"
        raise RuntimeError(msg)

    try:
        import optuna
    except ImportError as exc:
        msg = "hyperopt=True requires the 'optuna' package"
        raise MissingDependencyError(msg) from exc

    best_backend: BaseBoostingBackend | None = None
    best_params: dict[str, Any] = {}
    best_value = float("-inf") if direction == "maximize" else float("inf")

    def objective(trial: Any) -> float:
        nonlocal best_backend, best_params, best_value
        trial_started = perf_counter()
        params = suggest_params(trial, engine=engine, model_params=model_params, search_space=effective_space)
        backend = make_backend(params)
        backend.fit_prepared(prepared)
        value = objective_metric(valid_target, backend.predict_prepared_score(prepared.valid_prediction_features))
        improved = value > best_value if direction == "maximize" else value < best_value
        if best_backend is None or improved:
            best_backend = backend
            best_params = params
            best_value = value
        log_progress(
            "[optuna %d/%d] engine=%s completed duration_seconds=%.3f objective_value=%.12g",
            trial.number + 1,
            n_trials,
            engine,
            perf_counter() - trial_started,
            value,
        )
        return value

    search_started = perf_counter()
    log_progress("[optuna 0/%d] engine=%s creating study random_state=%d", n_trials, engine, random_state)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(verbose))
    if best_backend is None:
        msg = "Hyperparameter search completed without a fitted model"
        raise RuntimeError(msg)
    log_progress(
        "[optuna %d/%d] engine=%s search completed duration_seconds=%.3f best_value=%.12g",
        n_trials,
        n_trials,
        engine,
        perf_counter() - search_started,
        best_value,
    )
    return BoostingFitResult(best_backend, best_params, best_value)
