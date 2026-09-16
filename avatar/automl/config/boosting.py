"""Packaged model defaults and hyperparameter spaces for boosting engines."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_DEFAULTS_PATH = Path(__file__).with_name("defaults") / "boosting_search_space.yaml"
_DEFAULT_MODEL_PARAMS = {
    "catboost": {
        "thread_count": 4,
        "num_trees": 3000,
        "learning_rate": 0.03,
        "l2_leaf_reg": 1e-2,
        "bootstrap_type": "Bernoulli",
        "grow_policy": "SymmetricTree",
        "max_depth": 5,
        "min_data_in_leaf": 1,
        "one_hot_max_size": 10,
        "fold_permutation_block": 1,
        "boosting_type": "Plain",
        "boost_from_average": True,
        "od_type": "Iter",
        "od_wait": 100,
        "max_bin": 32,
        "feature_border_type": "GreedyLogSum",
        "nan_mode": "Min",
    },
    "xgboost": {
        "n_estimators": 3000,
        "early_stopping_rounds": 100,
    },
}


def default_model_params(
    engine: str,
    *,
    task: str | None = None,
    train_rows: int | None = None,
) -> dict[str, Any]:
    """Return LightAutoML model defaults, optionally adapted to the input size."""
    try:
        params = dict(_DEFAULT_MODEL_PARAMS[engine])
    except KeyError as exc:
        msg = f"Missing default model parameters for engine={engine!r}"
        raise ValueError(msg) from exc
    if engine != "catboost" or task is None or train_rows is None:
        return params
    if task == "binary":
        if train_rows <= 6_000:
            learning_rate, num_trees = 0.02, 500
        elif train_rows <= 20_000:
            learning_rate, num_trees = 0.035, 5_000
        elif train_rows <= 50_000:
            learning_rate, num_trees = 0.03, 5_000
        elif train_rows <= 60_000:
            learning_rate, num_trees = 0.05, 2_000
        elif train_rows <= 100_000:
            learning_rate, num_trees = 0.045, 1_500
        elif train_rows <= 150_000:
            learning_rate, num_trees = 0.045, 3_000
        elif train_rows <= 300_000:
            learning_rate, num_trees = 0.045, 2_000
        else:
            learning_rate, num_trees = 0.05, 3_000
        params.update(learning_rate=learning_rate, num_trees=num_trees)
    elif task == "multiclass":
        params.update(learning_rate=0.03, num_trees=3_000 if train_rows <= 100_000 else 4_000)
    elif task == "regression":
        params.update(learning_rate=0.05, num_trees=2_000, od_wait=300)
    return params


def default_search_space(
    engine: str,
    *,
    n_trials: int,
    has_nan: bool = False,
    has_categorical: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return the LightAutoML search dimensions enabled by data and trial budget."""
    payload = OmegaConf.to_container(OmegaConf.load(_DEFAULTS_PATH), resolve=True)
    if not isinstance(payload, dict) or engine not in payload:
        msg = f"Missing default search space for engine={engine!r} in {_DEFAULTS_PATH}"
        raise ValueError(msg)
    space = {name: dict(definition) for name, definition in payload[engine].items()}
    if engine == "catboost":
        if not has_nan:
            space.pop("nan_mode")
        if n_trials <= 20:
            space.pop("l2_leaf_reg")
        if n_trials <= 50:
            space.pop("min_data_in_leaf")
            space.pop("one_hot_max_size")
        elif not has_categorical:
            space.pop("one_hot_max_size")
    elif engine == "xgboost" and n_trials <= 30:
        for name in ("min_child_weight", "reg_alpha", "reg_lambda"):
            space.pop(name)
    return space
