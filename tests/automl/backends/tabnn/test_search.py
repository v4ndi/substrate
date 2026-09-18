"""Every parameter set is decided before the first trial runs.

All of them up front, because a search that fans out over jobs has no live
driver between waves, and a sampler that may not look at earlier results has no
reason to wait anyway.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.backends.search import plan_trials


def _plan(space, n_trials, **overrides):
    values = dict(
        backend="tabnn",
        engine="tabular_transformer",
        model_params={},
        search_space=space,
        n_trials=n_trials,
        random_state=42,
    )
    values.update(overrides)
    return plan_trials(**values)


def test_a_small_categorical_grid_is_run_whole():
    """Sampling a small discrete set spends the budget on repeats."""
    plan = _plan({"a": [1, 2], "b": ["x", "y"]}, n_trials=10)
    assert plan.exhaustive is True
    assert len(plan) == 4
    assert {tuple(sorted(item.items(), key=str)) for item in plan.params} == {
        (("a", 1), ("b", "x")),
        (("a", 1), ("b", "y")),
        (("a", 2), ("b", "x")),
        (("a", 2), ("b", "y")),
    }


def test_the_budget_is_a_ceiling_and_says_so():
    plan = _plan({"a": [1, 2]}, n_trials=7)
    assert len(plan) == 2
    assert plan.requested == 7
    assert plan.short_of_budget is True


def test_a_grid_larger_than_the_budget_is_sampled():
    plan = _plan({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}, n_trials=5)
    assert plan.exhaustive is False
    assert len(plan) == 5
    assert plan.short_of_budget is False


def test_sampled_sets_are_distinct():
    plan = _plan({"a": list(range(6)), "b": list(range(6))}, n_trials=12)
    seen = {tuple(sorted(item.items(), key=str)) for item in plan.params}
    assert len(seen) == len(plan)


def test_the_same_seed_plans_the_same_search():
    first = _plan({"a": list(range(8)), "b": list(range(8))}, n_trials=6)
    again = _plan({"a": list(range(8)), "b": list(range(8))}, n_trials=6)
    other = _plan(
        {"a": list(range(8)), "b": list(range(8))}, n_trials=6, random_state=7
    )
    assert first.params == again.params
    assert first.params != other.params


def test_a_continuous_axis_means_sampling_even_for_a_tiny_space():
    """An infinite product cannot be enumerated, however few axes there are."""
    plan = _plan(
        {"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}}, n_trials=3
    )
    assert plan.exhaustive is False
    assert len(plan) == 3


def test_fixed_parameters_travel_into_every_trial():
    plan = _plan({"a": [1, 2]}, n_trials=5, model_params={"batch_size": 512})
    assert all(item["batch_size"] == 512 for item in plan.params)


def test_the_packaged_tabnn_default_is_eighteen_points_against_a_budget_of_ten():
    plan = _plan(None, n_trials=10)
    assert plan.exhaustive is False
    assert len(plan) == 10
    assert {name for item in plan.params for name in item} == {
        "hidden_size",
        "num_layers",
        "lr",
    }


def test_raising_the_budget_over_the_default_grid_switches_to_full_enumeration():
    plan = _plan(None, n_trials=18)
    assert plan.exhaustive is True
    assert len(plan) == 18


@pytest.mark.parametrize("space", [{}, None])
def test_an_empty_space_is_not_mistaken_for_an_enumerable_one(space):
    plan = _plan(space or {"a": [1]}, n_trials=3)
    assert len(plan) >= 1


# --------------------------------------------------------------------------- #
# The search, end to end                                                       #
# --------------------------------------------------------------------------- #

TINY = {
    "max_epochs": [1],
    "batch_size": [128],
    "num_layers": [1],
    "num_heads": [8],
    "num_workers": [0],
    "evaluations_per_epoch": [1],
    "patience": [1],
}


def _write(directory, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    balance = rng.normal(size=rows)
    pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 1000,
        "segment": rng.choice(["a", "b"], rows),
        "balance": balance,
        "y": (rng.uniform(size=rows) < 1 / (1 + np.exp(-2 * balance))).astype(np.int8),
    }).write_parquet(directory / "part-0.parquet")
    return directory


@pytest.fixture
def data(tmp_path):
    return _write(tmp_path / "train", 400, 1), _write(tmp_path / "valid", 200, 2)


def _config(tmp_path, space, n_trials):
    return BinaryTaskConfig(
        env_type="local",
        backend="tabnn",
        engine="tabular_transformer",
        device="cpu",
        target_column="y",
        client_id_column="epk_id",
        group_column=None,
        date_column=None,
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hidden_state_columns=[],
        hyperopt=True,
        n_trials=n_trials,
        search_space=space,
        verbose=False,
        output_dir=tmp_path / "outputs",
        environment={},
    )


@pytest.mark.slow
def test_a_local_search_runs_every_planned_trial_and_keeps_the_best(tmp_path, data):
    train, valid = data
    space = TINY | {"hidden_size": [16, 32]}
    task = BinaryTask(_config(tmp_path, space, n_trials=5))
    training = task.train(train, valid)

    trials = sorted((tmp_path / "outputs" / "tabnn" / "global").iterdir())
    assert len(trials) == 2
    objectives = [
        json.loads((path / "result.json").read_text())["objective"] for path in trials
    ]
    assert training.validation_metrics["global"] == pytest.approx(max(objectives))
    assert training.best_params["global"]["hidden_size"] in {16, 32}


@pytest.mark.slow
def test_one_failing_trial_does_not_end_the_search(tmp_path, data):
    """18 does not divide by 8 heads: that trial cannot even be assembled."""
    train, valid = data
    space = TINY | {"hidden_size": [16, 18]}
    task = BinaryTask(_config(tmp_path, space, n_trials=5))
    training = task.train(train, valid)

    assert training.best_params["global"]["hidden_size"] == 16
    assert np.isfinite(training.validation_metrics["global"])


@pytest.mark.slow
def test_a_search_where_everything_fails_says_so(tmp_path, data):
    train, valid = data
    space = TINY | {"hidden_size": [18, 20]}
    task = BinaryTask(_config(tmp_path, space, n_trials=5))
    with pytest.raises(RuntimeError, match="Every TabNN trial"):
        task.train(train, valid)


@pytest.mark.slow
def test_the_data_is_encoded_once_for_the_whole_search(tmp_path, data):
    train, valid = data
    space = TINY | {"hidden_size": [16, 32]}
    task = BinaryTask(_config(tmp_path, space, n_trials=5))
    task.train(train, valid)
    processed = task.config.resolved_processed_data_path
    assert len([path for path in processed.iterdir() if path.is_dir()]) == 1
