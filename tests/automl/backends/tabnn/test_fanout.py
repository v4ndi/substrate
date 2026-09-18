"""One trial, one job: the search as a fan-out the driver need not outlive.

There are no waves. Every parameter set is known before the first submit, so
the driver's only remaining jobs are to submit and, later, to collect — which
matters because the driver is a notebook container that may die, and a job
cannot submit a job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from automl.scheduler_stub import LocalScheduler
from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.environment import EnvironmentRunner

pytestmark = pytest.mark.slow

TINY = {
    "max_epochs": [1],
    "batch_size": [128],
    "num_layers": [1],
    "num_heads": [8],
    "num_workers": [0],
    "evaluations_per_epoch": [1],
    "patience": [1],
    "hidden_size": [16, 32],
}


def _data(path: Path, rows: int = 400, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    balance = rng.normal(size=rows)
    pl.DataFrame({
        "epk_id": np.arange(rows),
        "segment": rng.choice(["a", "b"], rows),
        "balance": balance,
        "y": (rng.uniform(size=rows) < 1 / (1 + np.exp(-2 * balance))).astype(np.int8),
    }).write_parquet(path)
    return path


def _config(tmp_path, **overrides):
    venv = tmp_path / "venv"
    venv.mkdir(exist_ok=True)
    values = {
        "env_type": "osiris",
        "backend": "tabnn",
        "engine": "tabular_transformer",
        "device": "gpu",
        "target_column": "y",
        "client_id_column": "epk_id",
        "group_column": None,
        "date_column": None,
        "categorical_columns": ["segment"],
        "numerical_columns": ["balance"],
        "hidden_state_columns": [],
        "hyperopt": True,
        "n_trials": 5,
        "search_space": dict(TINY),
        "verbose": False,
        "output_dir": tmp_path / "outputs",
        "environment": {"venv_path": str(venv), "pool": "common", "num_gpus": 1},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


def _task(tmp_path, scheduler, **overrides):
    task = BinaryTask(_config(tmp_path, **overrides))
    task._environment_runner = EnvironmentRunner(scheduler)
    return task


def test_one_job_per_trial_and_the_best_one_becomes_the_model(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)

    task.train(train, train)
    table = task.status()

    assert len(scheduler.jobs) == 2
    assert all(job.state == "succeeded" for job in scheduler.jobs)
    assert table["state"].to_list() == ["succeeded", "succeeded"]

    operation = task._store.latest("train")
    assert operation["state"] == "succeeded"
    objectives = [
        json.loads(Path(job["result_path"]).read_text())["validation_metrics"]["global"]
        for job in operation["jobs"]
    ]
    published = json.loads((task.path / "training_result.json").read_text())
    assert published["validation_metrics"]["global"] == pytest.approx(max(objectives))


def test_every_trial_is_named_and_its_parameters_are_persisted_before_submit(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler)
    task.train(train, train)

    operation = task._store.latest("train")
    trials = [job["trial"] for job in operation["jobs"]]
    assert [item["trial_id"] for item in trials] == ["trial-0000", "trial-0001"]
    assert {item["params"]["hidden_size"] for item in trials} == {16, 32}
    assert [job.name.split("-")[-1] for job in scheduler.jobs] == ["0000", "0001"]


def test_a_failed_trial_does_not_stop_the_operation(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler)
    task.train(train, train)

    scheduler.jobs[0].state = "failed"
    scheduler.jobs[0].error = "the trial died"
    scheduler._run(scheduler.jobs[1])

    task.status()
    operation = task._store.latest("train")
    assert operation["state"] == "succeeded"
    published = json.loads((task.path / "training_result.json").read_text())
    assert published["best_params"]["global"]["hidden_size"] == 32


def test_a_capped_fan_out_submits_in_chunks_and_resumes(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler, max_parallel_jobs=1)
    task.train(train, train)

    assert len(scheduler.jobs) == 1
    operation = task._store.latest("train")
    assert operation["state"] == "submitting"
    assert [job["job_id"] for job in operation["jobs"]] == ["job-1", None]

    # The first job finishes; a later status() submits the next one.
    scheduler.run_pending()
    task.status()
    assert len(scheduler.jobs) == 2
    assert [job["job_id"] for job in task._store.latest("train")["jobs"]] == [
        "job-1",
        "job-2",
    ]


def test_a_driver_that_died_mid_submit_is_resumed_not_replanned(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler, max_parallel_jobs=1)
    task.train(train, train)
    planned_before = [
        job["trial"]["params"] for job in task._store.latest("train")["jobs"]
    ]

    # A different process, same entity on disk.
    revived = BinaryTask.load(task.path)
    revived._environment_runner = EnvironmentRunner(scheduler)
    scheduler.run_pending()
    revived.status()

    operation = revived._store.latest("train")
    assert [job["trial"]["params"] for job in operation["jobs"]] == planned_before
    assert len({job["job_id"] for job in operation["jobs"]}) == 2


def test_the_boosting_path_still_submits_one_job_per_part(tmp_path):
    """Fan-out is TabNN's; a boosting trial costs minutes and stays put."""
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(
        tmp_path,
        scheduler,
        backend="boosting",
        engine="catboost",
        hyperopt=True,
        n_trials=2,
        search_space={"depth": [2, 3]},
    )
    task.train(train, train)

    assert len(scheduler.jobs) == 1
    operation = task._store.latest("train")
    assert operation["jobs"][0]["trial"] is None
