"""The executing scheduler stub: proof that the remote path runs, not just submits.

These tests are the bench. What they check here is that the stand-in really
executes a job, and that each fault knob produces the condition it claims to --
so that the reliability work can be tested against a thing that misbehaves on
demand rather than against a wish.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from automl.scheduler_stub import (
    ContainerNotAvailable,
    LocalScheduler,
    SchedulerFault,
)
from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.environment import EnvironmentRunner


def _data(path: Path, rows: int = 200, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    pl.DataFrame({
        "epk_id": np.arange(rows),
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "group": rng.choice(["a", "b"], rows),
        "feature": rng.normal(size=rows),
        "target": rng.integers(0, 2, rows),
    }).write_parquet(path)
    return path


def _config(tmp_path, **overrides):
    venv = tmp_path / "venv"
    venv.mkdir(exist_ok=True)
    values = {
        "env_type": "osiris",
        "backend": "boosting",
        "engine": "catboost",
        "device": "gpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ["feature"],
        "hidden_state_columns": (),
        "model_layout": "global",
        "hyperopt": False,
        "model_params": {"iterations": 5, "depth": 2, "thread_count": 1},
        "verbose": False,
        "output_dir": tmp_path / "outputs",
        "environment": {"venv_path": str(venv), "pool": "common"},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


def _task(tmp_path, scheduler, **overrides):
    task = BinaryTask(_config(tmp_path, **overrides))
    task._environment_runner = EnvironmentRunner(scheduler)
    return task


# --------------------------------------------------------------------------- #
# It executes                                                                  #
# --------------------------------------------------------------------------- #
def test_a_submitted_job_is_actually_run(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)

    task.train(train, train)

    assert [job.state for job in scheduler.jobs] == ["succeeded"]
    spec = json.loads(Path(scheduler.jobs[0].spec_path).read_text())
    assert Path(spec["result_path"]).is_file()
    assert spec["action"] == "train"


def test_the_driver_sees_the_result_of_a_job_it_submitted(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)

    table = task.status()
    assert table["state"].to_list() == ["succeeded"]
    assert task._store.latest("train")["state"] in {"succeeded", "aggregate"}


def test_one_job_per_model_part(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler, model_layout="global_and_per_group")
    task.train(train, train)

    assert len(scheduler.jobs) == 3
    assert all(job.state == "succeeded" for job in scheduler.jobs)
    assert len({job.spec_path for job in scheduler.jobs}) == 3


def test_deferred_execution_lets_a_test_choose_when_a_job_finishes(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler)
    task.train(train, train)

    assert task.status()["state"].to_list() == ["queued"]
    assert [job.name for job in scheduler.run_pending()] == [scheduler.jobs[0].name]
    assert task.status()["state"].to_list() == ["succeeded"]


# --------------------------------------------------------------------------- #
# It misbehaves on cue                                                         #
# --------------------------------------------------------------------------- #
def test_a_refused_submit_raises_and_records_nothing_for_that_job(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(fail_create_on=2)
    task = _task(tmp_path, scheduler, model_layout="global_and_per_group")

    with pytest.raises(SchedulerFault, match="submit #2"):
        task.train(train, train)

    # The first submit really happened and really ran; the second never became
    # a job. Whether its id survived in operation.json is P2's business.
    assert len(scheduler.create_calls) == 2
    assert [job.state for job in scheduler.jobs] == ["succeeded"]


def test_a_vanished_job_stops_appearing_in_the_listing(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    name = scheduler.jobs[0].name

    assert scheduler.list()["jobs"]
    scheduler.vanish = [name]
    assert scheduler.list()["jobs"] == []
    assert task.status()["state"].to_list() == ["missing"]


def test_the_transient_container_error_is_served_once(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    scheduler.container_unavailable_for = [scheduler.jobs[0].name]

    first = scheduler.list()["jobs"][0]["state"]
    second = scheduler.list()["jobs"][0]["state"]
    assert "is not available" in first
    assert second == "succeeded"


def test_a_corrupt_result_is_left_for_the_driver_to_find(tmp_path):
    """A job that ran, and a result.json nobody can read: the half-written case."""
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler)
    task.train(train, train)

    job = scheduler.jobs[0]
    scheduler.corrupt_result = [job.name]
    scheduler.run_pending()

    assert job.state == "succeeded"
    spec = json.loads(Path(job.spec_path).read_text())
    with pytest.raises(json.JSONDecodeError):
        json.loads(Path(spec["result_path"]).read_text())


def test_a_job_whose_spec_cannot_run_is_reported_failed(tmp_path):
    scheduler = LocalScheduler()
    scheduler.create(name="broken", args=["--spec", str(tmp_path / "missing.json")])
    assert scheduler.jobs[0].state == "failed"
    assert "FileNotFoundError" in scheduler.jobs[0].error


def test_the_fault_names_are_importable_for_the_scenarios_that_need_them():
    """P4 distinguishes a transient start-up error from a lost job by type."""
    assert issubclass(ContainerNotAvailable, RuntimeError)
    assert issubclass(SchedulerFault, RuntimeError)
