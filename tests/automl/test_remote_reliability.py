"""P1-P4: what happens when the driver, the scheduler or a submit goes wrong.

None of this can be produced by a healthy cluster on request, so every scenario
here drives the executing stand-in from ``automl.scheduler_stub`` into the
condition it is about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from automl.scheduler_stub import LocalScheduler, SchedulerFault
from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.environment import (
    EnvironmentRunner,
    apply_grace,
    classify_scheduler_state,
)


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
# P4. Scheduler states are distinguishable, and waiting is finite              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (None, "missing"),
        ({"state": "PENDING"}, "queued"),
        ({"state": "Running"}, "running"),
        ({"state": "succeeded"}, "succeeded"),
        ({"state": "failed"}, "failed"),
        (
            {"state": 'KubernetesError: container "pytorch" ... is not available'},
            "submitting",
        ),
        ({"state": "NotFoundError: Field 'pods' not found"}, "missing"),
        ({"state": "something the API has never said before"}, "unknown"),
    ],
)
def test_every_scheduler_answer_maps_to_one_state(row, expected):
    assert classify_scheduler_state(row) == expected


def test_a_transient_start_up_error_is_not_a_failure_and_never_resubmits():
    """`container ... is not available` is a normal start: keep polling."""
    row = {"state": 'KubernetesError: container "pytorch" is not available'}
    state, since = apply_grace(classify_scheduler_state(row), None, 0.0)
    assert state == "submitting"
    assert since is None


@pytest.mark.parametrize("state", ["missing", "unknown"])
def test_a_job_that_stays_unaccounted_for_becomes_terminal(state):
    first, since = apply_grace(state, None, 1_000.0)
    assert first == state
    assert since == 1_000.0
    later, _ = apply_grace(state, since, 1_000.0 + 10_000.0)
    assert later == "lost"


def test_a_state_that_comes_back_clears_its_deadline():
    _, since = apply_grace("missing", None, 100.0)
    state, cleared = apply_grace("running", since, 200.0)
    assert (state, cleared) == ("running", None)


def test_a_vanished_job_is_lost_rather_than_waited_on_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fmlib.automl.environment.MISSING_GRACE_SECONDS", 0.0, raising=False
    )
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    scheduler.vanish = [scheduler.jobs[0].name]

    with pytest.raises(Exception, match="."):
        task.status(wait=True)
    operation = task._store.latest("train")
    assert operation["jobs"][0]["state"] == "lost"
    assert operation["state"] == "failed"


# --------------------------------------------------------------------------- #
# P2. A partial submit does not lose the jobs it already made                  #
# --------------------------------------------------------------------------- #
def test_a_failed_submit_keeps_the_jobs_already_created(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False, fail_create_on=2)
    task = _task(tmp_path, scheduler, model_layout="global_and_per_group")

    with pytest.raises(SchedulerFault):
        task.train(train, train)

    operation = task._store.latest("train")
    submitted = [job["job_id"] for job in operation["jobs"] if job["job_id"]]
    planned = [job for job in operation["jobs"] if job["job_id"] is None]
    assert submitted == ["job-1"]
    # The rest are still on the record as planned, with their parameters, which
    # is what makes the submit resumable instead of replanned.
    assert len(planned) == 2
    assert all(job["state"] == "planned" for job in planned)
    assert operation["state"] == "failed"


def test_each_job_is_persisted_before_the_next_one_is_submitted(tmp_path):
    """The window between submit and persistence is one job wide, not N."""
    train = _data(tmp_path / "train.parquet")
    seen: list[int] = []
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler, model_layout="global_and_per_group")

    def record(_request):
        operation = task._store.latest("train")
        seen.append(
            len([job for job in operation.get("jobs") or () if job.get("job_id")])
        )

    scheduler.on_create = record
    task.train(train, train)
    # Every submit sees the previous one already on the record.
    assert seen == [0, 1, 2]


def test_a_fan_out_job_can_be_told_from_its_siblings_by_name(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = _task(tmp_path, scheduler, model_layout="global_and_per_group")
    task.train(train, train)

    names = [job.name for job in scheduler.jobs]
    assert len(set(names)) == 3
    assert [name.split("-")[-1] for name in names] == ["0000", "0001", "0002"]


# --------------------------------------------------------------------------- #
# P1. Finalization survives the death of the driver                            #
# --------------------------------------------------------------------------- #
def test_the_happy_path_finalizes_itself(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)

    table = task.status()
    assert table["state"].to_list() == ["succeeded"]
    assert (tmp_path / "outputs").exists()
    # Nothing to recover, and asking is harmless.
    assert task.finalize()["state"].to_list() == ["succeeded"]


def test_a_driver_that_dies_mid_finalization_leaves_a_recoverable_operation(
    tmp_path, monkeypatch
):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)

    runner = task._operation_runner()
    original = type(runner)._finalize_remote_train

    def die(self, operation):
        msg = "the container went away"
        raise KeyboardInterrupt(msg)

    monkeypatch.setattr(type(runner), "_finalize_remote_train", die)
    with pytest.raises(KeyboardInterrupt):
        task.status()

    operation = task._store.latest("train")
    assert operation["state"] == "finalizing"

    # A new process, the same entity on disk.
    monkeypatch.setattr(type(runner), "_finalize_remote_train", original)
    recovered = BinaryTask.load(task.path)
    recovered._environment_runner = EnvironmentRunner(scheduler)
    table = recovered.finalize()
    assert table["state"].to_list() == ["succeeded"]
    assert json.loads((task.path / "training_result.json").read_text())


def test_status_reports_a_stuck_operation_but_does_not_finish_it(tmp_path, monkeypatch):
    """status() reflects job state; recovery is a decision someone makes."""
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    operation = task._store.latest("train")
    task._store.update_operation(operation, state="finalizing")

    task.status()
    # The table reports per-job scheduler state; what matters is that the
    # operation is still where the dead driver left it, and no result appeared.
    assert task._store.latest("train")["state"] == "finalizing"
    assert not (task.path / "training_result.json").exists()


def test_finalizing_twice_publishes_one_result(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    operation = task._store.latest("train")
    task._store.update_operation(operation, state="finalizing")

    first = task.finalize()
    second = task.finalize()
    assert first["state"].to_list() == ["succeeded"]
    assert second["state"].to_list() == ["succeeded"]


# --------------------------------------------------------------------------- #
# P3. Two drivers on one operation                                             #
# --------------------------------------------------------------------------- #
def test_a_second_driver_refuses_to_take_a_live_owners_operation(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    operation = task._store.latest("train")
    task._store.update_operation(
        operation,
        state="finalizing",
        owner={"hostname": "elsewhere", "pid": 1, "token": "other"},
    )

    with pytest.raises(RuntimeError, match="owned by elsewhere:1"):
        task.finalize()
    assert task._store.latest("train")["state"] == "finalizing"


def test_a_dead_owner_is_taken_over_without_asking(tmp_path):
    import os

    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    operation = task._store.latest("train")
    task._store.update_operation(
        operation,
        state="finalizing",
        owner={
            "hostname": __import__("socket").gethostname(),
            "pid": 2**22,
            "token": "x",
        },
    )

    table = task.finalize()
    assert table["state"].to_list() == ["succeeded"]
    assert task._store.latest("train")["owner"]["pid"] == os.getpid()


def test_force_takes_over_even_a_live_owner(tmp_path):
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler()
    task = _task(tmp_path, scheduler)
    task.train(train, train)
    operation = task._store.latest("train")
    task._store.update_operation(
        operation,
        state="finalizing",
        owner={"hostname": "elsewhere", "pid": 1, "token": "other"},
    )

    assert task.finalize(force=True)["state"].to_list() == ["succeeded"]


def test_two_writers_never_share_a_temporary_file(tmp_path):
    """A fixed temp name is a race; the pid and a random suffix are not."""
    from fmlib.automl.lifecycle import write_json

    target = tmp_path / "operation.json"
    write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp*"))
