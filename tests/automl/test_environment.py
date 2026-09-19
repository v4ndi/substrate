"""Execution-environment and persistent remote-operation tests."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from automl.osiris_contract import validate_create_request
from fmlib.automl import BinaryTask, BinaryTaskConfig, EnvironmentConfig
from fmlib.automl.environment import EnvironmentRunner
from fmlib.automl.exceptions import (
    ArtifactIntegrityError,
    ConfigError,
    RemoteExecutionError,
    SchemaError,
)


class _FakeOsiris:
    def __init__(self, states=None):
        self.create_calls = []
        self.states = list(states or [[]])
        self.list_calls = 0

    def create(self, **kwargs):
        # The recording client validates too: these tests are about pools,
        # profiles and job names, and the contract check rides along for free.
        validate_create_request(kwargs)
        self.create_calls.append(kwargs)
        return {"job_id": f"job-{len(self.create_calls)}"}

    def list(self):
        index = min(self.list_calls, len(self.states) - 1)
        self.list_calls += 1
        return {"jobs": self.states[index]}


def _remote_config(
    tmp_path, *, env_type="osiris", pool="common", num_gpus=None, num_nodes=None
):
    venv = tmp_path / "shared_venv"
    venv.mkdir(exist_ok=True)
    remote_venv = (
        f"/{venv.as_posix()}"
        if not venv.as_posix().startswith("/")
        else venv.as_posix()
    )
    return BinaryTaskConfig(
        env_type=env_type,
        backend="boosting",
        engine="catboost",
        device="gpu",
        target_column="target",
        client_id_column="epk_id",
        group_column="group",
        date_column="report_month",
        categorical_columns=(),
        numerical_columns=["feature"],
        hidden_state_columns=(),
        model_layout="global",
        hyperopt=False,
        output_dir=tmp_path / "outputs",
        environment=EnvironmentConfig(
            venv_path=remote_venv,
            log_dir=tmp_path / "logs",
            pool=pool,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
        ),
    )


def _train_data(path):
    pl.DataFrame({
        "epk_id": [1, 2, 3],
        "target": [0, 1, 0],
        "feature": [0.0, 1.0, 2.0],
        "group": ["a", "b", "a"],
        "report_month": ["2026-01-01"] * 3,
    }).write_parquet(path)


@pytest.mark.parametrize("pool", [None, "common"])
def test_standard_osiris_profile_uses_shared_venv_without_scheduler_pool(
    tmp_path, pool
):
    osiris = _FakeOsiris()
    runner = EnvironmentRunner(osiris)
    run_dir = tmp_path / "operation" / "job"
    job = runner.submit(
        config=_remote_config(tmp_path, pool=pool),
        action="train",
        payload={"train_path": "/data/train"},
        run_dir=run_dir,
    )
    request = osiris.create_calls[0]
    spec = json.loads((run_dir / "run_spec.json").read_text(encoding="utf-8"))
    assert request["command"][1:] == ["-m", "fmlib.automl.run"]
    assert request["command"][0].replace("\\", "/").endswith("/bin/python")
    assert "PYTHONPATH" not in request["envs"]
    assert request["num_nodes"] == 1 and request["num_gpus"] == 1
    assert "pool" not in request
    assert spec["config"]["env_type"] == "osiris" and spec["config"]["device"] == "gpu"
    assert spec["environment_profile"] == {
        "original_pool": pool,
        "effective_pool": None,
        "profile": "batch",
        "num_gpus": 1,
        "num_nodes": 1,
    }
    assert job["original_pool"] == pool and job["effective_pool"] is None
    assert job["resource_profile"] == "batch"
    assert job["job_id"] == "job-1"


def test_local_runner_writes_success_and_error_logs(tmp_path):
    config = BinaryTaskConfig(
        env_type="local",
        backend="boosting",
        engine="catboost",
        device="cpu",
        target_column="target",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=(),
        numerical_columns=("feature",),
        hidden_state_columns=(),
        hyperopt=False,
        output_dir=tmp_path / "outputs",
        environment=EnvironmentConfig(log_dir=tmp_path / "logs"),
    )
    runner = EnvironmentRunner(_FakeOsiris())
    assert runner.run_local(config=config, action="train", callback=lambda: 42) == 42
    train_log = next((tmp_path / "logs").glob("*.train.log")).read_text(
        encoding="utf-8"
    )
    assert (
        "Starting action=train" in train_log
        and "env_type=local device=cpu" in train_log
    )
    assert "Finished action=train duration_seconds=" in train_log

    def fail():
        msg = "diagnostic failure"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="diagnostic failure"):
        runner.run_local(config=config, action="predict", callback=fail)
    predict_log = next((tmp_path / "logs").glob("*.predict.log")).read_text(
        encoding="utf-8"
    )
    assert "diagnostic failure" in predict_log
    assert "AutoML action failed duration_seconds=" in predict_log


def _local_config(tmp_path, *, backend="boosting", engine="catboost", device="cpu"):
    return BinaryTaskConfig(
        env_type="local",
        backend=backend,
        engine=engine,
        device=device,
        target_column="target",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=(),
        numerical_columns=("feature",),
        hidden_state_columns=(),
        hyperopt=False,
        output_dir=tmp_path / "outputs",
        environment={},
    )


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_local_boosting_still_sees_only_cuda_device_zero(tmp_path, monkeypatch, device):
    """CatBoost with task_type='GPU' and no explicit devices spreads over all cards."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    observed = EnvironmentRunner(_FakeOsiris()).run_local(
        config=_local_config(tmp_path, device=device),
        action="train",
        callback=lambda: os.environ["CUDA_VISIBLE_DEVICES"],
    )
    assert observed == "0"


def test_local_run_restores_the_callers_cuda_visibility(tmp_path, monkeypatch):
    """One AutoML call must not narrow the notebook kernel for everything after it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    EnvironmentRunner(_FakeOsiris()).run_local(
        config=_local_config(tmp_path),
        action="train",
        callback=lambda: None,
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_local_run_restores_cuda_visibility_after_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    def fail():
        msg = "diagnostic failure"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="diagnostic failure"):
        EnvironmentRunner(_FakeOsiris()).run_local(
            config=_local_config(tmp_path), action="train", callback=fail
        )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_local_run_leaves_cuda_unset_when_the_caller_had_it_unset(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    EnvironmentRunner(_FakeOsiris()).run_local(
        config=_local_config(tmp_path),
        action="train",
        callback=lambda: None,
    )
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_local_tabnn_keeps_every_card_the_caller_exposed(tmp_path, monkeypatch, device):
    """num_gpus decides how many cards a trial uses; a pin would hide the rest."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    config = _local_config(
        tmp_path, backend="tabnn", engine="tabular_transformer", device=device
    )
    observed = EnvironmentRunner(_FakeOsiris()).run_local(
        config=config,
        action="train",
        callback=lambda: os.environ["CUDA_VISIBLE_DEVICES"],
    )
    assert observed == "2,3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


@pytest.mark.parametrize("action", ["train", "predict", "calibrate", "evaluate"])
@pytest.mark.parametrize("pool", ["b2c", "custom-pool"])
def test_custom_pool_uses_explicit_resources_and_exposes_device_zero(
    tmp_path, action, pool
):
    osiris = _FakeOsiris()
    config = _remote_config(tmp_path, pool=pool, num_nodes=2, num_gpus=3)
    EnvironmentRunner(osiris).submit(
        config=config, action=action, payload={}, run_dir=tmp_path / action
    )
    request = osiris.create_calls[0]
    assert request["pool"] == pool
    spec = json.loads((tmp_path / action / "run_spec.json").read_text(encoding="utf-8"))
    assert spec["config"]["env_type"] == "osiris"
    assert BinaryTaskConfig.from_mapping(spec["config"]).environment.pool == pool
    assert spec["requested_config"]["environment"]["pool"] == pool
    assert spec["environment_profile"]["profile"] == "supercomp"
    assert request["num_nodes"] == 2 and request["num_gpus"] == 3
    assert request["envs"]["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pool": "custom", "num_nodes": None, "num_gpus": 2}, "requires explicit"),
        ({"pool": "custom", "num_nodes": 1, "num_gpus": None}, "requires explicit"),
        ({"pool": "common", "num_nodes": 2}, "requires num_nodes=1 and num_gpus=1"),
        ({"pool": None, "num_gpus": 2}, "requires num_nodes=1 and num_gpus=1"),
    ],
)
def test_osiris_resource_profiles_reject_missing_or_conflicting_values(
    tmp_path, kwargs, message
):
    with pytest.raises(ConfigError, match=message):
        _remote_config(tmp_path, **kwargs)


def test_remote_submit_requires_exact_job_id(tmp_path, monkeypatch):
    runner = EnvironmentRunner(_FakeOsiris())
    monkeypatch.setattr(
        runner.osiris, "create", lambda **_kwargs: {"message": "created"}
    )
    with pytest.raises(RemoteExecutionError, match="exact job ID"):
        runner.submit(
            config=_remote_config(tmp_path),
            action="train",
            payload={},
            run_dir=tmp_path / "missing-id",
        )


def test_remote_train_is_owned_by_entity_and_returns_no_handle(tmp_path):
    train = tmp_path / "train.parquet"
    _train_data(train)
    task = BinaryTask(
        replace(_remote_config(tmp_path), model_layout="global_and_per_group")
    )
    osiris = _FakeOsiris()
    task._environment_runner = EnvironmentRunner(osiris)
    assert task.train(train, train) is None
    operation = task._store.latest("train")
    assert operation["state"] == "queued"
    assert operation["runtime"] == {
        "env_type": "osiris",
        "device": "gpu",
        "original_pool": "common",
        "effective_pool": None,
        "resource_profile": "batch",
        "num_gpus": 1,
        "num_nodes": 1,
    }
    assert len(operation["jobs"]) == 3
    assert [job["job_id"] for job in operation["jobs"]] == ["job-1", "job-2", "job-3"]
    assert not hasattr(task, "check_train_finish")


def test_status_uses_exact_job_ids_and_ignores_unrelated_jobs(tmp_path):
    train = tmp_path / "train.parquet"
    _train_data(train)
    osiris = _FakeOsiris(
        states=[
            [
                {"job_id": "job-1", "state": "running"},
                {"job_id": "unrelated", "state": "failed"},
            ]
        ]
    )
    task = BinaryTask(_remote_config(tmp_path))
    task._environment_runner = EnvironmentRunner(osiris)
    task.train(train, train)
    table = task.status()
    assert table["job_id"].to_list() == ["job-1"]
    assert table["state"].to_list() == ["running"]
    assert osiris.list_calls == 1


def test_entity_load_recovers_dead_local_operation(tmp_path, monkeypatch):
    task = BinaryTask(_remote_config(tmp_path))
    first = tmp_path / "first-test"
    operation = task._store.start_operation(
        "predict", dataset_path=first, runtime={"env_type": "local"}
    )
    task._store.update_operation(operation, state="running")
    monkeypatch.setattr(
        "fmlib.automl.lifecycle.AutoMLStore._local_owner_is_alive",
        staticmethod(lambda _owner: False),
    )

    restored = BinaryTask.load(task.path)
    assert (
        restored.status()
        .filter(pl.col("test_path") == str(first.resolve()))
        .item(0, "state")
        == "interrupted"
    )


def test_waiting_prediction_status_publishes_scores_for_the_exact_test_path(
    tmp_path, monkeypatch
):
    config = BinaryTaskConfig(
        env_type="local",
        backend="boosting",
        engine="catboost",
        device="cpu",
        target_column="target",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=(),
        numerical_columns=("feature",),
        hidden_state_columns=(),
        hyperopt=False,
        output_dir=tmp_path / "outputs",
        environment=EnvironmentConfig(
            venv_path="/home/datalab/nfs/sber-amazme-fmlib/env",
            poll_interval_seconds=0.01,
        ),
    )
    osiris = _FakeOsiris(
        states=[
            [{"job_id": "job-1", "state": "running"}],
            [{"job_id": "job-1", "state": "succeeded"}],
        ]
    )
    task = BinaryTask(config)
    task._environment_runner = EnvironmentRunner(osiris)
    monkeypatch.setattr(task, "_require_fitted", lambda: None)
    monkeypatch.setattr(task, "_validate_prediction_layout", lambda: None)
    monkeypatch.setattr(task, "save", lambda *_args, **_kwargs: tmp_path / "artifact")
    monkeypatch.setattr(
        EnvironmentRunner,
        "_remote_python",
        staticmethod(
            lambda _config: "/home/datalab/nfs/sber-amazme-fmlib/env/bin/python"
        ),
    )
    monkeypatch.setattr(
        "fmlib.automl.tasks.operations.time.sleep", lambda _seconds: None
    )
    test_path = tmp_path / "data" / "test"

    assert task.predict(test_path, env_type="osiris") is None
    operation = task._store.latest("predict", test_path)
    job = operation["jobs"][0]
    scores_path = Path(job["result_path"]).parent / "worker-scores.parquet"
    pl.DataFrame({
        "epk_id": [1, 2],
        "score": [0.2, 0.8],
        "__fmlib_remote_row_id": [0, 1],
    }).write_parquet(scores_path)
    Path(job["result_path"]).write_text(
        json.dumps({
            "status": "succeeded",
            "scores_path": str(scores_path),
            "class_order": None,
        }),
        encoding="utf-8",
    )

    table = task.status(wait=True)

    assert table["state"].to_list() == ["succeeded"]
    assert task.load_prediction(test_path).scores["score"].to_list() == [0.2, 0.8]
    assert "__fmlib_remote_row_id" not in task.load_prediction(test_path).scores.columns
    with pytest.raises(ArtifactIntegrityError, match="No operation is registered"):
        task.load_prediction(tmp_path / "another" / "test")
    assert osiris.list_calls == 2


def test_remote_prediction_parts_reject_unknown_channels_before_submission(tmp_path):
    test_path = tmp_path / "test.parquet"
    pl.DataFrame({
        "group": ["b", "unknown", "a"],
        "feature": [1.0, 2.0, 3.0],
    }).write_parquet(test_path)
    task = BinaryTask(
        replace(_remote_config(tmp_path), model_layout="global_and_per_group")
    )
    task._models = [
        SimpleNamespace(
            layout="global",
            group_value=None,
            single_group_global=False,
            single_group_value=None,
        ),
        SimpleNamespace(
            layout="per_group",
            group_value="a",
            single_group_global=False,
            single_group_value=None,
        ),
        SimpleNamespace(
            layout="per_group",
            group_value="b",
            single_group_global=False,
            single_group_value=None,
        ),
    ]

    with pytest.raises(SchemaError, match=r"Unknown group values.*'unknown'.*1 rows"):
        task._remote_prediction_parts(test_path)


def test_remote_group_prediction_filters_rows_without_losing_source_order(tmp_path):
    task = BinaryTask(replace(_remote_config(tmp_path), model_layout="per_group"))
    frame = pl.DataFrame({
        "epk_id": [1, 2, 3, 4],
        "group": ["b", "a", "b", "a"],
        "feature": [0.1, 0.2, 0.3, 0.4],
    })

    selected = task._prediction_input_frame(
        frame, remote_group_value="b", include_row_id=True
    )

    assert selected["epk_id"].to_list() == [1, 3]
    assert selected["__fmlib_remote_row_id"].to_list() == [0, 2]
    assert selected["__row_id"].to_list() == [0, 1]


def test_remote_both_prediction_runs_and_merges_one_job_per_model_part(
    tmp_path, monkeypatch
):
    osiris = _FakeOsiris(
        states=[
            [
                {"job_id": "job-1", "state": "succeeded"},
                {"job_id": "job-2", "state": "succeeded"},
                {"job_id": "job-3", "state": "succeeded"},
            ]
        ]
    )
    task = BinaryTask(
        replace(_remote_config(tmp_path), model_layout="global_and_per_group")
    )
    task._environment_runner = EnvironmentRunner(osiris)
    monkeypatch.setattr(task, "_require_fitted", lambda: None)
    monkeypatch.setattr(task, "_validate_prediction_layout", lambda: None)
    monkeypatch.setattr(task, "save", lambda *_args, **_kwargs: tmp_path / "artifact")
    monkeypatch.setattr(
        task,
        "_remote_prediction_parts",
        lambda _path: [("global", None), ("per_group", "a"), ("per_group", "b")],
    )
    test_path = tmp_path / "test"

    assert task.predict(test_path) is None
    operation = task._store.latest("predict", test_path)
    assert [job["job_id"] for job in operation["jobs"]] == ["job-1", "job-2", "job-3"]

    payloads = [
        json.loads(Path(job["spec_path"]).read_text(encoding="utf-8"))["payload"]
        for job in operation["jobs"]
    ]
    assert [
        (item["remote_layout"], item["remote_group_value"]) for item in payloads
    ] == [
        ("global", None),
        ("per_group", "a"),
        ("per_group", "b"),
    ]
    frames = [
        pl.DataFrame({
            "epk_id": [2, 1],
            "group": ["b", "a"],
            "score": [0.8, 0.2],
            "__fmlib_remote_row_id": [1, 0],
        }),
        pl.DataFrame({
            "epk_id": [1],
            "group": ["a"],
            "score": [0.3],
            "__fmlib_remote_row_id": [0],
        }),
        pl.DataFrame({
            "epk_id": [2],
            "group": ["b"],
            "score": [0.7],
            "__fmlib_remote_row_id": [1],
        }),
    ]
    for job, frame in zip(operation["jobs"], frames, strict=True):
        scores_path = Path(job["result_path"]).parent / "worker-scores.parquet"
        frame.write_parquet(scores_path)
        Path(job["result_path"]).write_text(
            json.dumps({
                "status": "succeeded",
                "scores_path": str(scores_path),
                "class_order": None,
            }),
            encoding="utf-8",
        )

    assert task.status()["state"].to_list() == ["succeeded", "succeeded", "succeeded"]
    scores = task.load_prediction(test_path).scores
    assert scores["epk_id"].to_list() == [1, 2, 1, 2]
    assert scores["model_layout"].to_list() == [
        "global",
        "global",
        "per_group",
        "per_group",
    ]
    assert "__fmlib_remote_row_id" not in scores.columns


@pytest.mark.parametrize(
    ("pool", "num_nodes", "num_gpus", "expected_pool"),
    [("common", None, None, None), ("prediction-pool", 2, 3, "prediction-pool")],
)
def test_remote_prediction_uses_operation_environment_override(
    tmp_path, monkeypatch, pool, num_nodes, num_gpus, expected_pool
):
    task = BinaryTask(replace(_remote_config(tmp_path, env_type="local"), device="cpu"))
    osiris = _FakeOsiris()
    task._environment_runner = EnvironmentRunner(osiris)
    monkeypatch.setattr(task, "_require_fitted", lambda: None)
    monkeypatch.setattr(task, "_validate_prediction_layout", lambda: None)
    monkeypatch.setattr(task, "save", lambda *_args, **_kwargs: tmp_path / "artifact")
    override_venv = tmp_path / "prediction_venv"
    override_venv.mkdir()
    environment = EnvironmentConfig(
        venv_path=override_venv,
        image="prediction-image",
        pool=pool,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
    )

    assert (
        task.predict(tmp_path / "test", env_type="osiris", environment=environment)
        is None
    )

    request = osiris.create_calls[0]
    assert request["command"][0] == str(override_venv / "bin" / "python")
    assert request["image"] == "prediction-image"
    assert request["num_nodes"] == (num_nodes or 1)
    assert request["num_gpus"] == (num_gpus or 1)
    assert request.get("pool") == expected_pool
    assert task.config.environment.pool == "common"
    assert task.config.environment != environment


def test_remote_prediction_accepts_mapping_resource_override(tmp_path, monkeypatch):
    task = BinaryTask(replace(_remote_config(tmp_path, env_type="local"), device="cpu"))
    osiris = _FakeOsiris()
    task._environment_runner = EnvironmentRunner(osiris)
    monkeypatch.setattr(task, "_require_fitted", lambda: None)
    monkeypatch.setattr(task, "_validate_prediction_layout", lambda: None)
    monkeypatch.setattr(task, "save", lambda *_args, **_kwargs: tmp_path / "artifact")
    override_venv = tmp_path / "mapping_venv"
    override_venv.mkdir()

    assert (
        task.predict(
            tmp_path / "test",
            env_type="osiris",
            environment={
                "venv_path": override_venv,
                "pool": "research",
                "num_nodes": 2,
                "num_gpus": 4,
            },
        )
        is None
    )

    request = osiris.create_calls[0]
    assert request["pool"] == "research"
    assert (request["num_nodes"], request["num_gpus"]) == (2, 4)
    assert request["envs"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_two_equal_configs_create_independent_entities(tmp_path):
    config = _remote_config(tmp_path)
    first, second = BinaryTask(config), BinaryTask(config)
    assert first.id != second.id
    assert first.path != second.path
