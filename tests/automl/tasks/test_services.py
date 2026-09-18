"""Service boundaries without a task facade or real native estimators."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Event
from threading import enumerate as enumerate_threads
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.data import FeatureSchema, ParquetSource
from fmlib.automl.exceptions import (
    ArtifactIntegrityError,
    RemoteExecutionError,
    SchemaError,
)
from fmlib.automl.execution import ExecutionContext
from fmlib.automl.lifecycle import AutoMLStore, write_json
from fmlib.automl.tasks.artifacts import ArtifactRepository, ArtifactState
from fmlib.automl.tasks.operations import OperationHooks, OperationRunner
from fmlib.automl.tasks.planning import FrameGroupView, ModelPlan
from fmlib.automl.tasks.preparation import DataPreparation
from fmlib.automl.tasks.routing import PredictionRouter
from fmlib.automl.tasks.state import ModelEntry
from fmlib.automl.tasks.training import TrainingCoordinator
from fmlib.automl.types import PredictionResult, TrainingResult


@pytest.fixture
def context(tmp_path):
    return ExecutionContext.from_config(
        BinaryTaskConfig(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            target_column="target",
            client_id_column="client",
            date_column="month",
            group_column="channel",
            categorical_columns=(),
            numerical_columns=("x",),
            hidden_state_columns=(),
            model_layout="global_and_per_group",
            hyperopt=False,
            output_dir=tmp_path,
            verbose=False,
        )
    )


class NativeStub:
    def save(self, path):
        path.mkdir(parents=True)
        (path / "model").write_text("native state")
        (path / "backend.json").write_text(json.dumps({"engine": "catboost"}))

    @classmethod
    def load(cls, path, *, device):
        assert (path / "model").read_text() == "native state"
        assert device == "cpu"
        return cls()


def _view(frame):
    """Planning takes a group view; in memory that is a frame."""
    return FrameGroupView(frame)


def model(layout="global", group=None):
    schema = FeatureSchema(
        (), ("x",), ("x",), {"x": "Float64"}, "target", "epk_id", None, "group"
    )
    return ModelEntry(NativeStub(), schema, {}, {}, 0.75, layout, group)


def test_preparation_keeps_hive_aliases_and_source_identity(context, tmp_path):
    path = tmp_path / "channel=a" / "split=train" / "part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({
        "client": [2, 1],
        "month": ["2026-01", "2026-02"],
        "x": [1.0, 2.0],
    }).write_parquet(path)
    preparation = DataPreparation(context)
    frame = preparation.read(path.parent)
    assert frame["epk_id"].to_list() == [2, 1]
    assert frame["group"].to_list() == ["a", "a"]
    assert frame["date"].dtype == pl.Date
    manifest = preparation.source_manifest(ParquetSource.resolve(path.parent))
    assert manifest == (
        {
            "path": str(path),
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        },
    )


def test_preparation_projects_features_but_still_rejects_alias_collisions(
    context, tmp_path
):
    path = tmp_path / "data.parquet"
    frame = pl.DataFrame({
        "client": [1],
        "month": ["2026-01"],
        "x": [None],
        "unused": ["payload"],
    })
    frame.write_parquet(path)
    prepared = DataPreparation(context).read(path)
    assert set(prepared.columns) == {"epk_id", "date", "x"}
    assert prepared["x"].null_count() == 1
    frame.with_columns(pl.lit(9).alias("epk_id")).write_parquet(path)
    with pytest.raises(SchemaError, match="collide"):
        DataPreparation(context).read(path)


@pytest.mark.parametrize("layout", ["global", "per_group", "global_and_per_group"])
def test_training_partitions_once_and_preserves_slice_order(
    context, tmp_path, monkeypatch, layout
):
    context = context.derive(model_layout=layout)
    path = tmp_path / "data.parquet"
    pl.DataFrame({
        "client": [3, 1, 2],
        "month": ["2026-01"] * 3,
        "channel": ["b", "a", "b"],
        "x": [3, 1, 2],
    }).write_parquet(path)
    calls = []
    partitions = []
    original = pl.DataFrame.partition_by

    def partition(frame, *args, **kwargs):
        partitions.append(frame.height)
        return original(frame, *args, **kwargs)

    def fit(train, valid, *, layout, group_value):
        calls.append((layout, group_value, train.require_frame()["epk_id"].to_list()))
        assert train.require_frame().equals(valid.require_frame())
        assert train.group_value == group_value
        return model(layout, group_value)

    monkeypatch.setattr(pl.DataFrame, "partition_by", partition)
    TrainingCoordinator(context, "binary", lambda *_: None, fit).execute(path, path)
    expected = (
        [("global", None, [3, 1, 2])]
        if layout in {"global", "global_and_per_group"}
        else []
    )
    if layout in {"per_group", "global_and_per_group"}:
        expected += [("per_group", "a", [1]), ("per_group", "b", [3, 2])]
    assert calls == expected
    assert partitions == ([] if layout == "global" else [3, 3])


def test_training_logs_single_group_combined_layout_resolution(
    context, tmp_path, caplog
):
    path = tmp_path / "data.parquet"
    pl.DataFrame({
        "client": [1, 2],
        "month": ["2026-01"] * 2,
        "channel": ["only", "only"],
        "x": [1, 2],
    }).write_parquet(path)

    with caplog.at_level("INFO", logger="fmlib.automl.progress"):
        outcome = TrainingCoordinator(
            context, "binary", lambda *_: None, lambda *_args, **_kwargs: model()
        ).execute(path, path)

    assert outcome.models[0].single_group_global is True
    assert "Requested model_layout='global_and_per_group'" in caplog.text
    assert "single group value 'only'" in caplog.text
    assert (
        "effective model_layout='global'; per-group branch is not created"
        in caplog.text
    )


def test_remote_training_plan_reads_only_group_column(context, tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"channel": ["b", "a"], "unused": [1, 2]}).write_parquet(path)
    task = BinaryTask(context.config)

    def no_full_read(*args, **kwargs):
        pytest.fail("Planning must not load feature frames")

    monkeypatch.setattr(ParquetSource, "read", no_full_read)
    assert task._remote_training_parts(path) == [
        ("global", None),
        ("per_group", "a"),
        ("per_group", "b"),
    ]


@pytest.mark.parametrize("layout", ["global", "per_group", "global_and_per_group"])
def test_one_training_group_resolves_both_to_global_in_local_and_remote_plans(
    context, layout
):
    context = context.derive(model_layout=layout)
    train = pl.DataFrame({"group": ["a", "a"]})
    valid = pl.DataFrame({"group": ["a"]})
    plan = ModelPlan.training(context, _view(train), _view(valid))
    assert plan == ModelPlan.remote_training(context, train)
    assert plan.parts == (
        (("per_group", "a"),) if layout == "per_group" else (("global", None),)
    )
    assert plan.layout == ("per_group" if layout == "per_group" else "global")


@pytest.mark.parametrize("layout", ["global", "per_group", "global_and_per_group"])
def test_model_plan_preserves_local_remote_order_and_validates_slices(context, layout):
    context = context.derive(model_layout=layout)
    train = pl.DataFrame({"group": ["b", "a", "b"]})
    valid = pl.DataFrame({"group": ["a", "b"]})
    plan = ModelPlan.training(context, _view(train), _view(valid))
    assert plan == ModelPlan.remote_training(context, train)
    expected = (
        (("global", None),)
        if layout == "global"
        else (("per_group", "a"), ("per_group", "b"))
    )
    if layout == "global_and_per_group":
        expected = (("global", None), *expected)
    assert plan.parts == expected
    with pytest.raises(FrozenInstanceError):
        plan.layout = "global"
    if layout != "global":
        with pytest.raises(SchemaError, match="Validation data has no rows"):
            ModelPlan.training(context, _view(train), _view(valid.head(1)))
        with pytest.raises(SchemaError, match="channel"):
            ModelPlan.training(context, _view(train), _view(valid.drop("group")))
        part = ModelPlan.training(
            context,
            _view(train),
            _view(valid),
            remote_layout="per_group",
            remote_group_value="b",
        )
        assert part.parts == (("per_group", "b"),)
        with pytest.raises(SchemaError, match="Training data has no rows"):
            ModelPlan.training(
                context,
                _view(train),
                _view(valid),
                remote_layout="per_group",
                remote_group_value="absent",
            )


def test_router_preserves_independent_branches_and_rejects_unknown_groups(context):
    models = (model(), model("per_group", "a"), model("per_group", "b"))
    frame = pl.DataFrame({"group": ["b", "a", "b"], "x": [1, 3, 4]}).with_row_index(
        "__row_id"
    )
    router = PredictionRouter(context, models)
    branches = dict(router.layout_frames(frame))
    assert branches["global"] is frame
    assert branches["per_group"]["x"].to_list() == [1, 3, 4]
    grouped = PredictionRouter(context.derive(model_layout="per_group"), models)
    raw = grouped.score(
        branches["per_group"],
        lambda part, item: part["x"].to_numpy()
        + (10 if item.group_value == "a" else 20),
    )
    np.testing.assert_array_equal(raw, [21, 13, 24])
    unknown = pl.DataFrame({"group": ["b", "unknown", "unknown"], "x": [1, 2, 3]})
    with pytest.raises(SchemaError, match=r"Unknown group values.*'unknown'.*2 rows"):
        router.layout_frames(unknown)
    with pytest.raises(SchemaError, match=r"Unknown group values.*'unknown'.*2 rows"):
        grouped.layout_frames(unknown)
    combined = router.combine([
        ("global", PredictionResult(pl.DataFrame({"score": [0.1, 0.2]}))),
        (
            "per_group",
            PredictionResult(pl.DataFrame({"score": [0.3]})),
        ),
    ])
    assert combined.scores["model_layout"].to_list() == [
        "global",
        "global",
        "per_group",
    ]
    assert frame["group"].to_list() == ["b", "a", "b"]


@pytest.mark.parametrize("failure", ["native_save", "atomic_replace"])
def test_artifact_repository_round_trip_and_failed_overwrite(
    context, tmp_path, monkeypatch, failure
):
    repository = ArtifactRepository("binary", "binary", lambda _family: NativeStub)
    metadata = {"labels": ["a", "b"]}
    state = ArtifactState(
        context.config, (model(),), {"train": ({"path": "train"},)}, metadata
    )
    metadata["labels"].append("c")
    assert state.task_manifest["labels"] == ("a", "b")
    path = repository.save(state, tmp_path / "snapshot")
    restored = repository.restore(path, context.config)
    assert restored.models[0].schema == state.models[0].schema
    assert restored.source_manifests == state.source_manifests
    assert restored.task_manifest["labels"] == ("a", "b")
    with pytest.raises(FrozenInstanceError):
        state.models[0].layout = "per_group"

    def fail(*args, **kwargs):
        msg = "injected artifact failure"
        raise OSError(msg)

    if failure == "native_save":
        monkeypatch.setattr(NativeStub, "save", fail)
    else:
        import fmlib.automl.tasks.artifacts as artifacts

        original = artifacts.os.replace

        def fail_install(source, destination):
            if Path(source).name.endswith(".tmp"):
                fail()
            return original(source, destination)

        monkeypatch.setattr(artifacts.os, "replace", fail_install)
    with pytest.raises(OSError, match="injected artifact failure"):
        repository.save(state, path, overwrite=True)
    assert repository.restore(path, context.config).models[0].validation_metric == 0.75
    assert not list(tmp_path.glob(".snapshot.*"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(task="regression"), "task mismatch"),
        (
            lambda manifest: manifest.update(backend_family="tabnn"),
            "backend_family mismatch",
        ),
        (lambda manifest: manifest.update(engine="xgboost"), "engine mismatch"),
        (
            lambda manifest: manifest["models"][0].update(scope="product"),
            "legacy model scope metadata",
        ),
        (
            lambda manifest: manifest["models"][0].update(
                model_layout="per_group", group_value=None
            ),
            "inconsistent model_layout",
        ),
        (
            lambda manifest: manifest["models"][0]["schema"].update(
                feature_order=["missing"]
            ),
            "schema is inconsistent",
        ),
    ],
)
def test_artifact_repository_prevalidates_complete_manifest(
    context, tmp_path, monkeypatch, mutate, message
):
    repository = ArtifactRepository("binary", "binary", lambda _family: NativeStub)
    path = repository.save(
        ArtifactState(context.config, (model(), model("per_group", "a")), {}, {}),
        tmp_path / "snapshot",
    )
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))
    loaded = False

    def fail_if_loaded(*_args, **_kwargs):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(NativeStub, "load", fail_if_loaded)
    with pytest.raises(ArtifactIntegrityError, match=message):
        repository.restore(path, context.config)
    assert not loaded


def test_artifact_repository_rejects_backend_engine_before_native_restore(
    context, tmp_path, monkeypatch
):
    repository = ArtifactRepository("binary", "binary", lambda _family: NativeStub)
    path = repository.save(
        ArtifactState(context.config, (model(), model("per_group", "a")), {}, {}),
        tmp_path / "snapshot",
    )
    backend_path = path / "model" / "1" / "backend.json"
    backend_path.write_text(json.dumps({"engine": "xgboost"}))
    loaded = False

    def fail_if_loaded(*_args, **_kwargs):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(NativeStub, "load", fail_if_loaded)
    with pytest.raises(ArtifactIntegrityError, match="backend engine mismatch"):
        repository.restore(path, context.config)
    assert not loaded


def test_failed_standalone_load_does_not_leave_entity(context, tmp_path, monkeypatch):
    monkeypatch.setattr(BinaryTask, "_backend_classes", {"boosting": NativeStub})
    task = BinaryTask(context.config)
    task._models = [model(), model("per_group", "a")]
    artifact = task.save(tmp_path / "standalone")
    (artifact / "model" / "1" / "model").unlink()
    entities = set((tmp_path / "automl").iterdir())

    with pytest.raises(ArtifactIntegrityError, match="native model #1"):
        BinaryTask.load(artifact)

    assert set((tmp_path / "automl").iterdir()) == entities


def test_standalone_load_reads_its_config_once(context, tmp_path, monkeypatch):
    monkeypatch.setattr(BinaryTask, "_backend_classes", {"boosting": NativeStub})
    task = BinaryTask(context.config)
    task._models = [model(), model("per_group", "a")]
    artifact = task.save(tmp_path / "standalone")
    original = ArtifactRepository.read_config
    calls = 0

    def counted(root, config_class):
        nonlocal calls
        calls += 1
        if calls > 1:
            msg = "standalone config was read more than once"
            raise AssertionError(msg)
        return original(root, config_class)

    monkeypatch.setattr(ArtifactRepository, "read_config", staticmethod(counted))

    restored = BinaryTask.load(artifact)

    assert restored.is_fitted
    assert calls == 1


def test_entity_load_compares_entity_and_artifact_configs(
    context, tmp_path, monkeypatch
):
    monkeypatch.setattr(BinaryTask, "_backend_classes", {"boosting": NativeStub})
    task = BinaryTask(context.config)
    task._models = [model(), model("per_group", "a")]
    artifact = task.save(task.path / "artifact")
    task._set_entity_artifact(artifact)
    config_path = artifact / "config.json"
    artifact_config = json.loads(config_path.read_text())
    artifact_config["target_column"] = "different_target"
    config_path.write_text(json.dumps(artifact_config))

    with pytest.raises(ArtifactIntegrityError, match=r"fields: \['target_column'\]"):
        BinaryTask.load(task.path)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "env_type": "osiris",
            "device": "gpu",
            "environment": {"venv_path": "/different/venv"},
        },
        {"output_dir": "different-output"},
        {"verbose": True},
        {"model_params": {"iterations": 17}},
        {"random_state": 7},
    ],
)
def test_entity_load_allows_non_model_artifact_config_to_differ(
    context, tmp_path, monkeypatch, updates
):
    monkeypatch.setattr(BinaryTask, "_backend_classes", {"boosting": NativeStub})
    task = BinaryTask(context.config)
    task._models = [model(), model("per_group", "a")]
    artifact = task.save(task.path / "artifact")
    task._set_entity_artifact(artifact)
    config_path = artifact / "config.json"
    artifact_config = json.loads(config_path.read_text())
    artifact_config.update(updates)
    config_path.write_text(json.dumps(artifact_config))

    restored = BinaryTask.load(task.path)

    assert restored.id == task.id
    assert restored.config.env_type == "local"
    assert restored.config.device == "cpu"
    assert restored._models[0].backend is not None


def test_internal_artifact_restore_separates_runtime_device_and_scope(
    context, tmp_path, monkeypatch
):
    loaded_devices = []

    class DeviceStub(NativeStub):
        @classmethod
        def load(cls, path, *, device):
            assert (path / "model").read_text() == "native state"
            loaded_devices.append(device)
            return cls()

    monkeypatch.setattr(BinaryTask, "_backend_classes", {"boosting": DeviceStub})
    source = BinaryTask(context.config)
    source._models = [model(), model("per_group", "a")]
    artifact = source.save(tmp_path / "standalone")
    runtime_config = replace(
        context.config, env_type="osiris", device="gpu", model_layout="global"
    )
    worker = BinaryTask(runtime_config, _entity_path=tmp_path / "worker")

    worker._restore_artifact(artifact)

    assert worker.is_fitted
    assert loaded_devices == ["gpu", "gpu"]


def test_entity_load_uses_same_artifact_prevalidation_and_preserves_identity(
    context, tmp_path, monkeypatch
):
    monkeypatch.setattr(BinaryTask, "_backend_classes", {"boosting": NativeStub})
    task = BinaryTask(context.config)
    task._models = [model(), model("per_group", "a")]
    artifact = task.save(task.path / "artifact")
    task._set_entity_artifact(artifact)
    identity = task.id

    restored = BinaryTask.load(task.path)

    assert restored.id == identity
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"][0]["schema"]["dtypes"] = {}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactIntegrityError, match="schema is inconsistent"):
        BinaryTask.load(task.path)
    assert [path.name for path in (tmp_path / "automl").iterdir()] == [identity]


@pytest.mark.parametrize("failure", ["execute", "save", "persist"])
def test_operation_failure_rolls_back_and_stops_heartbeat(
    context, tmp_path, monkeypatch, failure
):
    store = AutoMLStore.create(tmp_path, "binary", {})
    heartbeat = Event()
    events = []
    result = TrainingResult({}, {}, (), "boosting", "catboost", "binary")

    def step(name):
        events.append(name)
        if name == failure:
            msg = "injected operation failure"
            raise RuntimeError(msg)

    def execute(*args):
        assert heartbeat.wait(2), "Local heartbeat was not started"
        step("execute")
        return result

    monkeypatch.setattr(
        "fmlib.automl.tasks.operations._LOCAL_HEARTBEAT_INTERVAL_SECONDS", 0.001
    )
    monkeypatch.setattr(store, "heartbeat_operation", lambda _: heartbeat.set())
    monkeypatch.setattr(store, "persist_training", lambda _: step("persist"))
    environment = SimpleNamespace(run_local=lambda **kwargs: kwargs["callback"]())
    execution = SimpleNamespace(
        config=context.config, _runner=lambda: environment, _execute_train=execute
    )
    hooks = OperationHooks(
        resolve_config=lambda **kwargs: context.config,
        execution_view=lambda *args, **kwargs: execution,
        save=lambda *args, **kwargs: step("save"),
        adopt=lambda _: step("adopt"),
        rollback_training=lambda _: step("rollback"),
        set_artifact=lambda _: step("artifact"),
        assemble_training=lambda _: (),
        require_fitted=lambda: None,
        storage_frame=lambda result: result.scores,
    )
    runner = OperationRunner(context.config, store, environment, hooks, False)
    with pytest.raises(RuntimeError, match="injected operation failure"):
        runner.train("train", "valid")
    operation = store.latest("train")
    assert operation["state"] == "failed"
    assert events[-1] == "rollback" and "artifact" not in events
    assert not any(
        thread.name == f"automl-heartbeat-{operation['run_id'][:8]}"
        for thread in enumerate_threads()
    )


@pytest.mark.parametrize("fail_assembly", [False, True])
def test_remote_training_finalization_commits_only_complete_artifacts(
    context, tmp_path, monkeypatch, fail_assembly
):
    monkeypatch.setattr(BinaryTask, "_backend_classes", {"boosting": NativeStub})
    task = BinaryTask(context.config)
    identity = task.id
    repository = task._artifact_repository()
    jobs = []
    for index, entry in enumerate((model(), model("per_group", "a"))):
        root = repository.save(
            ArtifactState(context.config, (entry,), {}, {}), tmp_path / f"part-{index}"
        )
        spec, result = root / "spec.json", root / "result.json"
        write_json(spec, {"payload": {"artifact_path": str(root)}})
        name = "global" if index == 0 else "per_group:a"
        write_json(
            result, {"best_params": {name: {}}, "validation_metrics": {name: 0.75}}
        )
        jobs.append({
            "spec_path": str(spec),
            "result_path": str(result),
            "state": "succeeded",
            "job_id": str(index),
        })
    operation = task._store.start_operation(
        "train", runtime={"env_type": "osiris", "device": "gpu"}
    )
    artifact = task.path / "artifact"
    task._store.update_operation(
        operation, state="queued", jobs=jobs, artifact_path=str(artifact)
    )
    task._environment_runner = SimpleNamespace(
        poll_jobs=lambda jobs: ("succeeded", jobs),
        failure_logs=lambda jobs: "injected finalization failure",
    )
    if fail_assembly:
        original = NativeStub.load

        def fail_second(path, *, device):
            if "part-1" in str(path):
                msg = "second part is unreadable"
                raise OSError(msg)
            return original(path, device=device)

        monkeypatch.setattr(NativeStub, "load", fail_second)
        with pytest.raises(RemoteExecutionError, match="injected finalization failure"):
            task.status()
        assert not task.is_fitted and not artifact.exists()
        assert not (task.path / "training_result.json").exists()
        assert task._store.latest("train")["state"] == "failed"
    else:
        task.status()
        assert task._store.latest("train")["state"] == "succeeded"
        assert [(part.layout, part.group_value) for part in task._models] == [
            ("global", None),
            ("per_group", "a"),
        ]
        assert task.training_result().validation_metrics == {
            "global": 0.75,
            "per_group:a": 0.75,
        }
        assert BinaryTask.load(task.path).id == identity
    assert list((tmp_path / "automl").iterdir()) == [task.path]


@pytest.mark.parametrize("layout", ["global", "per_group"])
def test_a_streaming_backend_never_has_its_splits_read(
    context, tmp_path, monkeypatch, layout
):
    """The seam TabNN needs: a source descriptor, and no frame anywhere."""
    context = context.derive(
        backend="tabnn", engine="tabular_transformer", model_layout=layout
    )
    path = tmp_path / "data.parquet"
    pl.DataFrame({
        "client": [3, 1, 2],
        "month": ["2026-01"] * 3,
        "channel": ["b", "a", "b"],
        "x": [3, 1, 2],
        "target": [0, 1, 0],
    }).write_parquet(path)

    def no_full_read(*args, **kwargs):
        pytest.fail("A streaming backend must not have its splits materialized")

    monkeypatch.setattr(ParquetSource, "read", no_full_read)
    monkeypatch.setattr(
        pl.DataFrame,
        "partition_by",
        lambda *args, **kwargs: pytest.fail("Group routing must not partition a frame"),
    )

    seen = []

    def fit(train, valid, *, layout, group_value):
        seen.append((layout, group_value))
        assert train.frame is None
        assert valid.frame is None
        assert train.group_value == group_value
        assert train.source.files == (path,)
        with pytest.raises(RuntimeError, match="not materialized"):
            train.require_frame()
        return model(layout, group_value)

    TrainingCoordinator(context, "binary", lambda *_: None, fit).execute(path, path)
    assert seen == (
        [("global", None)]
        if layout == "global"
        else [("per_group", "a"), ("per_group", "b")]
    )


def test_a_materializing_backend_still_gets_its_frame(context, tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({
        "client": [1, 2],
        "month": ["2026-01"] * 2,
        "channel": ["a", "a"],
        "x": [1, 2],
    }).write_parquet(path)

    seen = []

    def fit(train, valid, *, layout, group_value):
        seen.append(train.require_frame().height)
        assert train.source.files == (path,)
        return model(layout, group_value)

    TrainingCoordinator(
        context.derive(model_layout="global"), "binary", lambda *_: None, fit
    ).execute(path, path)
    assert seen == [2]
