"""Tests for persistent AutoML identity and dataset/result routing."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from fmlib.automl.exceptions import ArtifactIntegrityError, RemoteExecutionError
from fmlib.automl.lifecycle import AutoMLStore, normalize_dataset_path
from fmlib.automl.types import PredictionResult


def _store(tmp_path: Path) -> AutoMLStore:
    return AutoMLStore.create(tmp_path, "binary", {"output_dir": str(tmp_path)})


def test_dataset_registry_is_a_bijection_between_canonical_path_and_key(tmp_path):
    store = _store(tmp_path)
    first = tmp_path / "first" / "test"
    second = tmp_path / "second" / "test"

    first_path, first_key = store.dataset(first, create=True)
    second_path, second_key = store.dataset(second, create=True)

    assert first_path == normalize_dataset_path(first)
    assert second_path == normalize_dataset_path(second)
    assert first_key != second_key
    registry = json.loads((store.root / "datasets.json").read_text(encoding="utf-8"))
    assert registry["by_path"] == {first_path: first_key, second_path: second_key}
    assert registry["by_key"] == {first_key: first_path, second_key: second_path}


def test_dataset_registry_rejects_a_broken_reverse_mapping(tmp_path):
    store = _store(tmp_path)
    dataset = tmp_path / "test"
    canonical, key = store.dataset(dataset, create=True)
    registry_path = store.root / "datasets.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["by_key"][key] = str(tmp_path / "another")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="not bijective"):
        store.dataset(canonical, create=False)


def test_prediction_can_only_be_loaded_for_its_exact_dataset_path(tmp_path):
    store = _store(tmp_path)
    first = tmp_path / "first" / "test"
    second = tmp_path / "second" / "test"
    operation = store.start_operation("predict", dataset_path=first)
    prediction = PredictionResult(pl.DataFrame({"epk_id": [1], "score": [0.75]}))
    scores_path = store.persist_prediction(first, prediction)
    store.update_operation(operation, state="succeeded", result_path=str(scores_path))

    assert store.load_prediction(first).scores.equals(prediction.scores)
    with pytest.raises(ArtifactIntegrityError, match="No operation is registered"):
        store.load_prediction(second)


def test_active_duplicate_prediction_for_the_same_path_is_rejected(tmp_path):
    store = _store(tmp_path)
    dataset = tmp_path / "test"
    operation = store.start_operation("predict", dataset_path=dataset)
    store.update_operation(operation, state="running")

    with pytest.raises(RemoteExecutionError, match="already active"):
        store.start_operation("predict", dataset_path=dataset)


def test_dead_local_owner_is_interrupted_and_no_longer_blocks_retry(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    dataset = tmp_path / "test"
    operation = store.start_operation(
        "predict", dataset_path=dataset, runtime={"env_type": "local"}
    )
    store.update_operation(operation, state="running")
    monkeypatch.setattr(
        AutoMLStore, "_local_owner_is_alive", staticmethod(lambda _owner: False)
    )

    recovered = store.reconcile_local_operations()

    assert len(recovered) == 1
    assert recovered[0]["state"] == "interrupted"
    retry = store.start_operation(
        "predict", dataset_path=dataset, runtime={"env_type": "osiris"}
    )
    assert retry["state"] == "created"


def test_live_local_owner_keeps_duplicate_prediction_locked(tmp_path, monkeypatch):
    store = _store(tmp_path)
    dataset = tmp_path / "test"
    operation = store.start_operation(
        "predict", dataset_path=dataset, runtime={"env_type": "local"}
    )
    store.update_operation(operation, state="running")
    monkeypatch.setattr(
        AutoMLStore, "_local_owner_is_alive", staticmethod(lambda _owner: True)
    )

    assert store.reconcile_local_operations() == []
    with pytest.raises(RemoteExecutionError, match="already active"):
        store.start_operation(
            "predict", dataset_path=dataset, runtime={"env_type": "osiris"}
        )


def test_expired_cross_host_heartbeat_is_interrupted(tmp_path, monkeypatch):
    store = _store(tmp_path)
    operation = store.start_operation(
        "evaluate", dataset_path=tmp_path / "test", runtime={"env_type": "local"}
    )
    store.update_operation(
        operation,
        state="running",
        owner={
            "hostname": "another-host",
            "pid": 123,
            "token": operation["owner"]["token"],
        },
        heartbeat_at=0.0,
    )
    monkeypatch.setattr("fmlib.automl.lifecycle.time.time", lambda: 121.0)

    recovered = store.reconcile_local_operations()
    assert recovered[0]["state"] == "interrupted"
    assert "heartbeat expired" in recovered[0]["error"]
