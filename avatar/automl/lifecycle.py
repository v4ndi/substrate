"""Persistent identity, dataset routing, and operation state for AutoML tasks."""

# ruff: noqa: EM101, EM102

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import polars as pl

from avatar.automl.exceptions import ArtifactIntegrityError, RemoteExecutionError
from avatar.automl.result_io import evaluation_from_payload, evaluation_payload
from avatar.automl.types import (
    CalibrationResult,
    EvaluationResult,
    PredictionResult,
    TrainingResult,
)

_ACTIVE_STATES = {"created", "queued", "running", "unknown"}
_LOCAL_HEARTBEAT_TIMEOUT_SECONDS = 120.0


def json_default(value: Any) -> Any:
    """Convert values used by lifecycle records to JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one human-readable JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, default=json_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    """Read one lifecycle JSON record."""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def normalize_dataset_path(path: str | Path) -> str:
    """Return the canonical absolute spelling used as a dataset identity."""
    return str(Path(path).expanduser().resolve())


def _dataset_slug(path: str) -> str:
    name = Path(path).name or "dataset"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "dataset"
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:48]}-{digest}"


class AutoMLStore:
    """Own the persistent state of one task instance."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    @classmethod
    def create(
        cls, output_dir: str | Path, task_name: str, config: Mapping[str, Any]
    ) -> AutoMLStore:
        """Create a new uniquely identified AutoML entity."""
        automl_id = uuid.uuid4().hex
        store = cls(Path(output_dir) / "automl" / automl_id)
        store.root.mkdir(parents=True, exist_ok=False)
        write_json(
            store.root / "entity.json",
            {
                "automl_id": automl_id,
                "task": task_name,
                "config": dict(config),
                "artifact_path": None,
            },
        )
        write_json(store.root / "datasets.json", {"by_path": {}, "by_key": {}})
        return store

    @property
    def entity(self) -> dict[str, Any]:
        return read_json(self.root / "entity.json")

    @property
    def automl_id(self) -> str:
        return str(self.entity["automl_id"])

    def dataset(self, path: str | Path, *, create: bool) -> tuple[str, str]:
        """Resolve the bijective canonical-path/dataset-key registry entry."""
        canonical = normalize_dataset_path(path)
        registry_path = self.root / "datasets.json"
        registry = read_json(registry_path)
        by_path = dict(registry.get("by_path", {}))
        by_key = dict(registry.get("by_key", {}))
        if canonical in by_path:
            key = str(by_path[canonical])
            if by_key.get(key) != canonical:
                raise ArtifactIntegrityError(
                    f"Dataset registry is not bijective for path {canonical!r}"
                )
            return canonical, key
        if not create:
            raise ArtifactIntegrityError(
                f"No operation is registered for dataset path: {canonical}"
            )
        key = _dataset_slug(canonical)
        owner = by_key.get(key)
        if owner is not None and owner != canonical:
            raise ArtifactIntegrityError(
                f"Dataset-key collision: {key!r} maps to both {owner!r} and {canonical!r}"
            )
        by_path[canonical] = key
        by_key[key] = canonical
        write_json(registry_path, {"by_path": by_path, "by_key": by_key})
        return canonical, key

    def operation_dir(self, action: str, run_id: str) -> Path:
        return self.root / "operations" / action / run_id

    def start_operation(
        self,
        action: str,
        *,
        dataset_path: str | Path | None = None,
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a persistent operation record before local work or submission."""
        self.reconcile_local_operations()
        canonical = key = None
        if dataset_path is not None:
            canonical, key = self.dataset(dataset_path, create=True)
            active = [
                item
                for item in self.operations(action=action, dataset_key=key)
                if item.get("state") in _ACTIVE_STATES
            ]
            if active:
                raise RemoteExecutionError(
                    f"{action} is already active for test_path={canonical!r}; run_id={active[-1]['run_id']}"
                )
        run_id = uuid.uuid4().hex
        runtime_values = dict(runtime or {})
        record = {
            "run_id": run_id,
            "action": action,
            "state": "created",
            "dataset_path": canonical,
            "dataset_key": key,
            "runtime": runtime_values,
            "jobs": [],
            "result_path": None,
            "error": None,
        }
        if runtime_values.get("env_type") == "local":
            record["owner"] = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "token": uuid.uuid4().hex,
            }
            record["heartbeat_at"] = time.time()
        write_json(self.operation_dir(action, run_id) / "operation.json", record)
        return record

    def update_operation(
        self, record: Mapping[str, Any], **updates: Any
    ) -> dict[str, Any]:
        """Persist state changes for an existing operation."""
        path = (
            self.operation_dir(str(record["action"]), str(record["run_id"]))
            / "operation.json"
        )
        current = read_json(path) if path.is_file() else dict(record)
        updated = current | updates
        write_json(path, updated)
        return updated

    @staticmethod
    def _local_owner_is_alive(owner: Mapping[str, Any]) -> bool | None:
        """Return local PID liveness, or ``None`` when the owner is on another host."""
        if owner.get("hostname") != socket.gethostname():
            return None
        try:
            pid = int(owner["pid"])
            if pid < 1:
                return False
            os.kill(pid, 0)
        except (KeyError, TypeError, ValueError, ProcessLookupError):
            return False
        except PermissionError:
            return True
        return True

    def heartbeat_operation(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Refresh a running local operation without reviving a terminal record."""
        path = (
            self.operation_dir(str(record["action"]), str(record["run_id"]))
            / "operation.json"
        )
        current = read_json(path)
        expected_token = (record.get("owner") or {}).get("token")
        current_token = (current.get("owner") or {}).get("token")
        if (
            current.get("state") not in _ACTIVE_STATES
            or current_token != expected_token
        ):
            return current
        return self.update_operation(current, heartbeat_at=time.time())

    @classmethod
    def _local_interruption_reason(cls, record: Mapping[str, Any]) -> str | None:
        owner = record.get("owner")
        heartbeat_at = record.get("heartbeat_at")
        if not isinstance(owner, Mapping) or not isinstance(heartbeat_at, int | float):
            return None
        alive = cls._local_owner_is_alive(owner)
        if alive is False:
            return f"local owner process {owner.get('hostname')}:{owner.get('pid')} is not running"
        if (
            alive is None
            and time.time() - float(heartbeat_at) > _LOCAL_HEARTBEAT_TIMEOUT_SECONDS
        ):
            return f"local heartbeat expired after {_LOCAL_HEARTBEAT_TIMEOUT_SECONDS:g} seconds"
        return None

    def reconcile_local_operations(self) -> list[dict[str, Any]]:
        """Mark verifiably abandoned local operations as interrupted."""
        recovered: list[dict[str, Any]] = []
        for record in self.operations():
            if (
                record.get("state") not in _ACTIVE_STATES
                or record.get("jobs")
                or (record.get("runtime") or {}).get("env_type") != "local"
            ):
                continue
            reason = self._local_interruption_reason(record)
            if reason is None:
                continue
            recovered.append(
                self.update_operation(
                    record,
                    state="interrupted",
                    error=f"Interrupted local operation: {reason}",
                )
            )
        return recovered

    def operations(
        self, *, action: str | None = None, dataset_key: str | None = None
    ) -> list[dict[str, Any]]:
        """Return operation records ordered by filesystem modification time."""
        base = self.root / "operations"
        paths = list(base.glob("*/*/operation.json")) if base.exists() else []
        records = [
            read_json(path)
            for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns)
        ]
        if action is not None:
            records = [item for item in records if item.get("action") == action]
        if dataset_key is not None:
            records = [
                item for item in records if item.get("dataset_key") == dataset_key
            ]
        return records

    def latest(
        self, action: str, dataset_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Return the latest matching operation or a clear lifecycle error."""
        key = None
        if dataset_path is not None:
            _, key = self.dataset(dataset_path, create=False)
        records = self.operations(action=action, dataset_key=key)
        if not records:
            suffix = (
                ""
                if dataset_path is None
                else f" for test_path={normalize_dataset_path(dataset_path)!r}"
            )
            raise ArtifactIntegrityError(f"{action} was not started{suffix}")
        return records[-1]

    def prediction_dir(self, dataset_key: str) -> Path:
        return self.root / "predictions" / dataset_key

    def evaluation_dir(self, dataset_key: str) -> Path:
        return self.root / "evaluations" / dataset_key

    def calibration_dir(self, dataset_key: str) -> Path:
        return self.root / "calibrations" / dataset_key

    def persist_training(self, result: TrainingResult) -> Path:
        path = self.root / "training_result.json"
        write_json(path, asdict(result))
        return path

    def load_training(self) -> TrainingResult:
        path = self.root / "training_result.json"
        if not path.exists():
            raise ArtifactIntegrityError("Training result is not available")
        value = read_json(path)
        return TrainingResult(
            best_params=value["best_params"],
            validation_metrics=value["validation_metrics"],
            feature_names=tuple(value["feature_names"]),
            backend_name=value["backend_name"],
            engine_name=value["engine_name"],
            task_name=value["task_name"],
        )

    def persist_prediction(
        self, dataset_path: str | Path, prediction: PredictionResult
    ) -> Path:
        canonical, key = self.dataset(dataset_path, create=True)
        destination = self.prediction_dir(key)
        destination.mkdir(parents=True, exist_ok=True)
        scores_path = destination / "scores.parquet"
        temporary = destination / "scores.tmp.parquet"
        stored = prediction.scores
        stored.write_parquet(temporary)
        os.replace(temporary, scores_path)
        write_json(
            destination / "prediction.json",
            {
                "test_path": canonical,
                "dataset_key": key,
                "scores_path": str(scores_path),
                "class_order": prediction.class_order,
            },
        )
        return scores_path

    def load_prediction(self, dataset_path: str | Path) -> PredictionResult:
        scores_path, metadata = self.prediction_path(dataset_path)
        stored = pl.read_parquet(scores_path)
        class_order = metadata.get("class_order")
        return PredictionResult(
            stored, class_order=None if class_order is None else tuple(class_order)
        )

    def prediction_path(self, dataset_path: str | Path) -> tuple[Path, dict[str, Any]]:
        """Validate a completed prediction and return its parquet without reading it."""
        canonical, key = self.dataset(dataset_path, create=False)
        operation = self.latest("predict", canonical)
        state = operation.get("state")
        if state != "succeeded":
            raise RemoteExecutionError(
                f"Prediction for test_path={canonical!r} is not available: state={state}, error={operation.get('error')!r}"
            )
        metadata_path = self.prediction_dir(key) / "prediction.json"
        if not metadata_path.exists():
            raise ArtifactIntegrityError(
                f"Succeeded prediction has no metadata: {metadata_path}"
            )
        metadata = read_json(metadata_path)
        if metadata.get("test_path") != canonical or metadata.get("dataset_key") != key:
            raise ArtifactIntegrityError(
                f"Prediction metadata does not match dataset registry for {canonical!r}"
            )
        scores_path = Path(metadata["scores_path"])
        if not scores_path.is_file():
            raise ArtifactIntegrityError(
                f"Succeeded prediction has no scores parquet: {scores_path}"
            )
        return scores_path, metadata

    def persist_calibration(
        self,
        test_path: str | Path,
        calibration_path: str | Path,
        result: CalibrationResult,
        state: Mapping[str, Any],
    ) -> Path:
        canonical, key = self.dataset(test_path, create=True)
        canonical_calibration, _ = self.dataset(calibration_path, create=False)
        destination = self.calibration_dir(key)
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / "scores.parquet"
        temporary = destination / "scores.tmp.parquet"
        result.scores.write_parquet(temporary)
        os.replace(temporary, result_path)
        write_json(
            destination / "calibration.json",
            {
                "test_path": canonical,
                "calibration_path": canonical_calibration,
                "dataset_key": key,
                "result_path": str(result_path),
                "calibration_strategy": result.calibration_strategy,
                "state": dict(state),
            },
        )
        return result_path

    def load_calibration(self, test_path: str | Path) -> CalibrationResult:
        canonical, key = self.dataset(test_path, create=False)
        operation = self.latest("calibrate", canonical)
        if operation.get("state") != "succeeded":
            raise RemoteExecutionError(
                f"Calibration for test_path={canonical!r} is not available: state={operation.get('state')}, "
                f"error={operation.get('error')!r}"
            )
        metadata_path = self.calibration_dir(key) / "calibration.json"
        if not metadata_path.is_file():
            raise ArtifactIntegrityError(
                f"Succeeded calibration has no metadata: {metadata_path}"
            )
        metadata = read_json(metadata_path)
        if metadata.get("test_path") != canonical or metadata.get("dataset_key") != key:
            raise ArtifactIntegrityError(
                f"Calibration metadata does not match dataset registry for {canonical!r}"
            )
        result_path = Path(metadata["result_path"])
        if not result_path.is_file():
            raise ArtifactIntegrityError(
                f"Succeeded calibration has no scores parquet: {result_path}"
            )
        return CalibrationResult(
            pl.read_parquet(result_path), str(metadata["calibration_strategy"])
        )

    def score_kind(
        self, scores_path: str | Path
    ) -> Literal["raw", "calibrated"] | None:
        """Identify a persisted prediction or calibration result parquet path."""
        canonical = normalize_dataset_path(scores_path)
        kinds: set[Literal["raw", "calibrated"]] = set()
        for metadata_path in self.root.glob("predictions/*/prediction.json"):
            metadata = read_json(metadata_path)
            if normalize_dataset_path(metadata["scores_path"]) == canonical:
                kinds.add("raw")
        for metadata_path in self.root.glob("calibrations/*/calibration.json"):
            metadata = read_json(metadata_path)
            if normalize_dataset_path(metadata["result_path"]) == canonical:
                kinds.add("calibrated")
        return next(iter(kinds)) if len(kinds) == 1 else None

    def persist_evaluation(
        self, dataset_path: str | Path, result: EvaluationResult
    ) -> tuple[Path, EvaluationResult]:
        canonical, key = self.dataset(dataset_path, create=True)
        directory = self.evaluation_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "evaluation_result.json"
        merged = (
            evaluation_from_payload(read_json(path)).merge(result)
            if path.is_file()
            else result
        )
        payload = evaluation_payload(merged)
        figure_paths = {}
        for kind in ("raw", "calibrated"):
            figure_paths[f"figure_paths_{kind}"] = {
                name: str(directory / f"{name}.png")
                for name in getattr(merged, f"figures_{kind}")
                if (directory / f"{name}.png").is_file()
            }
        payload.update({
            "test_path": canonical,
            "dataset_key": key,
            **figure_paths,
        })
        write_json(path, payload)
        return path, evaluation_from_payload(payload)

    def load_evaluation(self, dataset_path: str | Path) -> EvaluationResult:
        canonical, key = self.dataset(dataset_path, create=False)
        operation = self.latest("evaluate", canonical)
        if operation.get("state") != "succeeded":
            raise RemoteExecutionError(
                f"Evaluation for test_path={canonical!r} is not available: state={operation.get('state')}, "
                f"error={operation.get('error')!r}"
            )
        path = self.evaluation_dir(key) / "evaluation_result.json"
        if not path.exists():
            raise ArtifactIntegrityError(f"Succeeded evaluation has no result: {path}")
        value = read_json(path)
        if value.get("test_path") != canonical or value.get("dataset_key") != key:
            raise ArtifactIntegrityError(
                f"Evaluation metadata does not match dataset registry for {canonical!r}"
            )
        return evaluation_from_payload(value)
