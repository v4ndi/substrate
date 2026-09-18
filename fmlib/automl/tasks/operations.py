"""Local/remote operation lifecycle over explicit execution and commit hooks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from fmlib.automl.calibrators import CalibrationStrategy
from fmlib.automl.config.base import BaseTaskConfig, EnvironmentConfig
from fmlib.automl.environment import EnvironmentRunner
from fmlib.automl.exceptions import (
    ArtifactIntegrityError,
    ConfigError,
    RemoteExecutionError,
)
from fmlib.automl.execution import ExecutionContext
from fmlib.automl.lifecycle import AutoMLStore, normalize_dataset_path, read_json
from fmlib.automl.metrics import resolve_metric
from fmlib.automl.result_io import evaluation_from_payload
from fmlib.automl.types import (
    CalibrationResult,
    EvaluationKind,
    EvaluationResult,
    ParquetPath,
    PredictionResult,
    TrainingResult,
)

from .artifacts import jsonable as _jsonable

if TYPE_CHECKING:
    from .base import BaseTask
logger = logging.getLogger(__name__)
_LOCAL_HEARTBEAT_INTERVAL_SECONDS = 10.0
_REMOTE_ROW_ID = "__fmlib_remote_row_id"
_CALIBRATION_ROW_ID = "__fmlib_calibration_row_id"


def _runtime_metadata(config: BaseTaskConfig, **extra: Any) -> dict[str, Any]:
    environment = config.environment
    return {
        "env_type": config.env_type,
        "device": config.device,
        "original_pool": environment.pool,
        "effective_pool": environment.effective_pool,
        "resource_profile": environment.resource_profile,
        "num_gpus": environment.resolved_num_gpus,
        "num_nodes": environment.resolved_num_nodes,
        **extra,
    }


@dataclass(frozen=True)
class RemoteTrainingParts:
    """Where a remote training run left its artifact and per-part outputs."""

    artifact_path: Path
    part_paths: tuple[Path, ...]


@dataclass(frozen=True)
class OperationHooks:
    """Only the facade can create execution views or commit fitted state."""

    resolve_config: Callable[..., BaseTaskConfig]
    execution_view: Callable[..., BaseTask]
    save: Callable[..., Path]
    adopt: Callable[[BaseTask], None]
    rollback_training: Callable[[Path], None]
    set_artifact: Callable[[Path], None]
    assemble_training: Callable[[RemoteTrainingParts], tuple[str, ...]]
    require_fitted: Callable[[], None]
    storage_frame: Callable[[PredictionResult], pl.DataFrame]


@dataclass(frozen=True)
class OperationRunner:
    """Lifecycle state lives in the store; task changes use explicit commit hooks."""

    config: BaseTaskConfig
    store: AutoMLStore
    environment: EnvironmentRunner
    hooks: OperationHooks
    is_fitted: bool

    @property
    def path(self) -> Path:
        return self.store.root

    @contextmanager
    def _track_local_operation(
        self, operation: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Keep a local operation lease alive until its synchronous work finishes."""
        current = self.store.update_operation(
            operation, state="running", heartbeat_at=time.time()
        )
        stop = Event()

        def heartbeat() -> None:
            while not stop.wait(_LOCAL_HEARTBEAT_INTERVAL_SECONDS):
                try:
                    self.store.heartbeat_operation(current)
                except Exception:
                    logger.exception(
                        "Cannot persist heartbeat for local run_id=%s",
                        current["run_id"],
                    )

        thread = Thread(
            target=heartbeat,
            name=f"automl-heartbeat-{current['run_id'][:8]}",
            daemon=True,
        )
        thread.start()
        try:
            yield current
        finally:
            stop.set()
            thread.join()

    def train(
        self,
        train_path: ParquetPath,
        valid_path: ParquetPath,
        *,
        env_type: Literal["local", "osiris"] | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
    ) -> TrainingResult | None:
        """Train a model and optionally run Optuna hyperparameter search."""
        runtime_config = self.hooks.resolve_config(
            env_type=env_type,
            device=device,
            environment=environment,
        )
        if self.is_fitted:
            msg = "This AutoML entity is already fitted; create a new task to retrain"
            raise ConfigError(msg)
        if self.store.operations(action="train"):
            msg = "This AutoML entity already has a training operation; create a new task to retrain"
            raise ConfigError(msg)
        operation = self.store.start_operation(
            "train",
            runtime=_runtime_metadata(runtime_config),
        )
        artifact_path = self.path / "artifact"
        try:
            execution = self.hooks.execution_view(
                ExecutionContext.from_config(runtime_config), prepare_backends=False
            )
            if runtime_config.env_type == "local":
                with self._track_local_operation(operation) as operation:
                    result = execution._runner().run_local(
                        config=execution.config,
                        action="train",
                        callback=lambda: execution._execute_train(
                            train_path, valid_path
                        ),
                    )
                    self.hooks.adopt(execution)
                    self.hooks.save(artifact_path)
                    self.store.persist_training(result)
                    self.hooks.set_artifact(artifact_path)
                self.store.update_operation(
                    operation,
                    state="succeeded",
                    result_path=str(self.path / "training_result.json"),
                )
                return result

            plan = execution._remote_job_plan(train_path)
            planned: list[dict[str, Any]] = []
            for index, entry in enumerate(plan):
                part_artifact = (
                    self.path
                    / ".remote_parts"
                    / operation["run_id"]
                    / f"part-{index:04d}"
                )
                planned.append({
                    "state": "planned",
                    "job_id": None,
                    "index": index,
                    "layout": entry["layout"],
                    "group_value": entry["group_value"],
                    "trial": entry["trial"],
                    "run_dir": str(
                        self.store.operation_dir("train", operation["run_id"])
                        / f"job-{index:04d}"
                    ),
                    "payload": {
                        "train_path": normalize_dataset_path(train_path),
                        "valid_path": normalize_dataset_path(valid_path),
                        "entity_config": _jsonable(asdict(self.config)),
                        "artifact_path": str(part_artifact),
                        "remote_layout": entry["layout"],
                        "remote_group_value": entry["group_value"],
                        **({"trial": entry["trial"]} if entry["trial"] else {}),
                    },
                })
            # Parameters are written before anything is submitted, so a search
            # that is interrupted mid-submit can be resumed from the record
            # rather than replanned -- replanning would renumber the trials.
            operation = self.store.update_operation(
                operation,
                state="submitting",
                jobs=planned,
                artifact_path=str(artifact_path),
            )
            operation = self._submit_planned_jobs(operation, execution)
            return None
        except Exception as exc:
            self.hooks.rollback_training(artifact_path)
            # The jobs already submitted stay on the record: they are running,
            # and an id nobody kept is an orphan nobody can stop.
            self.store.update_operation(
                operation,
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _poll(self, operation: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        """Poll the submitted jobs, leaving the planned ones alone.

        A planned job has no scheduler id, so asking about it would read as a
        job the scheduler has lost. It is simply not submitted yet.
        """
        jobs = list(operation.get("jobs") or ())
        submitted = [job for job in jobs if job.get("job_id") is not None]
        planned = [job for job in jobs if job.get("job_id") is None]
        if not submitted:
            return "submitting", jobs
        aggregate, polled = self.environment.poll_jobs(submitted)
        merged = sorted(
            [*polled, *planned],
            key=lambda job: job.get("index", 0),
        )
        if planned and aggregate in {"succeeded", "failed", "partial_failed"}:
            # Some of the plan has not been submitted yet, so nothing about it
            # is finished, whatever the submitted part says.
            aggregate = "submitting"
        return aggregate, merged

    def _submit_planned_jobs(
        self, operation: Mapping[str, Any], execution: Any | None = None
    ) -> dict[str, Any]:
        """Submit jobs still ``planned``, up to ``max_parallel_jobs`` in flight.

        Resumable by construction: the plan lives on the record, so a driver
        that died mid-submit leaves entries a later ``status()`` picks up, and
        an entry that already has a ``job_id`` is never submitted twice.

        Args:
            operation: The operation whose plan to advance.
            execution: The execution view to submit through; built if omitted.

        Returns:
            The updated operation.
        """
        jobs = [dict(job) for job in operation.get("jobs") or ()]
        pending = [job for job in jobs if job.get("job_id") is None]
        if not pending:
            return self.store.update_operation(
                operation, state=operation.get("state", "queued"), jobs=jobs
            )
        if execution is None:
            execution = self.hooks.execution_view(
                ExecutionContext.from_config(self.config), prepare_backends=False
            )
        limit = self.config.max_parallel_jobs
        in_flight = sum(
            1
            for job in jobs
            if job.get("job_id") is not None
            and job.get("state") not in {"succeeded", "failed", "lost"}
        )
        for job in pending:
            if limit is not None and in_flight >= limit:
                break
            suffix = f"part-{job['index']:04d}"
            if job.get("trial"):
                suffix = f"{suffix}-{job['trial']['trial_id']}"
            handle = execution._runner().submit(
                config=execution.config,
                action="train",
                payload=job["payload"],
                run_dir=Path(job["run_dir"]),
                job_suffix=suffix,
            )
            job.update(handle)
            job["state"] = handle.get("state", "queued")
            in_flight += 1
            # Persisted after *every* submit, before the next one. A crash here
            # leaves a job that is running and holding cards; losing its id
            # would leave nobody able to find it.
            operation = self.store.update_operation(
                operation, state="submitting", jobs=jobs
            )
        remaining = any(job.get("job_id") is None for job in jobs)
        return self.store.update_operation(
            operation, state="submitting" if remaining else "queued", jobs=jobs
        )

    def _finalize_remote_train(self, operation: Mapping[str, Any]) -> None:
        """Publish training only after all worker artifacts have been assembled.

        With one job per model part -- which is every boosting run -- every
        successful job contributes. With one job per *trial*, the jobs of one
        part are competitors, and only the winner's artifact is assembled.
        """
        artifact_path = Path(str(operation["artifact_path"]))
        winners = self._winning_jobs(operation)
        part_paths = (
            ()
            if artifact_path.exists()
            else tuple(
                Path(read_json(Path(job["spec_path"]))["payload"]["artifact_path"])
                for job in winners
            )
        )
        feature_names = self.hooks.assemble_training(
            RemoteTrainingParts(artifact_path, part_paths)
        )
        self.hooks.set_artifact(artifact_path)
        best_params: dict[str, Any] = {}
        validation_metrics: dict[str, float] = {}
        for job in winners:
            result = read_json(Path(job["result_path"]))
            best_params.update(result.get("best_params", {}))
            validation_metrics.update(result.get("validation_metrics", {}))
        self.store.persist_training(
            TrainingResult(
                best_params=best_params,
                validation_metrics=validation_metrics,
                feature_names=feature_names,
                backend_name=self.config.backend,
                engine_name=self.config.engine,
                task_name=self.store.entity["task"],
            )
        )

    def _winning_jobs(self, operation: Mapping[str, Any]) -> list[dict[str, Any]]:
        """One job per model part: the best trial of it, or the only one.

        Failed trials are dropped rather than fatal: a search survives losing
        some of its trials, and an operation is successful when every part
        ended up with a model.

        Returns:
            The contributing jobs, in plan order.
        """
        jobs = list(operation.get("jobs") or ())
        if not any(job.get("trial") for job in jobs):
            # One job per model part: every one of them contributes, which is
            # what this has always done and what boosting still does.
            return [dict(job) for job in jobs]
        metric = resolve_metric(
            self.config.optimization_metric, self.config.task_name, "optimization"
        )
        maximize = metric.optimization_direction == "maximize"
        best: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
        for job in jobs:
            if job.get("state") != "succeeded":
                continue
            result = read_json(Path(job["result_path"]))
            values = [
                float(value)
                for value in (result.get("validation_metrics") or {}).values()
            ]
            score = (max if maximize else min)(values) if values else None
            key = (job.get("layout"), job.get("group_value"), job.get("index", 0))
            part = key[:2] if job.get("trial") else key
            if score is None:
                best.setdefault(
                    part, (float("-inf") if maximize else float("inf"), job)
                )
                continue
            current = best.get(part)
            better = (
                current is None
                or (maximize and score > current[0])
                or (not maximize and score < current[0])
            )
            if better:
                best[part] = (score, job)
        return [
            job for _, (_, job) in sorted(best.items(), key=lambda item: str(item[0]))
        ]

    def _finalize_remote_prediction(self, operation: Mapping[str, Any]) -> None:
        """Merge ordered global/per-group job scores and publish one prediction."""
        parts: list[tuple[str, pl.DataFrame]] = []
        class_orders: set[tuple[Any, ...] | None] = set()
        for job in operation["jobs"]:
            result = read_json(Path(job["result_path"]))
            scores_path = Path(result["scores_path"])
            if not scores_path.is_file():
                msg = f"Remote prediction did not create scores: {scores_path}"
                raise ArtifactIntegrityError(msg)
            stored = pl.read_parquet(scores_path)
            if _REMOTE_ROW_ID not in stored.columns:
                msg = f"Remote prediction scores are missing internal row identity: {scores_path}"
                raise ArtifactIntegrityError(msg)
            spec = read_json(Path(job["spec_path"]))
            parts.append((str(spec["payload"]["remote_layout"]), stored))
            class_order = result.get("class_order")
            class_orders.add(None if class_order is None else tuple(class_order))
        if len(class_orders) != 1:
            msg = f"Remote prediction jobs returned inconsistent class orders: {class_orders!r}"
            raise ArtifactIntegrityError(msg)

        parent_layout = str((operation.get("runtime") or {}).get("model_layout"))
        layout_order = [
            layout
            for layout in ("global", "per_group")
            if any(item[0] == layout for item in parts)
        ]

        def combine() -> pl.DataFrame:
            frames: list[pl.DataFrame] = []
            for layout in layout_order:
                layout_frames = [
                    scores for part_layout, scores in parts if part_layout == layout
                ]
                frame = (
                    layout_frames[0]
                    if len(layout_frames) == 1
                    else pl.concat(layout_frames, how="diagonal_relaxed")
                ).sort(_REMOTE_ROW_ID)
                if parent_layout == "global_and_per_group":
                    frame = frame.with_columns(pl.lit(layout).alias("model_layout"))
                frames.append(frame.drop(_REMOTE_ROW_ID))
            return (
                frames[0]
                if len(frames) == 1
                else pl.concat(frames, how="diagonal_relaxed")
            )

        scores = combine()
        class_order = next(iter(class_orders))
        self.store.persist_prediction(
            str(operation["dataset_path"]),
            PredictionResult(scores, class_order=class_order),
        )

    def _finalize_remote_evaluation(self, operation: Mapping[str, Any]) -> None:
        """Publish the JSON result produced by a remote evaluation worker."""
        result = read_json(Path(operation["jobs"][0]["result_path"]))
        evaluation = evaluation_from_payload(result)
        self.store.persist_evaluation(
            str(operation["dataset_path"]),
            evaluation,
        )

    def _evaluation_kind(
        self, scores: PredictionResult | CalibrationResult | ParquetPath | None
    ) -> EvaluationKind:
        if scores is None or isinstance(scores, PredictionResult):
            return "raw"
        if isinstance(scores, CalibrationResult):
            return "calibrated"
        kind = self.store.score_kind(scores)
        if kind is None:
            msg = (
                f"Cannot determine whether scores_path={normalize_dataset_path(scores)!r} is raw or calibrated; "
                "pass PredictionResult/CalibrationResult or a parquet path registered by this task"
            )
            raise ConfigError(msg)
        return kind

    def _finalize_remote_calibration(self, operation: Mapping[str, Any]) -> None:
        """Merge at most one worker result per model branch in source-row order."""
        frames: list[pl.DataFrame] = []
        states: dict[str, Any] = {}
        strategy: str | None = None
        for job in operation["jobs"]:
            payload = read_json(Path(job["result_path"]))
            frame = pl.read_parquet(payload["scores_path"])
            if _CALIBRATION_ROW_ID not in frame.columns:
                msg = "Remote calibration scores are missing internal row identity"
                raise ArtifactIntegrityError(msg)
            frames.append(frame)
            job_strategy = str(payload["calibration_strategy"])
            if strategy is not None and strategy != job_strategy:
                msg = "Remote calibration jobs returned inconsistent strategies"
                raise ArtifactIntegrityError(msg)
            strategy = job_strategy
            states.update(payload["state"]["branches"])
        scores = (
            pl.concat(frames, how="diagonal_relaxed")
            .sort(_CALIBRATION_ROW_ID)
            .drop(_CALIBRATION_ROW_ID)
        )
        state = {
            "task": self.config.task_name,
            "calibration_strategy": strategy,
            "branches": states,
        }
        self.store.persist_calibration(
            str(operation["dataset_path"]),
            str(operation["runtime"]["calibration_path"]),
            CalibrationResult(scores, strategy or ""),
            state,
        )

    def status(self, *, wait: bool = False) -> pl.DataFrame:
        """Inspect all local and remote operations owned by this task."""
        last_snapshot: str | None = None
        while True:
            self.store.reconcile_local_operations()
            active = False
            failure_messages: list[str] = []
            for operation in self.store.operations():
                if not operation.get("jobs") or operation.get("state") in {
                    "succeeded",
                    "failed",
                    "partial_failed",
                }:
                    continue
                if operation.get("state") == "finalizing":
                    # Left mid-finalization by a driver that died. status() is
                    # a reflection of job state, not a recovery mechanism, so
                    # it reports this and leaves the decision to finalize().
                    active = True
                    continue
                aggregate, jobs = self._poll(operation)
                operation = self.store.update_operation(
                    operation, state=aggregate, jobs=jobs
                )
                if any(job.get("job_id") is None for job in jobs):
                    # Poll first: how many jobs are in flight is what decides
                    # whether the next chunk of the plan may be submitted.
                    operation = self._submit_planned_jobs(operation)
                    aggregate, jobs = self._poll(operation)
                    operation = self.store.update_operation(
                        operation, state=aggregate, jobs=jobs
                    )
                if aggregate == "succeeded" or (
                    aggregate == "partial_failed" and _every_part_has_a_model(jobs)
                ):
                    try:
                        self._finalize(operation)
                    except Exception as exc:
                        self._record_finalization_failure(operation, exc)
                        failure_messages.append(self.environment.failure_logs(jobs))
                elif aggregate in {"failed", "partial_failed"}:
                    diagnostics = self.environment.failure_logs(jobs)
                    self.store.update_operation(operation, error=diagnostics)
                    failure_messages.append(diagnostics)
                else:
                    active = True
            table = self._status_table()
            snapshot = repr(table.to_dicts())
            if snapshot != last_snapshot:
                print(table)
                last_snapshot = snapshot
            if failure_messages:
                diagnostics = "\n".join(failure_messages)
                logger.error("Remote AutoML operation failed:\n%s", diagnostics)
                raise RemoteExecutionError(diagnostics)
            if not wait or not active:
                return table
            time.sleep(self.config.environment.poll_interval_seconds)

    def _finalize(self, operation: Mapping[str, Any]) -> dict[str, Any]:
        """Assemble a finished remote operation and publish it, in that order.

        ``finalizing`` is written first and is **not** terminal, so a driver
        that dies between the jobs finishing and the result being assembled
        leaves an operation that can be picked up again. ``succeeded`` is
        written only once the result exists.

        Args:
            operation: The operation whose jobs have all finished.

        Returns:
            The published operation record.
        """
        operation = self.store.update_operation(operation, state="finalizing")
        action = operation["action"]
        if action == "train":
            self._finalize_remote_train(operation)
            result_path = self.path / "training_result.json"
        elif action == "predict":
            self._finalize_remote_prediction(operation)
            result_path = (
                self.store.prediction_dir(str(operation["dataset_key"]))
                / "scores.parquet"
            )
        elif action == "evaluate":
            self._finalize_remote_evaluation(operation)
            result_path = (
                self.store.evaluation_dir(str(operation["dataset_key"]))
                / "evaluation_result.json"
            )
        else:
            self._finalize_remote_calibration(operation)
            result_path = (
                self.store.calibration_dir(str(operation["dataset_key"]))
                / "scores.parquet"
            )
        return self.store.update_operation(
            operation, state="succeeded", result_path=str(result_path)
        )

    def _record_finalization_failure(
        self, operation: Mapping[str, Any], error: Exception
    ) -> None:
        if operation["action"] == "train":
            self.hooks.rollback_training(Path(str(operation["artifact_path"])))
        self.store.update_operation(
            operation,
            state="failed",
            error=f"Finalization failed: {type(error).__name__}: {error}",
        )

    def finalize(self, *, force: bool = False) -> pl.DataFrame:
        """Finish remote operations whose jobs are done but whose result is not.

        Idempotent: running it when there is nothing to finish does nothing,
        and running it twice on the same operation produces the same published
        result rather than two of anything. The happy path never needs it --
        a live driver finalizes inside ``status()`` -- it exists for the case
        where that driver is gone.

        Args:
            force: Take over an operation another process still owns.

        Returns:
            The status table, as ``status()`` would report it.

        Raises:
            RemoteExecutionError: If an operation could not be finalized.
        """
        failures: list[str] = []
        for operation in self.store.operations():
            if not operation.get("jobs"):
                continue
            if operation.get("state") in {"succeeded", "failed", "partial_failed"}:
                continue
            aggregate, jobs = self.environment.poll_jobs(operation["jobs"])
            if operation.get("state") != "finalizing" and aggregate != "succeeded":
                continue
            claimed = self.store.claim_operation(operation, force=force)
            claimed = self.store.update_operation(claimed, jobs=jobs)
            try:
                self._finalize(claimed)
            except Exception as exc:
                self._record_finalization_failure(claimed, exc)
                failures.append(f"{type(exc).__name__}: {exc}")
        if failures:
            diagnostics = "\n".join(failures)
            logger.error("Finalization failed:\n%s", diagnostics)
            raise RemoteExecutionError(diagnostics)
        return self._status_table()

    def _status_table(self) -> pl.DataFrame:
        """Build a compact status table without exposing internal record objects."""
        rows: list[dict[str, Any]] = []
        for operation in self.store.operations():
            jobs = operation.get("jobs") or [None]
            for job in jobs:
                rows.append({
                    "action": operation["action"],
                    "test_path": operation.get("dataset_path"),
                    "state": operation["state"]
                    if job is None
                    else job.get("state", operation["state"]),
                    "run_id": operation["run_id"],
                    "job_id": None if job is None else job.get("job_id"),
                    "job_name": None if job is None else job.get("job_name"),
                    "result_path": operation.get("result_path"),
                    "error": operation.get("error"),
                })
        columns = (
            "action",
            "test_path",
            "state",
            "run_id",
            "job_id",
            "job_name",
            "result_path",
            "error",
        )
        return (
            pl.DataFrame(rows)
            if rows
            else pl.DataFrame(schema=dict.fromkeys(columns, pl.String))
        )

    def predict(
        self,
        test_path: ParquetPath,
        *,
        env_type: Literal["local", "osiris"] | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
        model_layout: Literal["global", "per_group", "global_and_per_group"]
        | None = None,
    ) -> PredictionResult | None:
        """Return model scores for every row under ``test_path``."""
        self.hooks.require_fitted()
        runtime_config = self.hooks.resolve_config(
            env_type=env_type,
            device=device,
            environment=environment,
            model_layout=model_layout,
        )

        operation = self.store.start_operation(
            "predict",
            dataset_path=test_path,
            runtime=_runtime_metadata(
                runtime_config, model_layout=runtime_config.resolved_model_layout
            ),
        )
        try:
            artifact_path = (
                self.hooks.save(self.path / "artifact", overwrite=True)
                if runtime_config.env_type != "local"
                else None
            )
            execution = self.hooks.execution_view(
                ExecutionContext.from_config(runtime_config),
                prepare_backends=runtime_config.env_type == "local",
            )
            execution._validate_prediction_layout()
            if runtime_config.env_type == "local":
                with self._track_local_operation(operation) as operation:
                    prediction = execution._runner().run_local(
                        config=execution.config,
                        action="predict",
                        callback=lambda: execution._execute_predict(test_path),
                    )
                    scores_path = self.store.persist_prediction(test_path, prediction)
                self.store.update_operation(
                    operation, state="succeeded", result_path=str(scores_path)
                )
                return prediction
            jobs: list[dict[str, Any]] = []
            for index, (layout, group_value) in enumerate(
                execution._remote_prediction_parts(test_path)
            ):
                run_dir = (
                    self.store.operation_dir("predict", operation["run_id"])
                    / f"job-{index:04d}"
                )
                output_path = run_dir / "scores.parquet"
                jobs.append(
                    execution._runner().submit(
                        config=execution.config,
                        action="predict",
                        payload={
                            "test_path": normalize_dataset_path(test_path),
                            "artifact_path": str(artifact_path),
                            "scores_path": str(output_path),
                            "remote_layout": layout,
                            "remote_group_value": group_value,
                        },
                        run_dir=run_dir,
                    )
                )
            self.store.update_operation(operation, state="queued", jobs=jobs)
            return None
        except Exception as exc:
            self.store.update_operation(
                operation, state="failed", error=f"{type(exc).__name__}: {exc}"
            )
            raise

    def evaluate(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | CalibrationResult | ParquetPath | None = None,
        *,
        metric_names: tuple[str, ...],
        env_type: Literal["local", "osiris"] | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
    ) -> EvaluationResult | None:
        """Evaluate scores against the target stored in the test dataset."""
        self.hooks.require_fitted()
        if isinstance(scores, CalibrationResult) and self.config.task_name not in {
            "response",
            "uplift",
        }:
            msg = f"{type(self.config).__name__} does not accept CalibrationResult"
            raise ConfigError(msg)
        evaluation_kind = self._evaluation_kind(scores)
        runtime_config = self.hooks.resolve_config(
            env_type=env_type,
            device=device,
            environment=environment,
        )
        canonical, dataset_key = self.store.dataset(
            test_path, create=scores is not None
        )
        operation = self.store.start_operation(
            "evaluate",
            dataset_path=canonical,
            runtime=_runtime_metadata(
                runtime_config, evaluation_kind=evaluation_kind, metrics=metric_names
            ),
        )
        evaluation_dir = self.store.evaluation_dir(dataset_key)
        runtime_config = replace(runtime_config, output_dir=evaluation_dir)
        try:
            if scores is None:
                if runtime_config.env_type == "local":
                    scores = self.store.load_prediction(canonical)
                else:
                    scores, _ = self.store.prediction_path(canonical)
            artifact_path = (
                self.hooks.save(self.path / "artifact", overwrite=True)
                if runtime_config.env_type != "local"
                else None
            )
            execution = self.hooks.execution_view(
                ExecutionContext.from_config(runtime_config), prepare_backends=False
            )
            if runtime_config.env_type == "local":
                with self._track_local_operation(operation) as operation:
                    result = execution._runner().run_local(
                        config=execution.config,
                        action="evaluate",
                        callback=lambda: execution._execute_evaluate(
                            canonical, scores, evaluation_kind, metric_names
                        ),
                    )
                    result_path, merged = self.store.persist_evaluation(
                        canonical, result
                    )
                self.store.update_operation(
                    operation, state="succeeded", result_path=str(result_path)
                )
                return merged
            run_dir = (
                self.store.operation_dir("evaluate", operation["run_id"]) / "job-0000"
            )
            if isinstance(scores, PredictionResult | CalibrationResult):
                scores_path = run_dir / f"scores_{evaluation_kind}.parquet"
                scores_path.parent.mkdir(parents=True, exist_ok=True)
                scores.scores.write_parquet(scores_path)
            else:
                scores_path = Path(scores).expanduser().resolve()
            job = execution._runner().submit(
                config=execution.config,
                action="evaluate",
                payload={
                    "test_path": canonical,
                    "artifact_path": str(artifact_path),
                    "scores_path": str(scores_path),
                    "evaluation_kind": evaluation_kind,
                    "metrics": metric_names,
                },
                run_dir=run_dir,
            )
            self.store.update_operation(operation, state="queued", jobs=[job])
            return None
        except Exception as exc:
            self.store.update_operation(
                operation, state="failed", error=f"{type(exc).__name__}: {exc}"
            )
            raise

    def calibrate(
        self,
        test_path: ParquetPath,
        calibration_path: ParquetPath,
        *,
        calibration_strategy: CalibrationStrategy,
        env_type: Literal["local", "osiris"] | None,
        environment: EnvironmentConfig | Mapping[str, Any] | None,
    ) -> CalibrationResult | None:
        """Calibrate one stored test prediction from one stored calibration prediction."""
        canonical_test_path = normalize_dataset_path(test_path)
        canonical_calibration_path = normalize_dataset_path(calibration_path)
        if canonical_test_path == canonical_calibration_path:
            msg = "test_path and calibration_path must identify different datasets"
            raise ConfigError(msg)

        def required_prediction(
            path: ParquetPath, role: str
        ) -> tuple[Path, dict[str, Any]]:
            try:
                return self.store.prediction_path(path)
            except (ArtifactIntegrityError, RemoteExecutionError) as exc:
                msg = (
                    f"Calibration requires a succeeded predict({role}) operation for {role}="
                    f"{normalize_dataset_path(path)!r}"
                )
                raise type(exc)(msg) from exc

        test_scores_path, _ = required_prediction(test_path, "test_path")
        calibration_scores_path, calibration_metadata = required_prediction(
            calibration_path, "calibration_path"
        )
        canonical_calibration_path = str(calibration_metadata["test_path"])
        runtime_config = self.hooks.resolve_config(
            env_type=env_type, environment=environment
        )
        operation = self.store.start_operation(
            "calibrate",
            dataset_path=test_path,
            runtime=_runtime_metadata(
                runtime_config, calibration_path=canonical_calibration_path
            ),
        )
        try:
            execution = self.hooks.execution_view(
                ExecutionContext.from_config(runtime_config), prepare_backends=False
            )
            if runtime_config.env_type == "local":
                with self._track_local_operation(operation) as operation:
                    output = execution._runner().run_local(
                        config=execution.config,
                        action="calibrate",
                        callback=lambda: execution._execute_calibrate(
                            test_scores_path,
                            calibration_scores_path,
                            canonical_calibration_path,
                            calibration_strategy,
                        ),
                    )
                    result_path = self.store.persist_calibration(
                        test_path,
                        canonical_calibration_path,
                        output.result,
                        output.state,
                    )
                self.store.update_operation(
                    operation, state="succeeded", result_path=str(result_path)
                )
                return output.result

            jobs: list[dict[str, Any]] = []
            for index, layout in enumerate(
                execution._remote_calibration_parts(test_scores_path)
            ):
                run_dir = (
                    self.store.operation_dir("calibrate", operation["run_id"])
                    / f"job-{index:04d}"
                )
                output_path = run_dir / "scores.parquet"
                jobs.append(
                    execution._runner().submit(
                        config=execution.config,
                        action="calibrate",
                        payload={
                            "test_scores_path": str(test_scores_path),
                            "calibration_scores_path": str(calibration_scores_path),
                            "calibration_path": canonical_calibration_path,
                            "calibration_strategy": calibration_strategy,
                            "remote_layout": layout,
                            "output_path": str(output_path),
                        },
                        run_dir=run_dir,
                    )
                )
            self.store.update_operation(operation, state="queued", jobs=jobs)
            return None
        except Exception as exc:
            self.store.update_operation(
                operation, state="failed", error=f"{type(exc).__name__}: {exc}"
            )
            raise


def _every_part_has_a_model(jobs: Sequence[Mapping[str, Any]]) -> bool:
    """Whether each model part of a fan-out ended with at least one success.

    Only meaningful for a fan-out: with one job per part, a failure means that
    part has no model and the operation genuinely failed.
    """
    parts: dict[tuple[Any, ...], bool] = {}
    for job in jobs:
        if not job.get("trial"):
            return False
        key = (job.get("layout"), job.get("group_value"))
        parts[key] = parts.get(key, False) or job.get("state") == "succeeded"
    return bool(parts) and all(parts.values())
