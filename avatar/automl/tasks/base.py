"""Shared path-based lifecycle for boosting tasks."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import polars as pl

from avatar.automl.backends.boosting.interface import BoostingBackend
from avatar.automl.config.base import BaseTaskConfig, EnvironmentConfig
from avatar.automl.data import CanonicalColumnMapper, ParquetSource
from avatar.automl.environment import EnvironmentRunner
from avatar.automl.exceptions import (
    ArtifactIntegrityError,
    ConfigError,
    NotFittedError,
)
from avatar.automl.execution import ExecutionContext
from avatar.automl.lifecycle import AutoMLStore, read_json, write_json
from avatar.automl.metrics import resolve_evaluation_metrics
from avatar.automl.reporting import EvaluationData, export_evaluation
from avatar.automl.tasks.artifacts import ArtifactRepository, ArtifactState
from avatar.automl.tasks.artifacts import jsonable as _jsonable
from avatar.automl.tasks.operations import (
    OperationHooks,
    OperationRunner,
    RemoteTrainingParts,
)
from avatar.automl.tasks.planning import ModelPlan
from avatar.automl.tasks.prediction import execute_prediction
from avatar.automl.tasks.preparation import DataPreparation
from avatar.automl.tasks.routing import PredictionRouter
from avatar.automl.tasks.state import ModelEntry as _ModelEntry
from avatar.automl.tasks.training import TrainingCoordinator
from avatar.automl.types import (
    CalibrationResult,
    EvaluationKind,
    EvaluationResult,
    ParquetPath,
    PredictionResult,
    TrainingResult,
)

_BackendT = TypeVar("_BackendT", bound=BoostingBackend)


class BaseBoostingTask(ABC, Generic[_BackendT]):
    """Common lifecycle for independent boosting task facades.

    Input methods accept a parquet file or a directory containing parquet files.
    Directory discovery is recursive and reconstructs Hive ``key=value`` path
    segments as columns. ``train`` and ``valid`` are loaded into memory with
    Polars; the fitted model is not retrained on their union.

    Args:
        config: Validated task-specific boosting configuration.

    Attributes:
        config: Public configuration supplied by the caller.
    """

    _backend_class: type[_BackendT]
    _config_class: type[BaseTaskConfig]
    _task_name: str
    _artifact_directory: str

    def __init__(self, config: BaseTaskConfig, *, _entity_path: Path | None = None):
        """Initialize shared state from a validated task configuration.

        Args:
            config: Task-specific configuration with ``backend='boosting'``.

        Raises:
            ConfigError: If the configuration belongs to another task or a
                non-boosting backend is requested.
        """
        expected_config = type(self)._config_class
        if type(config) is not expected_config:
            msg = f"{type(self).__name__} requires {expected_config.__name__}; got {type(config).__name__}"
            raise ConfigError(msg)
        if config.backend != "boosting":
            msg = f"{type(self).__name__} currently implements only backend='boosting'"
            raise ConfigError(msg)
        self.config = config
        self._column_mapper = CanonicalColumnMapper.from_config(config)
        self._internal_config = self._column_mapper.normalize_config(config)
        self._models: list[_ModelEntry[_BackendT]] = []
        self._source_manifests: dict[str, tuple[dict[str, Any], ...]] = {}
        self._environment_runner: EnvironmentRunner | None = None
        if _entity_path is None:
            self._store = AutoMLStore.create(
                config.output_dir, self._task_name, _jsonable(asdict(config))
            )
        else:
            self._store = AutoMLStore(_entity_path)

    @property
    def id(self) -> str:
        """Return the persistent identifier of this AutoML lifecycle."""
        return self._store.automl_id

    @property
    def path(self) -> Path:
        """Return the persistent directory of this AutoML lifecycle."""
        return self._store.root

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the task contains a trained or loaded model.

        Returns:
            ``True`` when at least one global or per-group model is available.
        """
        return bool(self._models)

    @abstractmethod
    def _target(self, frame: pl.DataFrame) -> np.ndarray:
        """Validate and return the task-specific target vector."""

    def _prepare_training_state(
        self, train_frame: pl.DataFrame, valid_frame: pl.DataFrame
    ) -> None:
        """Resolve task-specific fitted state before model parts are trained."""
        return None

    def _context(self) -> ExecutionContext:
        return getattr(
            self, "_execution_context", None
        ) or ExecutionContext.from_config(self.config)

    def _data_preparation(self) -> DataPreparation:
        return DataPreparation(self._context())

    def _prediction_router(self) -> PredictionRouter:
        return PredictionRouter(self._context(), tuple(self._models))

    def _read(self, path: ParquetPath) -> pl.DataFrame:
        """Read parquet shards and map configured role aliases to internal names."""
        return self._data_preparation().read(path)

    @staticmethod
    def _model_name(layout: str, group_value: Any | None = None) -> str:
        """Return a stable public key for global or per-group model results."""
        return "global" if layout == "global" else f"per_group:{group_value}"

    def _copy_task_state(self, other: BaseBoostingTask) -> None:
        """Copy concrete task state from one complete loaded artifact."""
        return None

    def _merge_task_state(self, other: BaseBoostingTask) -> None:
        """Merge concrete task state from one independently trained model part."""
        self._copy_task_state(other)

    def _task_manifest(self) -> Mapping[str, Any]:
        """Return concrete task state for the JSON artifact manifest."""
        return {}

    def _restore_task_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Restore concrete task state from a JSON artifact manifest."""
        return None

    def _reset_task_state(self) -> None:
        """Clear concrete fitted state after an unsuccessful training transaction."""
        return None

    def _rollback_failed_training(self, artifact_path: Path) -> None:
        """Remove every in-memory and persisted result of a failed training run."""
        self._models = []
        self._source_manifests = {}
        self._reset_task_state()
        training_result = self.path / "training_result.json"
        training_result.unlink(missing_ok=True)
        if artifact_path.exists():
            shutil.rmtree(artifact_path)
        entity = self._store.entity
        write_json(self.path / "entity.json", entity | {"artifact_path": None})

    @abstractmethod
    def _fit_one(
        self,
        train_frame: pl.DataFrame,
        valid_frame: pl.DataFrame,
        *,
        layout: str,
        group_value: Any | None = None,
    ) -> _ModelEntry[_BackendT]:
        """Train one global/per-group part using the task's training strategy."""

    def _execute_train(
        self,
        train_path: ParquetPath,
        valid_path: ParquetPath,
        *,
        remote_layout: str | None = None,
        remote_group_value: Any | None = None,
    ) -> TrainingResult:
        """Execute global/per-group training synchronously in the current process."""
        outcome = TrainingCoordinator(
            self._context(),
            self._task_name,
            self._prepare_training_state,
            self._fit_one,
        ).execute(
            train_path,
            valid_path,
            remote_layout=remote_layout,
            remote_group_value=remote_group_value,
        )
        self._models = list(outcome.models)
        self._source_manifests = dict(outcome.source_manifests)
        return outcome.result

    def _remote_training_parts(
        self, train_path: ParquetPath
    ) -> list[tuple[str, Any | None]]:
        """Resolve the submission plan without reading unused global data."""
        frame = None
        if self._internal_config.resolved_model_layout in {
            "per_group",
            "global_and_per_group",
        }:
            values, has_nulls = ParquetSource.resolve(train_path).unique_column_values(
                self.config.group_column
            )
            frame = pl.DataFrame({
                self._internal_config.group_column: [
                    *values,
                    *([None] if has_nulls else []),
                ]
            })
        return list(ModelPlan.remote_training(self._context(), frame).parts)

    def _remote_prediction_parts(
        self, test_path: ParquetPath
    ) -> list[tuple[str, Any | None]]:
        """Resolve independently runnable prediction branches without loading all columns."""
        return self._prediction_router().remote_parts(test_path)

    def _select_remote_prediction_part(
        self, layout: str, group_value: Any | None
    ) -> None:
        """Keep exactly the fitted model used by one remote prediction job."""
        selected = [
            item
            for item in self._models
            if item.layout == layout
            and (layout == "global" or item.group_value == group_value)
        ]
        if len(selected) != 1:
            msg = f"Remote prediction part {(layout, group_value)!r} resolved to {len(selected)} fitted models"
            raise ArtifactIntegrityError(msg)
        self._models = selected

    def _runner(self) -> EnvironmentRunner:
        """Return the task runner, allowing a fake to be injected in unit tests."""
        if self._environment_runner is None:
            self._environment_runner = EnvironmentRunner()
        return self._environment_runner

    def _operation_runner(self) -> OperationRunner:
        hooks = OperationHooks(
            resolve_config=self._resolve_runtime_config,
            execution_view=self._execution_view,
            save=self.save,
            adopt=self._adopt,
            rollback_training=self._rollback_failed_training,
            set_artifact=self._set_entity_artifact,
            assemble_training=self._assemble_remote_training,
            require_fitted=self._require_fitted,
            storage_frame=self._prediction_storage_frame,
        )
        return OperationRunner(
            self._context().config, self._store, self._runner(), hooks, self.is_fitted
        )

    def train(
        self,
        train_path: ParquetPath,
        valid_path: ParquetPath,
        *,
        env_type: Literal["local", "osiris"] | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
    ) -> TrainingResult | None:
        """Train a model and optionally run Optuna hyperparameter search.

        ``train_path`` and ``valid_path`` may be parquet files or directory roots,
        including Hive split roots such as ``.../split_type=train``. Validation is
        mandatory and controls early stopping and trial selection. The best fitted
        trial is returned directly without an additional training pass.

        Args:
            train_path: Parquet file or directory containing the training split.
            valid_path: Parquet file or directory containing the validation split.
            env_type: Execution-environment override for this training operation.
            device: Local CPU/GPU override. Remote environments reject it.
            environment: Execution settings override for this operation only.

        Returns:
            A :class:`TrainingResult` for synchronous local execution. Remote
            execution returns ``None`` and is inspected through :meth:`status`.

        Raises:
            SchemaError: If required columns, group partitions, or feature
                types are invalid.
        """
        return self._operation_runner().train(
            train_path,
            valid_path,
            env_type=env_type,
            device=device,
            environment=environment,
        )

    def _set_entity_artifact(self, artifact_path: Path) -> None:
        """Record the internal fitted artifact used to restore this entity."""
        entity = self._store.entity
        write_json(
            self.path / "entity.json",
            entity | {"artifact_path": str(artifact_path.resolve())},
        )

    def training_result(self) -> TrainingResult:
        """Load the completed training summary for this entity."""
        return self._store.load_training()

    def _adopt(self, restored: BaseBoostingTask) -> None:
        """Adopt fitted model and task-specific state from an artifact instance."""
        self._models = restored._models
        self._source_manifests = restored._source_manifests
        self._copy_task_state(restored)

    def _assemble_remote_training(
        self, request: RemoteTrainingParts
    ) -> tuple[str, ...]:
        """Adopt complete worker artifacts and merge formulation-specific state."""
        if request.artifact_path.exists():
            self._restore_artifact(request.artifact_path)
        else:
            self._models = []
            self._source_manifests = {}
            repository = self._artifact_repository()
            for part_root in request.part_paths:
                part_config = repository.read_config(
                    part_root, type(self)._config_class
                )
                part = type(self)(part_config, _entity_path=self.path)
                part._restore_artifact(part_root)
                self._models.extend(part._models)
                self._merge_task_state(part)
                self._source_manifests.update(part._source_manifests)
            self.save(request.artifact_path)
        return tuple(
            self._column_mapper.to_external.get(name, name)
            for name in dict.fromkeys(
                name for item in self._models for name in item.schema.feature_order
            )
        )

    def status(self, *, wait: bool = False) -> pl.DataFrame:
        """Inspect all local and remote operations owned by this task.

        Args:
            wait: Poll until every currently active remote operation reaches a
                terminal state. Failed jobs raise after their complete available
                scheduler and shared-filesystem diagnostics are logged.

        Returns:
            One table containing action, dataset path, state, run and scheduler IDs.
        """
        return self._operation_runner().status(wait=wait)

    def _require_fitted(self) -> None:
        """Raise when an operation requires a fitted model."""
        if not self.is_fitted:
            msg = f"{type(self).__name__} is not fitted; call train() or load() first"
            raise NotFittedError(msg)

    def _resolve_runtime_config(
        self,
        *,
        env_type: Literal["local", "osiris"] | None,
        device: Literal["cpu", "gpu"] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
        model_layout: Literal["global", "per_group", "global_and_per_group"]
        | None = None,
        output_dir: str | Path | None = None,
    ) -> BaseTaskConfig:
        """Build a validated inference config without changing training state."""
        updates: dict[str, Any] = {}
        if env_type is not None:
            updates["env_type"] = env_type
            if env_type != "local" or (
                device is None and self.config.env_type != "local"
            ):
                updates["device"] = "gpu"
        effective_env = env_type or self.config.env_type
        if device is not None:
            if effective_env != "local":
                msg = "device can be passed only together with env_type='local'"
                raise ConfigError(msg)
            updates["device"] = device
        if environment is not None:
            updates["environment"] = environment
        if model_layout is not None:
            updates["model_layout"] = model_layout
        if output_dir is not None:
            updates["output_dir"] = output_dir
        return replace(self.config, **updates)

    def _execution_view(
        self, context: ExecutionContext, *, prepare_backends: bool = False
    ):
        """Bind operation settings and scratch state without mutating the entity.

        This shallow view shares fitted data and the persistent store. Only
        explicit training adoption updates the entity's fitted state.
        """
        execution = copy(self)
        execution._execution_context = context
        execution._artifact_config = getattr(self, "_artifact_config", self.config)
        execution.config = context.config
        execution._internal_config = context.internal_config
        execution._models = [
            replace(
                item, backend=item.backend.for_execution(context.config.resolved_device)
            )
            if prepare_backends
            else item
            for item in self._models
        ]
        execution._source_manifests = dict(self._source_manifests)
        return execution

    def _validate_prediction_layout(self) -> None:
        """Ensure runtime routing can be satisfied by fitted model parts."""
        return self._prediction_router().validate_layout()

    @abstractmethod
    def _predict_entry(
        self, frame: pl.DataFrame, item: _ModelEntry[_BackendT]
    ) -> np.ndarray:
        """Prepare task-specific roles and score one fitted model part."""

    @abstractmethod
    def _normalize_prediction_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Normalize the task's feature roles for prediction and reference data."""

    @abstractmethod
    def _score_prediction_branch(
        self,
        frame: pl.DataFrame,
    ) -> np.ndarray:
        """Return validated raw branch scores."""

    @abstractmethod
    def _prediction_result(
        self, frame: pl.DataFrame, raw: np.ndarray
    ) -> PredictionResult:
        """Construct the public raw-score table."""

    def _execute_predict(
        self,
        test_path: ParquetPath,
        *,
        remote_group_value: Any | None = None,
        include_row_id: bool = False,
        include_group: bool = False,
    ) -> PredictionResult:
        """Run the common prediction coordinator on this operation's context."""
        return execute_prediction(
            self,
            test_path,
            remote_group_value=remote_group_value,
            include_row_id=include_row_id,
            include_group=include_group,
        )

    def _prediction_input_frame(
        self,
        frame: pl.DataFrame,
        *,
        remote_group_value: Any | None,
        include_row_id: bool,
    ) -> pl.DataFrame:
        """Filter one remote group part while retaining the source row order."""
        return self._prediction_router().input_frame(
            frame, remote_group_value=remote_group_value, include_row_id=include_row_id
        )

    def _filter_prediction_group(
        self, frame: pl.DataFrame, remote_group_value: Any | None
    ) -> pl.DataFrame:
        """Select one remote group part from a normalized prediction frame."""
        return self._prediction_router().filter_group(frame, remote_group_value)

    @staticmethod
    def _with_prediction_row_id(
        result: PredictionResult, frame: pl.DataFrame
    ) -> PredictionResult:
        """Attach an internal source row ID used only to merge remote job outputs."""
        return PredictionRouter.attach_row_id(result, frame)

    def _with_model_layout(self, layout: Literal["global", "per_group"]):
        """Create branch-local settings."""
        context = ExecutionContext.from_config(self.config)
        if self.config.group_column is not None:
            context = context.derive(model_layout=layout)
        return self._execution_view(context)

    def _prediction_layout_frames(
        self,
        frame: pl.DataFrame,
    ) -> list[tuple[str, pl.DataFrame]]:
        """Return independent global/per-group inputs for the requested runtime layout.

        ``global_and_per_group`` is a pair of inference branches rather than row routing. The
        global branch receives every row. The per-group branch receives rows for
        which a fitted group model exists; unknown groups are reported and
        omitted only from that branch.
        """
        return self._prediction_router().layout_frames(frame)

    def _combine_layout_predictions(
        self, results: list[tuple[str, PredictionResult]]
    ) -> PredictionResult:
        """Combine independent global and per-group branches into one self-describing result."""
        return self._prediction_router().combine(results)

    def _with_prediction_group(
        self, result: PredictionResult, frame: pl.DataFrame
    ) -> PredictionResult:
        """Attach the row's group so long-form branch scores remain unambiguous."""
        return self._prediction_router().attach_group(result, frame)

    def _predict_frame(self, frame: pl.DataFrame) -> np.ndarray:
        """Score one selected global or per-group branch in input row order."""
        return self._prediction_router().score(frame, self._predict_entry)

    @property
    def _prediction_device(self) -> Literal["cpu", "gpu"]:
        """Return the device actually used by native prediction calls."""
        if self.config.engine == "catboost":
            return "cpu"
        return self.config.resolved_device

    @staticmethod
    def _prediction_storage_frame(prediction: PredictionResult) -> pl.DataFrame:
        """Return the public raw score columns for remote transport."""
        return prediction.scores

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
        """Return model scores for every row under ``test_path``.

        The result contains client ID, the configured date column when
        present, and ``score``. No thresholded decision column is produced.
        With ``model_layout="global_and_per_group"``, global and per-group branches are returned
        as separate rows identified by ``model_layout``. Per-group and combined
        layouts reject rows whose group was not present during training.

        Args:
            test_path: Parquet file or directory containing rows to score.
            env_type: Optional inference environment override.
            device: Local inference device. Remote environments reject it.
            environment: Execution settings override for this operation only.
            model_layout: Optional routing layout supported by the fitted artifact.

        Returns:
            Row-level scores for local execution. Remote execution returns
            ``None``; use :meth:`status` and :meth:`load_prediction`.

        Raises:
            NotFittedError: If the task has not been trained or loaded.
            SchemaError: If prediction data is incompatible with the artifact.
        """
        return self._operation_runner().predict(
            test_path,
            env_type=env_type,
            device=device,
            environment=environment,
            model_layout=model_layout,
        )

    def load_prediction(self, test_path: ParquetPath) -> PredictionResult:
        """Load scores belonging exactly to the canonical ``test_path``."""
        return self._store.load_prediction(test_path)

    @abstractmethod
    def _evaluate_data(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | CalibrationResult | ParquetPath,
        evaluation_kind: EvaluationKind,
        metric_names: tuple[str, ...],
    ) -> EvaluationData:
        """Compute task metrics, tables and plot inputs without exporting files."""

    def _execute_evaluate(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | CalibrationResult | ParquetPath,
        evaluation_kind: EvaluationKind,
        metric_names: tuple[str, ...],
    ) -> EvaluationResult:
        """Compute evaluation data and export it in the operation directory."""
        data = self._evaluate_data(test_path, scores, evaluation_kind, metric_names)
        return export_evaluation(data, Path(self.config.output_dir))

    def evaluate(
        self,
        test_path: ParquetPath,
        scores: PredictionResult | CalibrationResult | ParquetPath | None = None,
        *,
        metrics: Sequence[str] | None = None,
        env_type: Literal["local", "osiris"] | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
    ) -> EvaluationResult | None:
        """Evaluate scores against the target stored in the test dataset.

        Args:
            test_path: Parquet file or directory containing target and role columns.
            scores: :class:`PredictionResult`, a task-supported
                :class:`CalibrationResult`, a registered prediction/calibration
                parquet path, or ``None`` to use the completed prediction stored
                for this exact ``test_path``.
            metrics: Registered metric names. ``None`` uses task defaults.
            env_type: Optional evaluation environment override.
            device: Local evaluation device. Remote environments reject it.
            environment: Execution settings override for this operation only.

        Returns:
            Raw/calibrated metrics and reports accumulated for this test path
            during local execution. Remote execution returns ``None``; use
            :meth:`status` and :meth:`load_evaluation`.

        Raises:
            NotFittedError: If the task has not been trained or loaded.
            ConfigError: If the score kind cannot be determined or calibration
                is unsupported by this task.
            SchemaError: If targets and scores cannot be aligned.
        """
        metric_names = tuple(
            metric.name
            for metric in resolve_evaluation_metrics(metrics, self._task_name)
        )
        return self._operation_runner().evaluate(
            test_path,
            scores,
            metric_names=metric_names,
            env_type=env_type,
            device=device,
            environment=environment,
        )

    def load_evaluation(self, test_path: ParquetPath) -> EvaluationResult:
        """Load the persisted evaluation summary for exactly ``test_path``."""
        return self._store.load_evaluation(test_path)

    def save(self, path: str | Path | None = None, *, overwrite: bool = False) -> Path:
        """Atomically save config, schema, metadata and native model files.

        When ``path`` is omitted, the artifact is written below
        ``<output_dir>/artifacts/<task-specific directory>``. An existing destination is
        rejected unless ``overwrite=True`` is passed explicitly.

        Args:
            path: Destination artifact directory. Defaults to the configured
                task artifact directory below ``output_dir``.
            overwrite: Explicitly replace an existing destination atomically when ``True``.

        Returns:
            Absolute path to the saved artifact directory.

        Raises:
            NotFittedError: If the task contains no fitted model.
            ArtifactError: If the target exists and overwriting is disabled.
        """
        self._require_fitted()
        state = ArtifactState(
            getattr(self, "_artifact_config", self.config),
            tuple(self._models),
            dict(self._source_manifests),
            self._task_manifest(),
        )
        return self._artifact_repository().save(state, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> BaseBoostingTask:
        """Restore a complete AutoML entity or a standalone model artifact.

        Args:
            path: Entity directory containing ``entity.json`` or an artifact
                directory created by :meth:`save`.

        Returns:
            A fitted instance of the concrete task class.

        Raises:
            ArtifactIntegrityError: If required files, task kind, schemas, or
                model metadata are inconsistent.
        """
        root = Path(path).expanduser().resolve()
        entity_path = root / "entity.json"
        if entity_path.is_file():
            try:
                entity = read_json(entity_path)
            except Exception as exc:
                msg = f"AutoML entity metadata is invalid: {entity_path}: {exc}"
                raise ArtifactIntegrityError(msg) from exc
            if entity.get("task") != cls._task_name:
                msg = f"{cls.__name__} cannot load task={entity.get('task')!r}"
                raise ArtifactIntegrityError(msg)
            try:
                config = cls._config_class.from_mapping(entity["config"])
            except Exception as exc:
                msg = f"AutoML entity config is invalid: {entity_path}: {exc}"
                raise ArtifactIntegrityError(msg) from exc
            artifact = entity.get("artifact_path")
            state = None
            if artifact:
                repository = cls._repository()
                artifact_root = Path(artifact)
                artifact_config = repository.read_config(
                    artifact_root, cls._config_class
                )
                cls._validate_entity_artifact_config(config, artifact_config)
                state = repository.restore(
                    artifact_root,
                    artifact_config,
                    require_complete=True,
                    runtime_device=config.resolved_device,
                )
            task = cls(config, _entity_path=root)
            if state is not None:
                try:
                    task._adopt_artifact_state(state)
                except ArtifactIntegrityError:
                    raise
                except Exception as exc:
                    msg = f"Artifact task metadata cannot be restored: {exc}"
                    raise ArtifactIntegrityError(msg) from exc
            task._store.reconcile_local_operations()
            return task
        task = cls._load_artifact(root)
        try:
            task._set_entity_artifact(root)
        except Exception:
            shutil.rmtree(task.path, ignore_errors=True)
            raise
        return task

    @classmethod
    def _load_artifact(cls, root: Path) -> BaseBoostingTask:
        """Load a standalone artifact into a fresh AutoML entity."""
        repository = cls._repository()
        config = repository.read_config(root, cls._config_class)
        runtime_device = config.resolved_device
        if runtime_device not in {"cpu", "gpu"}:
            msg = f"Unsupported runtime device: {runtime_device!r}"
            raise ArtifactIntegrityError(msg)
        state = repository.restore(root, config, require_complete=True)
        task = cls(config)
        try:
            task._adopt_artifact_state(state)
            return task
        except Exception as exc:
            shutil.rmtree(task.path, ignore_errors=True)
            if isinstance(exc, ArtifactIntegrityError):
                raise
            msg = f"Artifact task metadata cannot be restored: {exc}"
            raise ArtifactIntegrityError(msg) from exc

    def _restore_artifact(
        self, root: Path, *, manifest: Mapping[str, Any] | None = None
    ) -> None:
        """Apply artifact values at the explicit fitted-state boundary."""
        repository = self._artifact_repository()
        artifact_config = repository.read_config(root, type(self)._config_class)
        state = repository.restore(
            root,
            artifact_config,
            manifest=manifest,
            runtime_device=self.config.resolved_device,
        )
        self._adopt_artifact_state(state)

    @staticmethod
    def _validate_entity_artifact_config(
        entity_config: BaseTaskConfig, artifact_config: BaseTaskConfig
    ) -> None:
        """Require an entity and artifact to agree on model interpretation."""
        entity_payload = _jsonable(asdict(entity_config))
        artifact_payload = _jsonable(asdict(artifact_config))
        comparable_fields = {
            "backend",
            "engine",
            "target_column",
            "client_id_column",
            "date_column",
            "group_column",
            "treatment_column",
            "inverse_treatment",
            "categorical_columns",
            "numerical_columns",
            "hidden_state_columns",
            "model_layout",
        }
        if (
            "estimate_propensity" in entity_payload
            or "estimate_propensity" in artifact_payload
        ):
            comparable_fields.add("estimate_propensity")
        differing_fields = sorted(
            field
            for field in comparable_fields
            if entity_payload.get(field) != artifact_payload.get(field)
        )
        if not differing_fields:
            return
        msg = f"Entity config does not match artifact config for fields: {differing_fields}"
        raise ArtifactIntegrityError(msg)

    def _adopt_artifact_state(self, state: ArtifactState) -> None:
        """Adopt one completely validated and restored artifact state."""
        self._restore_task_manifest(state.task_manifest)
        self._models = list(state.models)
        self._source_manifests = dict(state.source_manifests)

    @classmethod
    def _repository(cls) -> ArtifactRepository:
        return ArtifactRepository(
            cls._task_name, cls._artifact_directory, cls._backend_class
        )

    def _artifact_repository(self) -> ArtifactRepository:
        return type(self)._repository()
