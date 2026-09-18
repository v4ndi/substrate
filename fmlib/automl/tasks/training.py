"""Training sequence returning fitted model parts for explicit adoption."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from fmlib.automl.backends import MATERIALIZED_BACKENDS
from fmlib.automl.data import CanonicalColumnMapper, ParquetSource
from fmlib.automl.execution import ExecutionContext, _freeze
from fmlib.automl.progress import log_progress
from fmlib.automl.types import ParquetPath, TrainingResult

from .planning import FrameGroupView, GroupView, ModelPlan, SourceGroupView
from .preparation import DataPreparation
from .state import ModelEntry, TrainingInput


def part_name(layout: str, group_value: Any | None = None) -> str:
    """Name of the model part for ``layout`` and, per group, ``group_value``."""
    return "global" if layout == "global" else f"per_group:{group_value}"


@dataclass(frozen=True)
class TrainingOutcome:
    """Models fitted by one training run, with their source manifests."""

    models: tuple[ModelEntry, ...]
    source_manifests: Mapping[str, tuple[dict[str, Any], ...]]
    result: TrainingResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_manifests", _freeze(self.source_manifests))


@dataclass(frozen=True)
class TrainingCoordinator:
    """Drives one training run: read, plan, fit every part, collect state."""

    context: ExecutionContext
    task_name: str
    prepare_state: Callable[[GroupView, GroupView], None]
    fit_one: Callable[..., ModelEntry]

    @property
    def materializes(self) -> bool:
        """Whether this backend family wants the splits read into memory."""
        return self.context.config.backend in MATERIALIZED_BACKENDS

    def execute(
        self,
        train_path: ParquetPath,
        valid_path: ParquetPath,
        *,
        remote_layout: str | None = None,
        remote_group_value: Any | None = None,
    ) -> TrainingOutcome:
        """Execute global/per-group training synchronously in the current process."""
        total_started = perf_counter()

        if (
            self.context.config.engine == "catboost"
            and self.context.config.resolved_device == "gpu"
        ):
            log_progress(
                "CatBoost device policy: training uses GPU; prediction always uses CPU because "
                "CatBoost GPU evaluation is not implemented for models with categorical features"
            )

        stage_started = perf_counter()
        log_progress("[train 1/5] resolving parquet sources")
        train_source = ParquetSource.resolve(train_path)
        valid_source = ParquetSource.resolve(valid_path)
        source_manifests = {
            "train": DataPreparation.source_manifest(train_source),
            "valid": DataPreparation.source_manifest(valid_source),
        }
        log_progress(
            "[train 1/5] completed duration_seconds=%.3f train_files=%d valid_files=%d",
            perf_counter() - stage_started,
            len(train_source.files),
            len(valid_source.files),
        )

        stage_started = perf_counter()
        if self.materializes:
            log_progress("[train 2/5] loading train and validation parquet")
            train_frame = DataPreparation(self.context).read_source(train_source)
            valid_frame = DataPreparation(self.context).read_source(valid_source)
            train_view = FrameGroupView(train_frame)
            valid_view = FrameGroupView(valid_frame)
            log_progress(
                "[train 2/5] completed duration_seconds=%.3f train_rows=%d valid_rows=%d train_columns=%d valid_columns=%d",
                perf_counter() - stage_started,
                train_frame.height,
                valid_frame.height,
                train_frame.width,
                valid_frame.width,
            )
        else:
            log_progress(
                "[train 2/5] backend=%s reads its own data; splits are not materialized",
                self.context.config.backend,
            )
            train_frame = valid_frame = None
            to_external = CanonicalColumnMapper.from_config(
                self.context.config
            ).to_external
            train_view = SourceGroupView(train_source, to_external)
            valid_view = SourceGroupView(valid_source, to_external)
            log_progress(
                "[train 2/5] completed duration_seconds=%.3f",
                perf_counter() - stage_started,
            )

        stage_started = perf_counter()
        log_progress(
            "[train 3/5] normalizing roles, validating routing, and building model plan"
        )
        self.prepare_state(train_view, valid_view)
        models = []
        plan = ModelPlan.training(
            self.context,
            train_view,
            valid_view,
            remote_layout=remote_layout,
            remote_group_value=remote_group_value,
        )
        model_parts = plan.parts
        effective_layout = plan.layout
        group_column = self.context.internal_config.group_column
        single_group_global = (
            self.context.internal_config.resolved_model_layout == "global_and_per_group"
            and group_column is not None
            and len(train_view.unique_values(group_column)[0]) == 1
        )
        if single_group_global:
            group_value = train_view.unique_values(group_column)[0][0]
            log_progress(
                "Requested model_layout='global_and_per_group' resolved for single group value %r: "
                "effective model_layout='global'; per-group branch is not created",
                group_value,
            )
        log_progress(
            "[train 3/5] completed duration_seconds=%.3f model_layout=%s models=%d",
            perf_counter() - stage_started,
            effective_layout,
            len(model_parts),
        )

        stage_started = perf_counter()
        log_progress("[train 4/5] training models models_total=%d", len(model_parts))
        train_parts = valid_parts = None
        for model_index, (layout, group_value) in enumerate(model_parts, start=1):
            model_started = perf_counter()
            model_name = part_name(layout, group_value)
            log_progress(
                "[train 4/5][model %d/%d] model=%s started",
                model_index,
                len(model_parts),
                model_name,
            )
            if not self.materializes:
                model_train = TrainingInput(train_source, group_value=group_value)
                model_valid = TrainingInput(valid_source, group_value=group_value)
            elif layout == "per_group":
                if train_parts is None:
                    train_parts = train_frame.partition_by(
                        group_column, as_dict=True, maintain_order=True
                    )
                    del train_frame
                    valid_parts = valid_frame.partition_by(
                        group_column, as_dict=True, maintain_order=True
                    )
                    del valid_frame
                model_train = TrainingInput(
                    train_source, train_parts.pop((group_value,)), group_value
                )
                model_valid = TrainingInput(
                    valid_source, valid_parts.pop((group_value,)), group_value
                )
            else:
                model_train = TrainingInput(train_source, train_frame)
                model_valid = TrainingInput(valid_source, valid_frame)
            models.append(
                self.fit_one(
                    model_train,
                    model_valid,
                    layout=layout,
                    group_value=group_value,
                )
            )
            if layout == "global" and single_group_global:
                group_value = train_view.unique_values(group_column)[0][0]
                models[-1] = replace(
                    models[-1], single_group_global=True, single_group_value=group_value
                )
            del model_train, model_valid
            log_progress(
                "[train 4/5][model %d/%d] model=%s completed duration_seconds=%.3f",
                model_index,
                len(model_parts),
                model_name,
                perf_counter() - model_started,
            )
        log_progress(
            "[train 4/5] completed duration_seconds=%.3f",
            perf_counter() - stage_started,
        )

        stage_started = perf_counter()
        log_progress("[train 5/5] assembling training result")
        canonical_features = tuple(
            dict.fromkeys(name for item in models for name in item.schema.feature_order)
        )
        external_names = CanonicalColumnMapper.from_config(
            self.context.config
        ).to_external
        feature_names = tuple(
            external_names.get(name, name) for name in canonical_features
        )
        best_params = {
            part_name(item.layout, item.group_value): item.best_params
            for item in models
        }
        validation_metrics = {
            part_name(item.layout, item.group_value): item.validation_metric
            for item in models
        }
        result = TrainingResult(
            best_params=best_params,
            validation_metrics=validation_metrics,
            feature_names=feature_names,
            backend_name=self.context.config.backend,
            engine_name=self.context.config.engine,
            task_name=self.task_name,
        )
        log_progress(
            "[train 5/5] completed duration_seconds=%.3f total_duration_seconds=%.3f",
            perf_counter() - stage_started,
            perf_counter() - total_started,
        )
        return TrainingOutcome(tuple(models), source_manifests, result)
