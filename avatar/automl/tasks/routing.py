"""Global/per-group routing independent of task lifecycle and artifact IO."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from avatar.automl.data import ParquetSource
from avatar.automl.exceptions import ArtifactIntegrityError, ConfigError, SchemaError
from avatar.automl.execution import ExecutionContext
from avatar.automl.types import ParquetPath, PredictionResult

from .state import ModelEntry

_REMOTE_ROW_ID = "__fmlib_remote_row_id"


@dataclass(frozen=True)
class PredictionRouter:
    """Read-only routing settings and fitted model references for one operation."""

    context: ExecutionContext
    models: tuple[ModelEntry, ...]

    @property
    def effective_layout(self) -> str:
        requested = self.context.internal_config.resolved_model_layout
        if requested == "global_and_per_group" and any(
            item.single_group_global for item in self.models
        ):
            return "global"
        return requested

    @property
    def trained_group_values(self) -> set[Any]:
        """Return group values represented by per-group or collapsed models."""
        values = {
            item.group_value for item in self.models if item.layout == "per_group"
        }
        values.update(
            item.single_group_value
            for item in self.models
            if item.single_group_global and item.single_group_value is not None
        )
        return values

    def remote_parts(self, test_path: ParquetPath) -> list[tuple[str, Any | None]]:
        """Resolve independently runnable prediction branches without loading all columns."""
        requested = self.context.internal_config.resolved_model_layout
        layout = self.effective_layout
        parts: list[tuple[str, Any | None]] = []
        if layout in {"global", "global_and_per_group"}:
            parts.append(("global", None))
        if requested == "global":
            return parts

        group_column = self.context.config.group_column
        if group_column is None:
            msg = "model_layout='per_group'/'global_and_per_group' requires the configured group_column"
            raise SchemaError(msg)
        group_frame = ParquetSource.resolve(test_path).read((group_column,))
        if group_frame[group_column].null_count():
            msg = f"Group column {group_column!r} contains null values during group prediction"
            raise SchemaError(msg)
        incoming = group_frame[group_column].unique(maintain_order=True).to_list()
        trained = self.trained_group_values
        unknown = [value for value in incoming if value not in trained]
        if unknown:
            unknown_rows = group_frame.filter(
                pl.col(group_column).is_in(unknown)
            ).height
            msg = f"Unknown group values during prediction: {unknown} ({unknown_rows} rows)"
            raise SchemaError(msg)
        if layout == "global":
            return parts
        parts.extend(("per_group", value) for value in incoming if value in trained)
        if not parts:
            msg = "Prediction data contains no rows for trained group models"
            raise SchemaError(msg)
        return parts

    def validate_layout(self) -> None:
        """Ensure runtime routing can be satisfied by fitted model parts."""
        requested = self.effective_layout
        has_global = any(item.layout == "global" for item in self.models)
        has_per_group = any(item.layout == "per_group" for item in self.models)
        missing = []
        if requested in {"global", "global_and_per_group"} and not has_global:
            missing.append("global")
        if requested in {"per_group", "global_and_per_group"} and not has_per_group:
            missing.append("per_group")
        if missing:
            available = [
                name
                for name, present in (
                    ("global", has_global),
                    ("per_group", has_per_group),
                )
                if present
            ]
            msg = (
                f"Runtime model_layout={requested!r} requires unavailable model branches: {missing}; "
                f"artifact contains {available}"
            )
            raise ConfigError(msg)

    def input_frame(
        self,
        frame: pl.DataFrame,
        *,
        remote_group_value: Any | None,
        include_row_id: bool,
    ) -> pl.DataFrame:
        """Filter one remote group part while retaining the source row order."""
        if include_row_id:
            frame = frame.with_row_index(_REMOTE_ROW_ID)
        frame = self.filter_group(frame, remote_group_value)
        if remote_group_value is not None and frame.is_empty():
            msg = f"Prediction data has no rows for group value: {remote_group_value!r}"
            raise SchemaError(msg)
        return frame.with_row_index("__row_id")

    def filter_group(
        self, frame: pl.DataFrame, remote_group_value: Any | None
    ) -> pl.DataFrame:
        """Select one remote group part from a normalized prediction frame."""
        if remote_group_value is None:
            return frame
        group_column = self.context.internal_config.group_column
        if group_column is None or group_column not in frame.columns:
            msg = "Remote group prediction requires the configured group_column"
            raise SchemaError(msg)
        return frame.filter(pl.col(group_column) == remote_group_value)

    @staticmethod
    def attach_row_id(
        result: PredictionResult, frame: pl.DataFrame
    ) -> PredictionResult:
        """Attach an internal source row ID used only to merge remote job outputs."""
        if _REMOTE_ROW_ID not in frame.columns:
            return result
        values = frame.sort("__row_id")[_REMOTE_ROW_ID]
        scores = result.scores.with_columns(values)
        return PredictionResult(scores=scores, class_order=result.class_order)

    def layout_frames(
        self,
        frame: pl.DataFrame,
    ) -> list[tuple[str, pl.DataFrame]]:
        """Return independent global/per-group inputs for the runtime layout.

        ``global_and_per_group`` is a pair of inference branches. The global
        branch receives every row. The per-group branch receives rows for
        which a fitted group model exists. Per-group and combined layouts reject
        unknown groups before any branch is scored.
        """
        self.validate_layout()
        requested = self.context.internal_config.resolved_model_layout
        if requested == "global":
            return [("global", frame)]

        group_column = self.context.internal_config.group_column
        if group_column is None or group_column not in frame.columns:
            msg = "Group prediction requires the configured group_column"
            raise SchemaError(msg)
        if frame[group_column].null_count():
            msg = f"Group column {self.context.config.group_column!r} contains null values during group prediction"
            raise SchemaError(msg)

        group_models = self.trained_group_values
        incoming = frame[group_column].unique(maintain_order=True).to_list()
        unknown = [value for value in incoming if value not in group_models]
        if unknown:
            unknown_rows = frame.filter(pl.col(group_column).is_in(unknown)).height
            msg = f"Unknown group values during prediction: {unknown} ({unknown_rows} rows)"
            raise SchemaError(msg)

        group_frame = frame
        if "__row_id" in group_frame.columns:
            group_frame = group_frame.drop("__row_id").with_row_index("__row_id")
        if requested == "per_group":
            return [("per_group", group_frame)]
        if self.effective_layout == "global":
            return [("global", frame)]
        return [("global", frame), ("per_group", group_frame)]

    def combine(self, results: list[tuple[str, PredictionResult]]) -> PredictionResult:
        """Combine independent layout branches into one self-describing result."""
        if (
            len(results) == 1
            and self.context.internal_config.resolved_model_layout
            != "global_and_per_group"
        ):
            return results[0][1]
        class_orders = {result.class_order for _, result in results}
        if len(class_orders) != 1:
            msg = "Prediction branches returned inconsistent multiclass class orders"
            raise ArtifactIntegrityError(msg)
        scores = pl.concat(
            [
                result.scores.with_columns(pl.lit(layout).alias("model_layout"))
                for layout, result in results
            ],
            how="diagonal_relaxed",
        )
        return PredictionResult(scores=scores, class_order=results[0][1].class_order)

    def attach_group(
        self, result: PredictionResult, frame: pl.DataFrame
    ) -> PredictionResult:
        """Attach the row's group so long-form branch scores remain unambiguous."""
        group = self.context.internal_config.group_column
        if group is None or group not in frame.columns:
            return result
        external_group = self.context.config.group_column or group
        values = frame[group].alias(external_group)
        scores = result.scores.with_columns(values)
        return PredictionResult(scores=scores, class_order=result.class_order)

    def score(
        self,
        frame: pl.DataFrame,
        score: Callable[[pl.DataFrame, ModelEntry], np.ndarray],
    ) -> np.ndarray:
        """Score one selected global or per-group branch in input row order."""
        self.validate_layout()
        if "__row_id" not in frame.columns:
            frame = frame.with_row_index("__row_id")
        config = self.context.internal_config
        global_model = next(
            (item for item in self.models if item.layout == "global"), None
        )
        group_models = {
            item.group_value: item for item in self.models if item.layout == "per_group"
        }
        output: np.ndarray | None = None

        if config.resolved_model_layout == "global":
            return score(frame, global_model)

        if config.resolved_model_layout == "global_and_per_group":
            msg = "model_layout='global_and_per_group' must be expanded into independent prediction branches"
            raise ArtifactIntegrityError(msg)

        if config.resolved_model_layout == "per_group":
            group_column = config.group_column
            if group_column is None or group_column not in frame.columns:
                msg = "Group prediction requires the configured group_column"
                raise SchemaError(msg)
            if frame[group_column].null_count():
                msg = f"Group column {self.context.config.group_column!r} contains null values during group routing"
                raise SchemaError(msg)
            incoming = frame[group_column].unique(maintain_order=True).to_list()
            unknown = [value for value in incoming if value not in group_models]
            if unknown:
                msg = f"No group models are available for group values: {unknown}"
                raise SchemaError(msg)
            partitions = frame.partition_by(
                group_column, as_dict=True, maintain_order=True
            )
            for group_value in incoming:
                partition = partitions.pop((group_value,))
                indices = partition["__row_id"].to_numpy()
                item = group_models[group_value]
                partition_output = score(partition, item)
                if output is None:
                    output = np.empty(
                        (frame.height, *partition_output.shape[1:]), dtype=float
                    )
                output[indices] = partition_output
                del partition
        if output is None:
            msg = "Artifact contains no global or per-group model output"
            raise ArtifactIntegrityError(msg)
        return output
