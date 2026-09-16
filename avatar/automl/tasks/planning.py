"""Deterministic model parts shared by local training and remote submission."""

from dataclasses import dataclass
from typing import Any

import polars as pl

from avatar.automl.exceptions import SchemaError
from avatar.automl.execution import ExecutionContext


@dataclass(frozen=True)
class ModelPlan:
    parts: tuple[tuple[str, Any | None], ...]
    layout: str

    @classmethod
    def training(
        cls,
        context: ExecutionContext,
        train_frame: pl.DataFrame,
        valid_frame: pl.DataFrame,
        *,
        remote_layout: str | None = None,
        remote_group_value: Any | None = None,
    ) -> "ModelPlan":
        config = context.internal_config
        group_column = config.group_column

        effective_layout = remote_layout or config.resolved_model_layout
        if effective_layout in {"per_group", "global_and_per_group"}:
            if group_column is None or group_column not in train_frame.columns:
                msg = f"group_column={context.config.group_column!r}: required by model_layout={effective_layout!r}"
                raise SchemaError(msg)
            if group_column not in valid_frame.columns:
                msg = f"group_column={context.config.group_column!r}: required column is missing from validation data"
                raise SchemaError(msg)
            if (
                train_frame[group_column].null_count()
                or valid_frame[group_column].null_count()
            ):
                msg = f"Group column {context.config.group_column!r} cannot contain null values for group routing"
                raise SchemaError(msg)

        model_parts: list[tuple[str, Any | None]] = []
        if effective_layout in {"global", "global_and_per_group"}:
            model_parts.append(("global", None))

        if effective_layout in {"per_group", "global_and_per_group"}:
            if (
                remote_layout == "per_group"
                and remote_group_value
                not in train_frame[group_column].unique().to_list()
            ):
                msg = (
                    f"Training data has no rows for group value: {remote_group_value!r}"
                )
                raise SchemaError(msg)
            group_values = (
                [remote_group_value]
                if remote_layout == "per_group"
                else sorted(train_frame[group_column].unique().to_list(), key=str)
            )
            valid_values = set(valid_frame[group_column].unique().to_list())
            missing_valid = [
                value for value in group_values if value not in valid_values
            ]
            if missing_valid:
                msg = f"Validation data has no rows for trained groups: {missing_valid}"
                raise SchemaError(msg)
            model_parts.extend(
                ("per_group", group_value) for group_value in group_values
            )
        if (
            effective_layout == "global_and_per_group"
            and train_frame[group_column].n_unique() == 1
        ):
            return cls((("global", None),), "global")
        return cls(tuple(model_parts), effective_layout)

    @classmethod
    def remote_training(
        cls, context: ExecutionContext, frame: pl.DataFrame | None
    ) -> "ModelPlan":
        parts: list[tuple[str, Any | None]] = []
        if context.internal_config.resolved_model_layout in {
            "global",
            "global_and_per_group",
        }:
            parts.append(("global", None))
        if context.internal_config.resolved_model_layout in {
            "per_group",
            "global_and_per_group",
        }:
            group_column = context.internal_config.group_column
            if group_column is None:
                msg = (
                    f"group_column={context.config.group_column!r}: "
                    f"required by model_layout={context.internal_config.resolved_model_layout!r}"
                )
                raise SchemaError(msg)
            if group_column not in frame.columns:
                msg = f"group_column={context.config.group_column!r}: required column is missing from training data"
                raise SchemaError(msg)
            if frame[group_column].null_count():
                msg = f"Group column {context.config.group_column!r} cannot contain null values for group routing"
                raise SchemaError(msg)
            parts.extend(
                ("per_group", value)
                for value in sorted(frame[group_column].unique().to_list(), key=str)
            )
        if (
            context.internal_config.resolved_model_layout == "global_and_per_group"
            and frame[group_column].n_unique() == 1
        ):
            return cls((("global", None),), "global")
        return cls(tuple(parts), context.internal_config.resolved_model_layout)
