"""Deterministic model parts shared by local training and remote submission."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import polars as pl

from fmlib.automl.data import ParquetSource
from fmlib.automl.exceptions import SchemaError
from fmlib.automl.execution import ExecutionContext


class GroupView(Protocol):
    """The only two questions planning asks about a split.

    Stated as a protocol because the answers come from a materialized frame
    when the backend family needs one anyway, and from the parquet footers plus
    one column when it does not. Planning must not care which.
    """

    def has_column(self, column: str) -> bool:
        """Return whether the split has this column at all."""
        ...

    def unique_values(self, column: str) -> tuple[list[Any], bool]:
        """Return the distinct non-null values of a column, and whether nulls exist."""
        ...


@dataclass(frozen=True)
class FrameGroupView:
    """Answer from rows already in memory."""

    frame: pl.DataFrame

    def has_column(self, column: str) -> bool:
        return column in self.frame.columns

    def unique_values(self, column: str) -> tuple[list[Any], bool]:
        series = self.frame[column]
        return series.drop_nulls().unique().to_list(), bool(series.null_count())


@dataclass
class SourceGroupView:
    """Answer from the parquet source, reading one column and no other data.

    Planning asks in canonical names, while the files carry the names the user
    configured, so ``to_external`` translates on the way in -- the materialized
    path gets the same translation for free from ``DataPreparation``.

    Answers are cached because reading a column means touching every shard, and
    planning asks the same question more than once.
    """

    source: ParquetSource
    to_external: Mapping[str, str] = field(default_factory=dict)
    _answers: dict[str, tuple[list[Any], bool]] = field(default_factory=dict)

    def _physical(self, column: str) -> str:
        return self.to_external.get(column, column)

    def has_column(self, column: str) -> bool:
        return self._physical(column) in self.source.scan().collect_schema().names()

    def unique_values(self, column: str) -> tuple[list[Any], bool]:
        if column not in self._answers:
            values, has_nulls = self.source.unique_column_values(self._physical(column))
            self._answers[column] = (list(values), has_nulls)
        return self._answers[column]


@dataclass(frozen=True)
class ModelPlan:
    """Which models one operation trains or scores, and under which layout."""

    parts: tuple[tuple[str, Any | None], ...]
    layout: str

    @classmethod
    def training(
        cls,
        context: ExecutionContext,
        train: GroupView,
        valid: GroupView,
        *,
        remote_layout: str | None = None,
        remote_group_value: Any | None = None,
    ) -> "ModelPlan":
        config = context.internal_config
        group_column = config.group_column

        effective_layout = remote_layout or config.resolved_model_layout
        if effective_layout not in {"per_group", "global_and_per_group"}:
            return cls((("global", None),), effective_layout)

        if group_column is None or not train.has_column(group_column):
            msg = f"group_column={context.config.group_column!r}: required by model_layout={effective_layout!r}"
            raise SchemaError(msg)
        if not valid.has_column(group_column):
            msg = f"group_column={context.config.group_column!r}: required column is missing from validation data"
            raise SchemaError(msg)
        train_values, train_has_nulls = train.unique_values(group_column)
        valid_values, valid_has_nulls = valid.unique_values(group_column)
        if train_has_nulls or valid_has_nulls:
            msg = f"Group column {context.config.group_column!r} cannot contain null values for group routing"
            raise SchemaError(msg)

        model_parts: list[tuple[str, Any | None]] = []
        if effective_layout == "global_and_per_group":
            model_parts.append(("global", None))

        if remote_layout == "per_group" and remote_group_value not in train_values:
            msg = f"Training data has no rows for group value: {remote_group_value!r}"
            raise SchemaError(msg)
        group_values = (
            [remote_group_value]
            if remote_layout == "per_group"
            else sorted(train_values, key=str)
        )
        missing_valid = [
            value for value in group_values if value not in set(valid_values)
        ]
        if missing_valid:
            msg = f"Validation data has no rows for trained groups: {missing_valid}"
            raise SchemaError(msg)
        model_parts.extend(("per_group", group_value) for group_value in group_values)
        if effective_layout == "global_and_per_group" and len(train_values) == 1:
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
