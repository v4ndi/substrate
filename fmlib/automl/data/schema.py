"""Feature-schema validation and deterministic data preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import polars as pl

from fmlib.automl.config.base import BaseTaskConfig
from fmlib.automl.exceptions import SchemaError


@dataclass(frozen=True)
class FeatureSchema:
    """Describe ordered features and role metadata required for inference.

    Attributes:
        categorical: Ordered categorical estimator inputs.
        numerical: Ordered numerical estimator inputs.
        feature_order: Complete estimator input order.
        dtypes: Polars dtype strings recorded at fit time.
        target_column: Canonical target-column name.
        client_id_column: Canonical identifier-column name.
        treatment_column: Canonical optional treatment-column name.
        group_column: Canonical optional group-column name.
        hidden_states: Width of every hidden-state column kept as a vector,
            ``{column: width}``. Empty when hidden states were expanded into
            scalar features instead, which is what boosting does.
    """

    categorical: tuple[str, ...]
    numerical: tuple[str, ...]
    feature_order: tuple[str, ...]
    dtypes: Mapping[str, str]
    target_column: str
    client_id_column: str
    treatment_column: str | None
    group_column: str | None
    hidden_states: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema into JSON-compatible values.

        Returns:
            A mapping suitable for an artifact manifest.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureSchema:
        """Restore a feature schema from serialized values.

        Args:
            value: Mapping previously produced by :meth:`to_dict`.

        Returns:
            Restored immutable feature schema.
        """
        payload = dict(value)
        for name in ("categorical", "numerical", "feature_order"):
            payload[name] = tuple(payload[name])
        # Artifacts written before hidden states could stay vectors have no
        # such key, and for them the empty default is the correct answer.
        payload["hidden_states"] = dict(payload.get("hidden_states") or {})
        return cls(**payload)


@dataclass(frozen=True)
class PreparedData:
    """Pair a normalized frame with the schema used by an estimator.

    Attributes:
        frame: Normalized Polars DataFrame.
        schema: Ordered feature schema describing ``frame``.
    """

    frame: pl.DataFrame
    schema: FeatureSchema


def normalize_date(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Convert a configured date-like column to ``Date``.

    Args:
        frame: Input Polars DataFrame.
        column: Configured date column name.

    Returns:
        The original frame when the column is absent or already ``Date``;
        otherwise a frame with the normalized column.

    Raises:
        SchemaError: If the dtype or a non-null string value cannot be parsed.
    """
    if column not in frame.columns:
        return frame
    dtype = frame.schema[column]
    if dtype == pl.Date:
        return frame
    if isinstance(dtype, pl.Datetime):
        return frame.with_columns(pl.col(column).dt.date())
    if dtype == pl.String:
        parsed_column = "__fmlib_parsed_date"
        value = pl.col(column)
        parsed = pl.coalesce(
            value.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            value.str.strptime(pl.Date, "%Y%m%d", strict=False),
            (value + "-01").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            (value + "01").str.strptime(pl.Date, "%Y%m%d", strict=False),
        )
        checked = frame.with_columns(parsed.alias(parsed_column))
        invalid = (
            checked.filter(
                pl.col(column).is_not_null() & pl.col(parsed_column).is_null()
            )
            .select(column)
            .head(5)
        )
        if invalid.height:
            msg = f"Column {column!r} contains invalid date values: {invalid.to_series().to_list()}"
            raise SchemaError(msg)
        return checked.drop(column).rename({parsed_column: column})
    msg = f"Column {column!r} must be Date, Datetime or a date string; got {dtype}"
    raise SchemaError(msg)


def normalize_optional_binary_treatment(
    frame: pl.DataFrame,
    column: str | None,
    *,
    public_name: str | None,
    inverse: bool,
) -> pl.DataFrame:
    """Validate an optional binary treatment role and apply its convention.

    Args:
        frame: Input Polars DataFrame.
        column: Internal treatment-column name, or ``None``.
        public_name: User-visible name included in validation errors.
        inverse: Replace each binary value ``x`` with ``1 - x`` when ``True``.

    Returns:
        A frame containing normalized ``Int8`` treatment values, or the original
        frame when the optional role is absent.

    Raises:
        SchemaError: If treatment contains nulls or values other than zero and one.
    """
    if column is None or column not in frame.columns:
        return frame
    treatment = frame[column]
    if treatment.null_count():
        msg = f"Treatment column {public_name!r} contains null values"
        raise SchemaError(msg)
    values = set(treatment.unique().to_list())
    if not values <= {0, 1}:
        msg = (
            f"Treatment column must contain only 0 and 1; got {sorted(values, key=str)}"
        )
        raise SchemaError(msg)
    expression = pl.col(column).cast(pl.Int8)
    if inverse:
        expression = 1 - expression
    return frame.with_columns(expression.alias(column))


def _normalize_hidden_states(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    expected: Mapping[str, int] | None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Validate numeric embedding containers and resolve their widths.

    Hidden states may be represented by Polars ``Array`` or ``List`` columns
    with ``Float32`` or ``Float64`` elements. Lists must have one fixed length
    across the frame. Values are always normalized to ``Float32`` so training
    and inference use the same estimator schema.

    Returns:
        The frame with every hidden-state column cast to ``List(Float32)``,
        and the observed width of each one.
    """
    dimensions: dict[str, int] = {}
    for column in columns:
        if column not in frame.columns:
            msg = f"Missing hidden-state column: {column!r}"
            raise SchemaError(msg)
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Array | pl.List) or dtype.inner not in {
            pl.Float32,
            pl.Float64,
        }:
            msg = f"Hidden-state column {column!r} must be List or Array with Float32/Float64 elements; got {dtype}"
            raise SchemaError(msg)
        if frame.select(pl.col(column).is_null().any()).item():
            msg = f"Hidden-state column {column!r} contains null embeddings"
            raise SchemaError(msg)

        as_list = (
            pl.col(column).arr.to_list()
            if isinstance(dtype, pl.Array)
            else pl.col(column)
        )
        frame = frame.with_columns(as_list.cast(pl.List(pl.Float32)).alias(column))
        if frame.select(
            pl.col(column).list.eval(pl.element().is_null()).list.any().any()
        ).item():
            msg = f"Hidden-state column {column!r} contains null elements"
            raise SchemaError(msg)

        observed_dimensions = (
            frame.select(pl.col(column).list.len().unique()).to_series().to_list()
        )
        if len(observed_dimensions) != 1 or observed_dimensions[0] == 0:
            msg = f"Hidden-state column {column!r} must have one non-zero fixed dimension; got {observed_dimensions}"
            raise SchemaError(msg)
        dimension = observed_dimensions[0]
        if expected is not None and expected.get(column) != dimension:
            msg = f"Hidden-state dimension mismatch for {column!r}: expected {expected.get(column)}, got {dimension}"
            raise SchemaError(msg)
        dimensions[column] = dimension
    return frame, dimensions


def _expand_hidden_states(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    expected: Mapping[str, int] | None,
) -> tuple[pl.DataFrame, tuple[str, ...], dict[str, int]]:
    """Expand validated embedding columns into one scalar feature per coordinate.

    This is the only shape a gradient-boosted tree can consume. It is the wrong
    shape for a network -- the embedding would dominate attention by column
    count, each coordinate would be standardized on its own, destroying the
    geometry the producing model made, and each would get its own input
    projection -- so the TabNN path keeps the vector instead.
    """
    frame, dimensions = _normalize_hidden_states(frame, columns, expected)
    expanded: list[str] = []
    for column, dimension in dimensions.items():
        names = tuple(f"{column}__{index}" for index in range(dimension))
        frame = frame.with_columns([
            pl.col(column).list.get(index).alias(name)
            for index, name in enumerate(names)
        ]).drop(column)
        expanded.extend(names)
    return frame, tuple(expanded), dimensions


def prepare_data(
    frame: pl.DataFrame,
    config: BaseTaskConfig,
    *,
    require_target: bool,
    fitted_schema: FeatureSchema | None = None,
    hidden_dimensions: Mapping[str, int] | None = None,
    categorical_role_columns: Sequence[str | None] = (),
    excluded_feature_columns: Sequence[str] = (),
    public_column_names: Mapping[str, str] | None = None,
    expand_hidden_states: bool = True,
) -> tuple[PreparedData, dict[str, int]]:
    """Validate and normalize a frame for training, inference or evaluation.

    The function resolves configured feature roles, converts the configured
    date column to ``Date``, expands hidden-state lists/arrays and optionally
    enforces a fitted feature schema.

    Args:
        frame: Input Polars DataFrame.
        config: Canonical task configuration.
        require_target: Require the target column for training or evaluation.
        fitted_schema: Training schema to enforce during inference.
        hidden_dimensions: Expected sizes of hidden-state columns.
        categorical_role_columns: Configured role columns to require and append as
            categorical features.
        excluded_feature_columns: Feature names excluded for the current model scope.
        public_column_names: Optional internal-to-public names used in schema errors.
        expand_hidden_states: Expand every embedding into one scalar feature per
            coordinate, as boosting requires. When ``False`` the column stays a
            ``List(Float32)`` vector, is not a feature at all, and its width is
            recorded in :attr:`FeatureSchema.hidden_states` instead.

    Returns:
        Prepared frame/schema and observed hidden-state dimensions.

    Raises:
        SchemaError: If columns, feature dtypes, embeddings, or numerical values
            are incompatible with the configuration or fitted schema.
    """
    role_fields = {
        config.target_column: "target_column",
        config.client_id_column: "client_id_column",
        **(
            {config.date_column: "date_column"}
            if config.date_column is not None
            else {}
        ),
    }
    if config.group_column is not None:
        role_fields[config.group_column] = "group_column"
    treatment_column = getattr(config, "treatment_column", None)
    if treatment_column is not None:
        role_fields[treatment_column] = "treatment_column"
    required = [
        config.client_id_column,
        *(column for column in (config.date_column,) if column is not None),
        *config.categorical_columns,
        *config.numerical_columns,
        *config.hidden_state_columns,
        *(column for column in categorical_role_columns if column is not None),
    ]
    if require_target:
        required.append(config.target_column)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        visible = (
            [public_column_names.get(name, name) for name in missing]
            if public_column_names
            else missing
        )
        details = []
        for internal, public in zip(missing, visible, strict=True):
            field_name = role_fields.get(internal)
            if field_name is None:
                if internal in config.categorical_columns:
                    field_name = "categorical_columns"
                elif internal in config.numerical_columns:
                    field_name = "numerical_columns"
                else:
                    field_name = "hidden_state_columns"
            details.append(f"{field_name}={public!r}")
        msg = f"Missing required columns configured by: {details}"
        raise SchemaError(msg)

    if config.date_column is not None:
        frame = normalize_date(frame, config.date_column)
    if expand_hidden_states:
        frame, expanded, dimensions = _expand_hidden_states(
            frame, tuple(config.hidden_state_columns), hidden_dimensions
        )
        hidden_states: dict[str, int] = {}
    else:
        frame, dimensions = _normalize_hidden_states(
            frame, tuple(config.hidden_state_columns), hidden_dimensions
        )
        expanded = ()
        hidden_states = dict(dimensions)

    categorical = list(config.categorical_columns)
    for column in categorical_role_columns:
        if column and column not in categorical:
            categorical.append(column)
    numerical = [*config.numerical_columns, *expanded]
    excluded = set(excluded_feature_columns)
    categorical = [name for name in categorical if name not in excluded]
    numerical = [name for name in numerical if name not in excluded]
    feature_order = (*categorical, *numerical)
    if not feature_order:
        msg = "No features configured; set categorical_columns, numerical_columns or hidden_state_columns"
        raise SchemaError(msg)

    if fitted_schema is not None:
        missing_fitted = sorted(set(fitted_schema.feature_order) - set(frame.columns))
        if missing_fitted:
            msg = f"Missing fitted features: {missing_fitted}"
            raise SchemaError(msg)
        feature_order = fitted_schema.feature_order
        categorical = list(fitted_schema.categorical)
        numerical = list(fitted_schema.numerical)
        dtype_mismatches = {
            name: (fitted_schema.dtypes[name], str(frame.schema[name]))
            for name in fitted_schema.feature_order
            if str(frame.schema[name]) != fitted_schema.dtypes[name]
        }
        if dtype_mismatches:
            details = ", ".join(
                f"{name}: expected {expected}, got {actual}"
                for name, (expected, actual) in dtype_mismatches.items()
            )
            msg = f"Feature dtype mismatch: {details}"
            raise SchemaError(msg)

    bad_inf = [
        name
        for name in numerical
        if frame.select(
            pl.col(name).cast(pl.Float64, strict=False).is_infinite().any()
        ).item()
    ]
    if bad_inf:
        msg = f"Infinite values are not allowed in numerical columns: {bad_inf}"
        raise SchemaError(msg)

    schema = FeatureSchema(
        categorical=tuple(categorical),
        numerical=tuple(numerical),
        feature_order=feature_order,
        dtypes={name: str(frame.schema[name]) for name in feature_order},
        target_column=config.target_column,
        client_id_column=config.client_id_column,
        treatment_column=treatment_column,
        group_column=config.group_column,
        hidden_states=hidden_states,
    )
    return PreparedData(frame=frame, schema=schema), dimensions
