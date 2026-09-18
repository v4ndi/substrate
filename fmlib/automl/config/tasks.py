"""Task-specific typed configurations."""

from dataclasses import dataclass

from fmlib.automl.exceptions import ConfigError

from .base import BaseTaskConfig


@dataclass(frozen=True, kw_only=True)
class BinaryTaskConfig(BaseTaskConfig):
    """Configure a binary classification task.

    Inherits all fields from :class:`BaseTaskConfig` and defaults
    ``optimization_metric`` to ``roc_auc``.

    Raises:
        ConfigError: If inherited configuration constraints are violated.
    """

    task_name = "binary"


@dataclass(frozen=True, kw_only=True)
class ResponseTaskConfig(BinaryTaskConfig):
    """Configure binary response modelling with an optional treatment feature.

    ``inverse_treatment`` must be boolean when ``treatment_column`` is set and
    must remain ``None`` otherwise.
    """

    task_name = "response"
    treatment_column: str | None = None
    inverse_treatment: bool | None = None


@dataclass(frozen=True, kw_only=True)
class RegressionTaskConfig(BaseTaskConfig):
    """Configure a continuous-target regression task.

    Inherits all fields from :class:`BaseTaskConfig`. Supported optimization
    metrics are ``mse`` and ``mae``; ``mse`` is the default.

    Raises:
        ConfigError: If the metric or an inherited field is invalid.
    """

    task_name = "regression"


@dataclass(frozen=True, kw_only=True)
class MulticlassTaskConfig(BaseTaskConfig):
    """Configure a multiclass classification task.

    Inherits all fields from :class:`BaseTaskConfig`. Supported optimization metrics
    are ``roc_auc_ovr_macro``, ``accuracy``, ``f1_macro``, and ``log_loss``.

    Raises:
        ConfigError: If the metric or an inherited field is invalid.
    """

    task_name = "multiclass"


@dataclass(frozen=True, kw_only=True)
class UpliftTaskConfig(BaseTaskConfig):
    """Configure binary-outcome treatment-effect estimation.

    Extends :class:`BaseTaskConfig` with required treatment semantics. The
    explicit ``estimate_propensity`` field fits treatment
    probabilities from the feature matrix instead of using the empirical rate.

    Attributes:
        estimate_propensity: Whether to estimate row-level treatment propensity.

    Raises:
        ConfigError: If uplift roles, optimization metric, or inherited fields are invalid.
    """

    task_name = "uplift"
    required_explicit_fields = (
        *BaseTaskConfig.required_explicit_fields,
        "treatment_column",
        "inverse_treatment",
        "estimate_propensity",
    )
    treatment_column: str
    inverse_treatment: bool
    estimate_propensity: bool

    def __post_init__(self) -> None:
        """Validate uplift roles and configure the Qini objective."""
        if self.treatment_column is None:
            msg = "UpliftTaskConfig requires treatment_column"
            raise ConfigError(msg)
        super().__post_init__()
        if self.treatment_column in {
            *self.categorical_columns,
            *self.numerical_columns,
            *self.hidden_state_columns,
        }:
            msg = "Uplift treatment_column is learner-controlled and cannot be listed as a feature"
            raise ConfigError(msg)
        if not isinstance(self.estimate_propensity, bool):
            msg = "estimate_propensity must be a boolean"
            raise ConfigError(msg)
