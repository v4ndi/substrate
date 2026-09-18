"""Shared typed configuration and configuration-loading utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

from fmlib.automl.exceptions import ConfigError, UnsupportedBackendError
from fmlib.automl.metrics import default_optimization_metric, resolve_metric

FeatureColumns = Sequence[str] | str | Path

_DEFAULT_REMOTE_IMAGE = "registry.ca.sbrf.ru/ci02684173/ci02697916/notebooks/python3.12/cuda12.4/d-03.000.00:d-03.000.00-gigachat"
_DEFAULT_SHARED_FMLIB_VENV = "/home/datalab/nfs/sber-amazme-fmlib/env"
_TABNN_ENGINE = "tabular_transformer"
#: Trial budget when hyperopt is on and n_trials is omitted. A boosting
#: trial costs minutes and a network trial costs hours, so the two families
#: cannot share one number.
_DEFAULT_N_TRIALS = {"boosting": 50, "tabnn": 10}


@dataclass(frozen=True, kw_only=True)
class EnvironmentConfig:
    """Configure execution resources and logging for remote AutoML jobs.

    Attributes:
        image: Container image used by Osiris jobs.
        pool: Scheduler pool. ``None`` and ``common`` select the standard profile.
        venv_path: Existing shared fmlib virtual environment activated inside
            the job. Defaults to the documented shared-checkout environment.
        env: Additional environment variables. ``PYTHONPATH`` and
            ``CUDA_VISIBLE_DEVICES`` are controlled by the launcher.
        poll_interval_seconds: Delay between scheduler status requests.
        log_dir: Optional local directory for job metadata and logs.
        num_gpus: GPUs allocated to one job -- that is, to one trial of a
            search, not to the operation as a whole.
        num_nodes: Nodes allocated to one job.

    Raises:
        ConfigError: If an environment value or resource request is invalid.
    """

    image: str = _DEFAULT_REMOTE_IMAGE
    pool: str | None = "common"
    num_gpus: int | None = None
    num_nodes: int | None = None
    venv_path: str | Path = _DEFAULT_SHARED_FMLIB_VENV
    env: Mapping[str, str] = field(
        default_factory=lambda: {"OMP_NUM_THREADS": "7", "NCCL_DEBUG": "INFO"}
    )
    poll_interval_seconds: float = 30.0
    log_dir: str | Path | None = None

    def __post_init__(self) -> None:
        """Normalize collection fields and validate generic environment values."""
        if self.pool is not None and (
            not isinstance(self.pool, str) or not self.pool.strip()
        ):
            msg = f"environment.pool={self.pool!r}: expected None or a non-empty string"
            raise ConfigError(msg)
        if self.venv_path is None:
            msg = "environment.venv_path cannot be None; omit it to use the default shared fmlib environment"
            raise ConfigError(msg)
        object.__setattr__(
            self, "env", {str(key): str(value) for key, value in self.env.items()}
        )
        reserved = sorted(
            name
            for name in self.env
            if name.upper() in {"CUDA_VISIBLE_DEVICES", "PYTHONPATH"}
        )
        if reserved:
            msg = f"Environment variables are managed by the fmlib launcher and cannot be configured: {reserved}"
            raise ConfigError(msg)
        if self.poll_interval_seconds <= 0:
            msg = "environment.poll_interval_seconds must be positive"
            raise ConfigError(msg)
        for name in ("num_gpus", "num_nodes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                msg = (
                    f"environment.{name}={value!r}: expected a positive integer or None"
                )
                raise ConfigError(msg)
        if self.pool in {None, "common"}:
            conflicting = {
                name: getattr(self, name)
                for name in ("num_gpus", "num_nodes")
                if getattr(self, name) not in {None, 1}
            }
            if conflicting:
                msg = f"Standard Osiris profile requires num_nodes=1 and num_gpus=1; got {conflicting}"
                raise ConfigError(msg)
        elif self.num_gpus is None or self.num_nodes is None:
            msg = "Custom Osiris pool requires explicit num_gpus and num_nodes"
            raise ConfigError(msg)

    @property
    def effective_pool(self) -> str | None:
        """Return the scheduler pool after normalizing ``common`` to no pool."""
        return None if self.pool in {None, "common"} else self.pool

    @property
    def resource_profile(self) -> Literal["batch", "supercomp"]:
        """Return the legacy-equivalent resource profile selected by pool."""
        return "batch" if self.effective_pool is None else "supercomp"

    @property
    def resolved_num_gpus(self) -> int:
        return 1 if self.effective_pool is None else int(self.num_gpus)

    @property
    def resolved_num_nodes(self) -> int:
        return 1 if self.effective_pool is None else int(self.num_nodes)


def _resolve_feature_columns(value: FeatureColumns, field_name: str) -> tuple[str, ...]:
    """Resolve a feature list supplied inline or in an absolute YAML file."""
    if not isinstance(value, str | Path):
        try:
            columns = tuple(value)
        except TypeError as exc:
            msg = f"{field_name}={value!r}: expected a sequence of non-empty column names or an absolute YAML path"
            raise ConfigError(msg) from exc
        if not all(isinstance(column, str) and column for column in columns):
            msg = f"{field_name}={value!r}: expected a sequence of non-empty column names or an absolute YAML path"
            raise ConfigError(msg)
        return columns

    from omegaconf import OmegaConf

    columns_path = Path(value).expanduser()
    if not columns_path.is_absolute():
        msg = f"{field_name} YAML path must be absolute: {columns_path}"
        raise ConfigError(msg)
    columns_path = columns_path.resolve()
    if not columns_path.is_file():
        msg = f"{field_name} YAML does not exist: {columns_path}"
        raise ConfigError(msg)
    try:
        columns = OmegaConf.to_container(OmegaConf.load(columns_path), resolve=True)
    except Exception as exc:
        msg = f"Cannot load {field_name} YAML {columns_path}: {exc}"
        raise ConfigError(msg) from exc
    if isinstance(columns, Mapping):
        columns = columns.get(field_name)
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        msg = f"{field_name} YAML must contain a list of column names: {columns_path}"
        raise ConfigError(msg)
    return tuple(columns)


@dataclass(frozen=True, kw_only=True)
class BaseTaskConfig:
    """Common configuration for an AutoML task.

    Core execution, role, feature, scope, and output fields must be supplied
    explicitly. ``environment`` defaults to the documented shared fmlib
    environment and may be overridden when a different shared environment is
    required. Local execution accepts ``cpu`` or ``gpu``; remote execution
    environments require ``gpu``.
    Feature columns may be sequences of names or absolute paths to YAML lists.

    Attributes:
        env_type: Execution environment: ``local`` or ``osiris``.
        backend: Model family: ``boosting`` or ``tabnn``.
        engine: Estimator implementation. Boosting supports ``catboost`` and
            ``xgboost``; TabNN supports ``tabular_transformer``.
        device: Explicit ``cpu`` or ``gpu`` choice; remote environments support
            only ``gpu``.
        target_column: Dataset column containing the target.
        client_id_column: Required client identifier column.
        group_column: Channel/group column, or ``None`` to disable the role.
        date_column: Optional date column used for evaluation slices and reports.
        categorical_columns: Categorical feature names or an absolute YAML-list path.
        numerical_columns: Numerical feature names or an absolute YAML-list path.
        hidden_state_columns: List/Array Float32/Float64 embedding columns;
            values are normalized to Float32 and expanded into scalar features.
        model_layout: Train a global model, per-group models, or both branches.
            It must be omitted when ``group_column=None``; execution then
            resolves internally to ``global``.
        hyperopt: Whether to run Optuna before selecting model parameters.
        optimization_metric: Registered validation metric used to rank models.
        verbose: Training-log switch or logging period.
        model_params: Explicit estimator parameters used only without hyperopt.
        search_space: Optuna search-space overrides used only with hyperopt;
            ``None`` uses task defaults.
        n_trials: Trial budget of the search. It must be omitted without
            hyperopt; with hyperopt, omission resolves per backend family --
            50 for boosting, 10 for TabNN, where one trial costs hours rather
            than minutes. An explicit value is never capped.
        random_state: Reproducibility seed passed to supported components.
        output_dir: Root directory for artifacts, reports, logs, and remote handles.
        processed_data_path: Where preprocessed TabNN data is written and reused.
            ``None`` resolves to ``<output_dir>/processed``. Boosting ignores it;
            it prepares its matrices in memory.
        max_parallel_jobs: Cap on remote jobs in flight at once. ``None`` submits
            every planned job immediately. With a cap, submission happens in
            chunks and is resumable: jobs still ``planned`` are submitted by a
            later ``status()``.
        environment: Optional execution environment configuration. When omitted,
            remote execution uses the documented shared fmlib environment.

    Raises:
        ConfigError: If values or their cross-field combinations are invalid.
        UnsupportedBackendError: If the backend/engine combination is unsupported.
    """

    task_name: ClassVar[str]
    required_explicit_fields: ClassVar[tuple[str, ...]] = (
        "env_type",
        "backend",
        "engine",
        "device",
        "target_column",
        "client_id_column",
        "group_column",
        "date_column",
        "categorical_columns",
        "numerical_columns",
        "hidden_state_columns",
        "hyperopt",
        "output_dir",
    )

    env_type: Literal["local", "osiris"]
    backend: Literal["boosting", "tabnn"]
    engine: str
    device: Literal["cpu", "gpu"]
    target_column: str
    client_id_column: str
    group_column: str | None
    date_column: str | None
    categorical_columns: FeatureColumns
    numerical_columns: FeatureColumns
    hidden_state_columns: Sequence[str]
    model_layout: Literal["global", "per_group", "global_and_per_group"] | None = None
    hyperopt: bool
    output_dir: str | Path
    processed_data_path: str | Path | None = None
    max_parallel_jobs: int | None = None
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    optimization_metric: str | None = None
    verbose: bool | int = True
    model_params: Mapping[str, Any] = field(default_factory=dict)
    search_space: Mapping[str, Any] | None = None
    n_trials: int | None = None
    random_state: int = 42

    def __post_init__(self) -> None:
        """Normalize immutable fields and validate cross-field constraints."""
        optimization_metric = (
            default_optimization_metric(self.task_name)
            if self.optimization_metric is None
            else self.optimization_metric
        )
        resolve_metric(optimization_metric, self.task_name, "optimization")
        object.__setattr__(self, "optimization_metric", optimization_metric)
        required_strings = {
            "engine": self.engine,
            "target_column": self.target_column,
            "client_id_column": self.client_id_column,
        }
        for name, value in required_strings.items():
            if not isinstance(value, str) or not value:
                msg = f"{name}={value!r}: expected a non-empty string"
                raise ConfigError(msg)
        optional_role_strings = {
            "date_column": self.date_column,
            "group_column": self.group_column,
            "treatment_column": getattr(self, "treatment_column", None),
        }
        for name, value in optional_role_strings.items():
            if value is not None and (not isinstance(value, str) or not value):
                msg = f"{name}={value!r}: expected a non-empty string or None"
                raise ConfigError(msg)
        if not isinstance(self.hyperopt, bool):
            msg = f"hyperopt={self.hyperopt!r}: expected True or False"
            raise ConfigError(msg)
        if not isinstance(self.output_dir, str | Path) or not str(self.output_dir):
            msg = f"output_dir={self.output_dir!r}: expected a non-empty path"
            raise ConfigError(msg)
        if self.processed_data_path is not None and (
            not isinstance(self.processed_data_path, str | Path)
            or not str(self.processed_data_path)
        ):
            msg = f"processed_data_path={self.processed_data_path!r}: expected a non-empty path or None"
            raise ConfigError(msg)
        if self.max_parallel_jobs is not None and (
            isinstance(self.max_parallel_jobs, bool)
            or not isinstance(self.max_parallel_jobs, int)
            or self.max_parallel_jobs < 1
        ):
            msg = f"max_parallel_jobs={self.max_parallel_jobs!r}: expected an integer >= 1 or None"
            raise ConfigError(msg)
        for name in ("categorical_columns", "numerical_columns"):
            object.__setattr__(
                self, name, _resolve_feature_columns(getattr(self, name), name)
            )
        hidden_state_columns = self.hidden_state_columns
        if hidden_state_columns is None:
            msg = "hidden_state_columns=None: expected a sequence of non-empty column names"
            raise ConfigError(msg)
        if isinstance(hidden_state_columns, str):
            hidden_state_columns = (hidden_state_columns,)
        else:
            try:
                hidden_state_columns = tuple(hidden_state_columns)
            except TypeError as exc:
                msg = f"hidden_state_columns={hidden_state_columns!r}: expected a sequence of non-empty column names"
                raise ConfigError(msg) from exc
        if not all(
            isinstance(column, str) and column for column in hidden_state_columns
        ):
            msg = f"hidden_state_columns={hidden_state_columns!r}: expected a sequence of non-empty column names"
            raise ConfigError(msg)
        object.__setattr__(self, "hidden_state_columns", tuple(hidden_state_columns))
        object.__setattr__(self, "model_params", dict(self.model_params))
        if isinstance(self.environment, Mapping):
            object.__setattr__(
                self, "environment", EnvironmentConfig(**self.environment)
            )
        if self.search_space is not None:
            object.__setattr__(self, "search_space", dict(self.search_space))

        if self.env_type in {"batch", "supercomp"}:
            msg = f"Incompatible legacy env_type={self.env_type!r}; use 'osiris' with EnvironmentConfig.pool"
            raise ConfigError(msg)
        if self.env_type not in {"local", "osiris"}:
            msg = f"env_type={self.env_type!r}: expected 'local' or 'osiris'"
            raise ConfigError(msg)
        if self.backend not in {"boosting", "tabnn"}:
            msg = f"backend={self.backend!r}: expected 'boosting' or 'tabnn'"
            raise ConfigError(msg)
        if self.device not in {"cpu", "gpu"}:
            msg = f"device={self.device!r}: expected 'cpu' or 'gpu'"
            raise ConfigError(msg)
        if self.env_type != "local" and self.device != "gpu":
            msg = f"{self.env_type!r} supports only device='gpu'"
            raise ConfigError(msg)
        if self.backend == "boosting" and self.engine not in {"catboost", "xgboost"}:
            msg = f"boosting backend does not support engine={self.engine!r}"
            raise UnsupportedBackendError(msg)
        if self.backend == "tabnn" and self.engine != _TABNN_ENGINE:
            # 'ste' is called out by name: the model class was renamed
            # STEv2 -> TabularTransformer and no artifact carries the old
            # engine, so this can only be a config written from memory.
            hint = (
                " (the model class was renamed STEv2 -> TabularTransformer)"
                if self.engine == "ste"
                else ""
            )
            msg = f"tabnn backend does not support engine={self.engine!r}; use {_TABNN_ENGINE!r}{hint}"
            raise UnsupportedBackendError(msg)
        if set(self.categorical_columns) & set(self.numerical_columns):
            overlap = sorted(
                set(self.categorical_columns) & set(self.numerical_columns)
            )
            msg = f"categorical_columns and numerical_columns overlap: {overlap}"
            raise ConfigError(msg)
        all_features = [
            *self.categorical_columns,
            *self.numerical_columns,
            *self.hidden_state_columns,
        ]
        if len(all_features) != len(set(all_features)):
            msg = "Feature names must be unique across categorical_columns, numerical_columns and hidden_state_columns"
            raise ConfigError(msg)
        treatment_column = getattr(self, "treatment_column", None)
        inverse_treatment = getattr(self, "inverse_treatment", None)
        if treatment_column is None:
            if inverse_treatment is not None:
                msg = f"inverse_treatment={inverse_treatment!r}: must be omitted when treatment_column=None"
                raise ConfigError(msg)
        elif not isinstance(inverse_treatment, bool):
            msg = f"inverse_treatment={inverse_treatment!r}: expected True or False when treatment_column={treatment_column!r}"
            raise ConfigError(msg)
        if self.hyperopt:
            if self.n_trials is None:
                object.__setattr__(self, "n_trials", _DEFAULT_N_TRIALS[self.backend])
            elif (
                isinstance(self.n_trials, bool)
                or not isinstance(self.n_trials, int)
                or self.n_trials < 1
            ):
                msg = "n_trials must be an integer greater than or equal to 1"
                raise ConfigError(msg)
        elif self.n_trials is not None:
            msg = "n_trials can be configured only when hyperopt=True"
            raise ConfigError(msg)
        if not isinstance(self.verbose, bool | int) or (
            isinstance(self.verbose, int) and self.verbose < 0
        ):
            msg = "verbose must be a boolean or a non-negative integer logging period"
            raise ConfigError(msg)
        task_owned_model_params = {
            "custom_loss",
            "custom_metric",
            "device",
            "devices",
            "eval_metric",
            "gpu_id",
            "logging_level",
            "loss_function",
            "objective",
            "random_seed",
            "random_state",
            "seed",
            "silent",
            "task_type",
            "tree_method",
            "verbose",
            "verbose_eval",
            "verbosity",
        }
        misplaced_model_params = sorted(
            task_owned_model_params.intersection(self.model_params)
        )
        misplaced_search_params = sorted(
            task_owned_model_params.intersection(self.search_space or {})
        )
        if misplaced_model_params or misplaced_search_params:
            misplaced = sorted(set(misplaced_model_params + misplaced_search_params))
            msg = (
                f"Task-level runtime parameters cannot be set in model_params/search_space: {misplaced}; "
                "use the matching top-level task config fields"
            )
            raise ConfigError(msg)
        if not self.hyperopt and self.search_space:
            msg = "search_space can be configured only when hyperopt=True"
            raise ConfigError(msg)
        if self.hyperopt and self.model_params:
            msg = (
                "model_params cannot be configured when hyperopt=True because Optuna selects model parameters; "
                "use search_space to control the values Optuna may choose"
            )
            raise ConfigError(msg)
        overlapping_parameters = sorted(
            set(self.model_params).intersection(self.search_space or {})
        )
        if overlapping_parameters:
            msg = (
                "Parameters cannot be fixed in model_params and tuned in search_space at the same time: "
                f"{overlapping_parameters}"
            )
            raise ConfigError(msg)
        if self.group_column is None:
            if self.model_layout is not None:
                msg = f"model_layout={self.model_layout!r}: must be omitted when group_column=None"
                raise ConfigError(msg)
        elif self.model_layout is None:
            msg = (
                "model_layout=None: expected 'global', 'per_group', or "
                f"'global_and_per_group' when group_column={self.group_column!r}"
            )
            raise ConfigError(msg)
        elif self.model_layout not in {"global", "per_group", "global_and_per_group"}:
            msg = f"model_layout={self.model_layout!r}: expected 'global', 'per_group', or 'global_and_per_group'"
            raise ConfigError(msg)
        if self.backend == "tabnn" and self.model_layout == "global_and_per_group":
            # A batch is drawn from one file and there is no shuffle buffer
            # across files, so a global model trained on group-partitioned
            # data would see one group per step. Either layout alone is fine:
            # 'global' reads the flat layout, 'per_group' the partitioned one.
            msg = (
                "model_layout='global_and_per_group' is not supported with backend='tabnn': "
                "the two layouts need different physical layouts of the processed data; "
                "run 'global' and 'per_group' as separate tasks"
            )
            raise ConfigError(msg)
        roles = {
            "target_column": self.target_column,
            "client_id_column": self.client_id_column,
            "date_column": self.date_column,
            "group_column": self.group_column,
            "treatment_column": treatment_column,
        }
        populated_roles = {
            name: value for name, value in roles.items() if value is not None
        }
        duplicate_roles = sorted(
            value
            for value in set(populated_roles.values())
            if list(populated_roles.values()).count(value) > 1
        )
        if duplicate_roles:
            msg = f"Column names cannot be shared by multiple roles: {duplicate_roles}"
            raise ConfigError(msg)
        forbidden_features = {
            self.target_column,
            self.client_id_column,
            self.date_column,
        }
        configured_features = (
            set(self.categorical_columns)
            | set(self.numerical_columns)
            | set(self.hidden_state_columns)
        )
        leaked = sorted(forbidden_features & configured_features)
        if leaked:
            msg = f"Role columns cannot be model features: {leaked}"
            raise ConfigError(msg)
        categorical_roles = {
            value
            for value in (self.group_column, treatment_column)
            if value is not None
        }
        numerical_roles = sorted(
            categorical_roles
            & (set(self.numerical_columns) | set(self.hidden_state_columns))
        )
        if numerical_roles:
            msg = f"group/treatment roles are categorical and cannot be in numerical_columns: {numerical_roles}"
            raise ConfigError(msg)

    @property
    def resolved_processed_data_path(self) -> Path:
        """Resolve where preprocessed data is written and looked up.

        Returns:
            The configured path, or ``<output_dir>/processed`` when omitted.
        """
        if self.processed_data_path is None:
            return Path(self.output_dir).expanduser() / "processed"
        return Path(self.processed_data_path).expanduser()

    @property
    def resolved_device(self) -> Literal["cpu", "gpu"]:
        """Resolve the device used by the execution environment.

        Returns:
            The configured device. Remote configurations are always ``gpu``.
        """
        return self.device

    @property
    def resolved_model_layout(
        self,
    ) -> Literal["global", "per_group", "global_and_per_group"]:
        """Return the internal global layout when the public field is disabled."""
        return "global" if self.model_layout is None else self.model_layout

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]):
        """Build a typed config from flat or sectioned mapping data.

        Recognized values may be placed at the root or in ``task``, ``data``,
        ``train``, and ``evaluate`` sections. Later sections override earlier ones.

        Args:
            mapping: Root configuration mapping, optionally split into supported sections.

        Returns:
            A validated instance of the concrete task configuration class.

        Raises:
            ConfigError: If a section has an invalid type or contains unknown fields.
        """
        legacy_fields = {"model_scope", "report_month_column"}
        legacy = sorted(
            legacy_fields
            & {
                key
                for section in (
                    mapping,
                    *(
                        mapping.get(name, {})
                        for name in ("task", "data", "train", "evaluate")
                    ),
                )
                if isinstance(section, Mapping)
                for key in section
            }
        )
        if legacy:
            msg = f"Incompatible legacy configuration fields: {legacy}"
            raise ConfigError(msg)
        valid_names = {item.name for item in fields(cls)}
        section_names = {"task", "data", "train", "evaluate", "environment"}
        unknown_root = sorted(set(mapping) - valid_names - section_names)
        if unknown_root:
            msg = f"Unknown {cls.__name__} configuration fields at root: {unknown_root}"
            raise ConfigError(msg)

        sections: list[Mapping[str, Any]] = [mapping]
        for section_name in ("task", "data", "train", "evaluate"):
            section = mapping.get(section_name, {})
            if not isinstance(section, Mapping):
                msg = f"Configuration section {section_name!r} must be a mapping"
                raise ConfigError(msg)
            unknown = sorted(set(section) - valid_names)
            if unknown:
                msg = f"Unknown {cls.__name__} configuration fields in section {section_name!r}: {unknown}"
                raise ConfigError(msg)
            sections.append(section)

        values: dict[str, Any] = {}
        for section in sections:
            values.update({
                key: value for key, value in section.items() if key in valid_names
            })
        environment = mapping.get("environment")
        if environment is not None:
            if isinstance(environment, EnvironmentConfig):
                values["environment"] = environment
            else:
                if not isinstance(environment, Mapping):
                    msg = "Configuration section 'environment' must be a mapping"
                    raise ConfigError(msg)
                valid_environment_names = {
                    item.name for item in fields(EnvironmentConfig)
                }
                unknown_environment = sorted(set(environment) - valid_environment_names)
                if unknown_environment:
                    msg = f"Unknown EnvironmentConfig fields: {unknown_environment}"
                    raise ConfigError(msg)
                values["environment"] = EnvironmentConfig(**environment)
        missing_required = sorted(set(cls.required_explicit_fields) - set(values))
        if missing_required:
            msg = f"Missing required {cls.__name__} configuration fields: {missing_required}"
            raise ConfigError(msg)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path):
        """Load and resolve a typed task configuration from YAML.

        Args:
            path: Path to a YAML file. OmegaConf interpolations are resolved.

        Returns:
            A validated instance of the concrete task configuration class.

        Raises:
            ConfigError: If the file cannot be loaded or its root is not a mapping.
        """
        from omegaconf import OmegaConf

        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            msg = f"YAML config does not exist: {config_path}"
            raise ConfigError(msg)
        try:
            payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        except Exception as exc:
            msg = f"Cannot load YAML config {config_path}: {exc}"
            raise ConfigError(msg) from exc
        if not isinstance(payload, Mapping):
            msg = f"YAML config root must be a mapping: {config_path}"
            raise ConfigError(msg)
        return cls.from_mapping(payload)
