from dataclasses import asdict
from inspect import signature

import pytest

from avatar.automl import BinaryTaskConfig, EnvironmentConfig, ResponseTaskConfig
from avatar.automl.exceptions import ConfigError, UnsupportedBackendError


def _config(**overrides):
    values = {
        "env_type": "local",
        "backend": "boosting",
        "engine": "catboost",
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": (),
        "hidden_state_columns": (),
        "model_layout": "global",
        "hyperopt": False,
        "output_dir": "outputs",
        "environment": {},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


@pytest.mark.parametrize("metric", ["auc", "accuracy", "roc-auc", "unknown"])
def test_binary_rejects_unknown_named_metric_during_config_creation(metric):
    with pytest.raises(ConfigError, match="Unknown metric|incompatible"):
        _config(optimization_metric=metric)


def test_binary_rejects_callable_optimization_metric():
    with pytest.raises(ConfigError, match="non-empty string"):
        _config(optimization_metric=lambda _target, _scores: 0.0)


@pytest.mark.parametrize("metric", ["mse", "unknown", "precision@10"])
def test_response_rejects_invalid_optimization_metric(metric):
    values = asdict(_config()) | {
        "optimization_metric": metric,
        "treatment_column": "treatment",
        "inverse_treatment": False,
    }
    with pytest.raises(ConfigError, match="Unknown metric|incompatible|only for evaluation"):
        ResponseTaskConfig(**values)


def test_python_and_mapping_configs_are_equal():
    expected = _config(
        device="gpu",
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hyperopt=True,
        n_trials=3,
    )
    actual = BinaryTaskConfig.from_mapping(
        {
            "env_type": "local",
            "backend": "boosting",
            "engine": "catboost",
            "device": "gpu",
            "data": {
                "target_column": "target",
                "client_id_column": "epk_id",
                "group_column": "group",
                "date_column": "report_month",
                "categorical_columns": ["segment"],
                "numerical_columns": ["balance"],
                "hidden_state_columns": [],
            },
            "train": {"model_layout": "global", "hyperopt": True, "n_trials": 3},
            "output_dir": "outputs",
            "environment": {},
        }
    )

    assert actual == expected


def test_python_and_yaml_configs_are_equal(tmp_path):
    categorical_columns_path = tmp_path / "categorical_columns.yaml"
    numerical_columns_path = tmp_path / "numerical_columns.yaml"
    categorical_columns_path.write_text("- segment\n- city\n", encoding="utf-8")
    numerical_columns_path.write_text("numerical_columns:\n  - balance\n  - age\n", encoding="utf-8")
    yaml_path = tmp_path / "binary.yaml"
    yaml_path.write_text(
        f"""
task:
  env_type: local
  backend: boosting
  engine: catboost
  device: gpu
data:
  target_column: target
  client_id_column: epk_id
  group_column: group
  date_column: report_month
  categorical_columns: {categorical_columns_path.as_posix()}
  numerical_columns: {numerical_columns_path.as_posix()}
  hidden_state_columns: []
train:
  model_layout: global
  hyperopt: true
  n_trials: 3
output_dir: outputs
environment: {{}}
""",
        encoding="utf-8",
    )

    assert BinaryTaskConfig.from_yaml(yaml_path) == _config(
        device="gpu",
        categorical_columns=["segment", "city"],
        numerical_columns=["balance", "age"],
        hyperopt=True,
        n_trials=3,
    )


def test_python_and_yaml_environment_configs_are_equal(tmp_path):
    yaml_path = tmp_path / "remote.yaml"
    yaml_path.write_text(
        f"""
env_type: osiris
backend: boosting
engine: catboost
device: gpu
target_column: target
client_id_column: epk_id
group_column: group
date_column: report_month
categorical_columns: []
numerical_columns: []
hidden_state_columns: []
model_layout: global_and_per_group
hyperopt: false
output_dir: outputs
environment:
  pool: b2c
  num_nodes: 1
  num_gpus: 1
  venv_path: /shared/fmlib/env
  poll_interval_seconds: 5
  log_dir: {tmp_path.as_posix()}/logs
""",
        encoding="utf-8",
    )
    expected = BinaryTaskConfig(
        env_type="osiris",
        backend="boosting",
        engine="catboost",
        device="gpu",
        target_column="target",
        client_id_column="epk_id",
        group_column="group",
        date_column="report_month",
        categorical_columns=(),
        numerical_columns=(),
        hidden_state_columns=(),
        model_layout="global_and_per_group",
        hyperopt=False,
        output_dir="outputs",
        environment=EnvironmentConfig(
            pool="b2c",
            num_nodes=1,
            num_gpus=1,
            venv_path="/shared/fmlib/env",
            poll_interval_seconds=5,
            log_dir=f"{tmp_path.as_posix()}/logs",
        ),
    )

    assert BinaryTaskConfig.from_yaml(yaml_path) == expected


def test_legacy_generated_window_fields_are_rejected():
    with pytest.raises(ConfigError, match="Unknown BinaryTaskConfig configuration fields"):
        BinaryTaskConfig.from_mapping(
            {
                "env_type": "local",
                "backend": "boosting",
                "engine": "catboost",
                "device": "cpu",
                "evaluate": {"calib_months": ["2025-10-01"], "oot_months": ["2025-11-01"]},
            }
        )


def test_python_config_accepts_absolute_feature_yaml_paths(tmp_path):
    categorical_columns_path = tmp_path / "categorical_columns.yaml"
    numerical_columns_path = tmp_path / "numerical_columns.yaml"
    categorical_columns_path.write_text("categorical_columns: [segment, city]\n", encoding="utf-8")
    numerical_columns_path.write_text("- balance\n- age\n", encoding="utf-8")

    config = _config(
        categorical_columns=categorical_columns_path.resolve(),
        numerical_columns=str(numerical_columns_path.resolve()),
    )

    assert config.categorical_columns == ("segment", "city")
    assert config.numerical_columns == ("balance", "age")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("features/categorical_columns.yaml", "must be absolute"),
        ("missing.yaml", "must be absolute"),
    ],
)
def test_relative_feature_yaml_path_is_rejected(value, message):
    with pytest.raises(ConfigError, match=message):
        _config(categorical_columns=value)


def test_missing_or_malformed_absolute_feature_yaml_is_rejected(tmp_path):
    missing_path = tmp_path / "missing.yaml"
    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text("categorical_columns: segment\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="does not exist"):
        _config(categorical_columns=missing_path)
    with pytest.raises(ConfigError, match="must contain a list"):
        _config(categorical_columns=malformed_path)


def test_missing_or_non_mapping_main_yaml_is_rejected(tmp_path):
    missing_path = tmp_path / "missing.yaml"
    sequence_path = tmp_path / "sequence.yaml"
    sequence_path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="does not exist"):
        BinaryTaskConfig.from_yaml(missing_path)
    with pytest.raises(ConfigError, match="root must be a mapping"):
        BinaryTaskConfig.from_yaml(sequence_path)


@pytest.mark.parametrize(
    "payload",
    [
        {"env_type": "local", "backend": "boosting", "engine": "catboost", "device": "cpu", "typo": 1},
        {
            "task": {"env_type": "local", "backend": "boosting", "engine": "catboost", "device": "cpu"},
            "train": {"early_stoping_rounds": 10},
        },
        {
            "env_type": "local",
            "backend": "boosting",
            "engine": "catboost",
            "device": "cpu",
            "environment": {"venv": "/shared/fmlib/env"},
        },
        {
            "env_type": "osiris",
            "backend": "boosting",
            "engine": "catboost",
            "environment": {"pool_name": "public"},
        },
    ],
)
def test_unknown_mapping_fields_are_rejected(payload):
    with pytest.raises(ConfigError, match="Unknown"):
        BinaryTaskConfig.from_mapping(payload)


@pytest.mark.parametrize("name", ["PYTHONPATH", "pythonpath", "CUDA_VISIBLE_DEVICES"])
def test_launcher_managed_environment_variables_are_rejected(name):
    with pytest.raises(ConfigError, match="managed by the fmlib launcher"):
        EnvironmentConfig(env={name: "/user/value"})


@pytest.mark.parametrize("pool", ["", "   ", 123, False, []])
def test_environment_pool_requires_none_or_a_non_empty_string(pool):
    with pytest.raises(ConfigError, match=r"environment\.pool=.*None or a non-empty string"):
        EnvironmentConfig(pool=pool)


def test_environment_pool_defaults_to_common():
    assert EnvironmentConfig().pool == "common"
    assert _config().environment.pool == "common"


def test_environment_resource_profile_is_selected_only_by_pool():
    for pool in (None, "common"):
        environment = EnvironmentConfig(pool=pool)
        assert environment.effective_pool is None
        assert environment.resource_profile == "batch"
        assert (environment.resolved_num_nodes, environment.resolved_num_gpus) == (1, 1)
    custom = EnvironmentConfig(pool="research", num_nodes=2, num_gpus=3)
    assert custom.effective_pool == "research"
    assert custom.resource_profile == "supercomp"
    assert (custom.resolved_num_nodes, custom.resolved_num_gpus) == (2, 3)


@pytest.mark.parametrize("field", ["num_nodes", "num_gpus"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_environment_resources_are_positive_integers(field, value):
    with pytest.raises(ConfigError, match=rf"environment\.{field}=.*positive integer"):
        EnvironmentConfig(**{field: value})


def test_legacy_per_action_resources_are_rejected_without_alias():
    values = asdict(_config())
    values["environment"] = {"resources": {"train": {"num_gpus": 1}}}
    with pytest.raises(ConfigError, match="Unknown EnvironmentConfig fields.*resources"):
        BinaryTaskConfig.from_mapping(values)


@pytest.mark.parametrize(
    ("backend", "engine"),
    [("boosting", "ste"), ("tabnn", "catboost"), ("tabnn", "dcn")],
)
def test_invalid_backend_engine_pair_is_rejected(backend, engine):
    with pytest.raises(UnsupportedBackendError):
        _config(backend=backend, engine=engine, device="gpu")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"categorical_columns": ["feature"], "numerical_columns": ["feature"]}, "overlap"),
        ({"categorical_columns": ["feature"], "hidden_state_columns": ["feature"]}, "must be unique"),
        ({"client_id_column": ""}, "client_id_column=''"),
        ({"hyperopt": True, "n_trials": 0}, "n_trials must be an integer"),
        ({"hyperopt": True, "n_trials": 1.5}, "n_trials must be an integer"),
        ({"verbose": -1}, "verbose must be"),
        ({"verbose": "yes"}, "verbose must be"),
        ({"model_layout": "product"}, "model_layout='product'"),
        ({"target_column": "feature", "numerical_columns": ["feature"]}, "cannot be model features"),
        ({"client_id_column": "feature", "categorical_columns": ["feature"]}, "cannot be model features"),
        ({"date_column": "feature", "numerical_columns": ["feature"]}, "cannot be model features"),
        ({"group_column": "feature", "numerical_columns": ["feature"]}, "categorical"),
        ({"target_column": "label", "group_column": "label"}, "multiple roles"),
    ],
)
def test_invalid_common_configuration_is_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        _config(**overrides)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("env_type", None),
        ("backend", None),
        ("engine", None),
        ("device", None),
        ("target_column", None),
        ("client_id_column", None),
        ("categorical_columns", None),
        ("numerical_columns", None),
        ("hidden_state_columns", None),
        ("hyperopt", None),
        ("output_dir", None),
    ],
)
def test_invalid_required_field_values_name_the_field_and_value(field_name, value):
    with pytest.raises(ConfigError, match=rf"{field_name}=None"):
        _config(**{field_name: value})


def test_date_column_may_be_explicitly_disabled():
    assert _config(date_column=None).date_column is None


@pytest.mark.parametrize("legacy_field", ["model_scope", "report_month_column"])
def test_legacy_role_and_layout_fields_are_rejected_explicitly(legacy_field):
    with pytest.raises(ConfigError, match=rf"Incompatible legacy configuration fields.*{legacy_field}"):
        BinaryTaskConfig.from_mapping(asdict(_config()) | {legacy_field: "legacy"})


def test_model_layout_is_conditional_on_group_column():
    without_group = _config(group_column=None, model_layout=None)

    assert without_group.model_layout is None
    assert without_group.resolved_model_layout == "global"
    with pytest.raises(ConfigError, match="model_layout='global'.*group_column=None"):
        _config(group_column=None, model_layout="global")
    with pytest.raises(ConfigError, match="model_layout=None.*group_column='group'"):
        _config(model_layout=None)
    with pytest.raises(ConfigError, match="model_layout='channel'"):
        _config(model_layout="channel")


def test_legacy_model_scope_is_not_accepted():
    values = asdict(_config())
    values.pop("model_layout")
    with pytest.raises(TypeError, match="model_scope"):
        BinaryTaskConfig(**(values | {"model_scope": "product"}))
    with pytest.raises(ConfigError, match="Incompatible legacy configuration fields.*model_scope"):
        BinaryTaskConfig.from_mapping(values | {"model_scope": "product"})


def test_default_role_columns_are_configured_separately():
    config = _config()

    assert config.group_column == "group"
    assert "treatment_column" not in signature(BinaryTaskConfig).parameters
    assert "inverse_treatment" not in signature(BinaryTaskConfig).parameters
    assert "group" not in config.categorical_columns
    assert config.model_layout == "global"
    assert config.n_trials is None


def test_early_stopping_is_native_instead_of_top_level():
    assert "early_stopping_rounds" not in signature(BinaryTaskConfig).parameters
    assert _config(model_params={"od_wait": 25}).model_params == {"od_wait": 25}
    xgboost = _config(engine="xgboost", model_params={"early_stopping_rounds": 25})
    assert xgboost.model_params == {"early_stopping_rounds": 25}
    tuned = _config(engine="xgboost", hyperopt=True, search_space={"early_stopping_rounds": [25]})
    assert tuned.search_space == {"early_stopping_rounds": [25]}


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_task_owned_native_parameters_cannot_be_hidden(field):
    with pytest.raises(ConfigError, match=rf"Task-level runtime parameters.*{field}"):
        _config(model_params={field: 1})
    with pytest.raises(ConfigError, match=rf"Task-level runtime parameters.*{field}"):
        _config(search_space={field: [1]}, hyperopt=True)


def test_n_trials_defaults_only_with_hyperopt():
    assert _config(hyperopt=True).n_trials == 50

    with pytest.raises(ConfigError, match="n_trials can be configured only when hyperopt=True"):
        _config(n_trials=50)


def test_device_is_explicit_and_remote_rejects_cpu():
    boosting = _config(device="gpu")

    assert boosting.device == "gpu"
    assert boosting.resolved_device == "gpu"
    with pytest.raises(ConfigError, match="supports only device='gpu'"):
        _config(env_type="osiris", device="cpu", environment={"venv_path": "/shared/fmlib/env"})
    remote = _config(env_type="osiris", device="gpu", environment={"venv_path": "/shared/fmlib/env"})
    assert remote.device == "gpu"
    assert remote.resolved_device == "gpu"


@pytest.mark.parametrize("env_type", ["batch", "supercomp"])
def test_legacy_remote_environment_values_are_rejected_explicitly(env_type):
    with pytest.raises(ConfigError, match=rf"Incompatible legacy env_type={env_type!r}"):
        _config(env_type=env_type, device="gpu")
    with pytest.raises(ConfigError, match="does not support"):
        _config(backend="tabnn", engine="ste", device="cpu")


def test_unknown_environment_is_rejected():
    with pytest.raises(ConfigError, match="env_type='other'"):
        _config(env_type="other")


def test_tabnn_hyperopt_is_rejected():
    with pytest.raises(ConfigError, match="only for boosting"):
        _config(backend="tabnn", engine="ste", device="gpu", hyperopt=True)


@pytest.mark.parametrize("field_name", BinaryTaskConfig.required_explicit_fields)
def test_mapping_requires_every_explicit_public_field(field_name):
    payload = asdict(_config())
    payload.pop(field_name)

    with pytest.raises(ConfigError, match=rf"Missing required .*{field_name}"):
        BinaryTaskConfig.from_mapping(payload)


def test_mapping_requires_model_layout_only_with_group_column():
    payload = asdict(_config())
    payload.pop("model_layout")

    with pytest.raises(ConfigError, match="model_layout=None.*group_column='group'"):
        BinaryTaskConfig.from_mapping(payload)

    payload["group_column"] = None
    assert BinaryTaskConfig.from_mapping(payload).resolved_model_layout == "global"


def test_environment_is_optional_and_defaults_to_the_shared_fmlib_venv():
    values = asdict(_config())
    values.pop("environment")

    expected = "/home/datalab/nfs/sber-amazme-fmlib/env"
    assert BinaryTaskConfig(**values).environment.venv_path == expected
    assert BinaryTaskConfig.from_mapping(values).environment.venv_path == expected
    assert BinaryTaskConfig(**(values | {"env_type": "osiris", "device": "gpu"})).environment.venv_path == expected
    assert BinaryTaskConfig.from_mapping(values | {"env_type": "osiris", "device": "gpu"}).environment.venv_path == expected

    with pytest.raises(ConfigError, match="cannot be None"):
        EnvironmentConfig(venv_path=None)


def test_static_validation_rejects_arguments_without_effect():
    with pytest.raises(ConfigError, match="search_space.*hyperopt=True"):
        _config(search_space={"depth": [2]})
    with pytest.raises(ConfigError, match="n_trials can be configured only when hyperopt=True"):
        _config(n_trials=2)
    with pytest.raises(ConfigError, match="model_params cannot be configured when hyperopt=True"):
        _config(hyperopt=True, model_params={"depth": 2})
