import pytest

from fmlib.automl import RegressionTaskConfig
from fmlib.automl.exceptions import ConfigError


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
    return RegressionTaskConfig(**values)


def test_regression_defaults_and_python_yaml_equivalence(tmp_path):
    yaml_path = tmp_path / "regression.yaml"
    yaml_path.write_text(
        """
task:
  env_type: local
  backend: boosting
  engine: catboost
  device: cpu
data:
  target_column: target
  client_id_column: epk_id
  group_column: group
  date_column: report_month
  categorical_columns: [segment]
  numerical_columns: [balance]
  hidden_state_columns: []
train:
  model_layout: global
  hyperopt: true
  n_trials: 3
  optimization_metric: mae
output_dir: outputs
environment: {}
""",
        encoding="utf-8",
    )

    assert _config().optimization_metric == "mse"
    assert RegressionTaskConfig.from_yaml(yaml_path) == _config(
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hyperopt=True,
        n_trials=3,
        optimization_metric="mae",
    )


@pytest.mark.parametrize("metric", ["rmse", "roc_auc", "MAPE"])
def test_regression_rejects_unknown_named_metric(metric):
    with pytest.raises(ConfigError, match="Unknown metric|incompatible"):
        _config(optimization_metric=metric)


def test_callable_metric_is_not_a_public_config_value():
    with pytest.raises(ConfigError, match="non-empty string"):
        _config(optimization_metric=lambda _target, _score: 0.0)


@pytest.mark.parametrize("field", ["verbose", "verbosity", "logging_level", "silent"])
def test_task_runtime_settings_cannot_be_hidden_in_model_params_or_search_space(field):
    with pytest.raises(ConfigError, match="Task-level runtime parameters"):
        _config(model_params={field: 1})
    with pytest.raises(ConfigError, match="Task-level runtime parameters"):
        _config(search_space={field: [1]}, hyperopt=True)
