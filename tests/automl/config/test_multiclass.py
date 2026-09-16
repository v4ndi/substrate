from pathlib import Path

import pytest

from avatar.automl import BinaryTaskConfig, MulticlassTaskConfig, RegressionTaskConfig
from avatar.automl.config.base import BaseTaskConfig
from avatar.automl.exceptions import ConfigError


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
    return MulticlassTaskConfig(**values)


def test_multiclass_defaults_and_objective_directions():
    assert _config().optimization_metric == "roc_auc_ovr_macro"
    assert _config().model_layout == "global"
    assert _config().random_state == 42


def test_multiclass_config_is_an_independent_task_config_sibling():
    assert MulticlassTaskConfig.__bases__ == (BaseTaskConfig,)
    assert not issubclass(MulticlassTaskConfig, BinaryTaskConfig | RegressionTaskConfig)


@pytest.mark.parametrize(
    "metric", ["roc_auc_ovr_macro", "accuracy", "f1_macro", "log_loss"]
)
def test_multiclass_supported_objective_metrics(metric):
    assert _config(optimization_metric=metric).optimization_metric == metric


def test_multiclass_python_yaml_and_feature_sources_are_equivalent(tmp_path):
    cat_path = (tmp_path / "cat.yaml").resolve()
    num_path = (tmp_path / "num.yaml").resolve()
    cat_path.write_text("categorical_columns: [segment, city]\n", encoding="utf-8")
    num_path.write_text("- balance\n- age\n", encoding="utf-8")
    yaml_path = tmp_path / "multiclass.yaml"
    yaml_path.write_text(
        f"""
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
  categorical_columns: {cat_path.as_posix()}
  numerical_columns: {num_path.as_posix()}
  hidden_state_columns: []
train:
  model_layout: global
  optimization_metric: log_loss
  hyperopt: true
  n_trials: 3
  search_space:
    depth: {{type: int, low: 3, high: 7, step: 2}}
    learning_rate: {{type: float, low: 0.1, high: 0.3, step: 0.1}}
    grow_policy: {{type: categorical, choices: [SymmetricTree, Depthwise]}}
output_dir: outputs
environment: {{}}
""",
        encoding="utf-8",
    )
    expected = _config(
        categorical_columns=cat_path,
        numerical_columns=num_path,
        optimization_metric="log_loss",
        hyperopt=True,
        n_trials=3,
        search_space={
            "depth": {"type": "int", "low": 3, "high": 7, "step": 2},
            "learning_rate": {"type": "float", "low": 0.1, "high": 0.3, "step": 0.1},
            "grow_policy": {
                "type": "categorical",
                "choices": ["SymmetricTree", "Depthwise"],
            },
        },
    )
    assert MulticlassTaskConfig.from_yaml(yaml_path) == expected
    assert expected.categorical_columns == ("segment", "city")
    assert expected.numerical_columns == ("balance", "age")


@pytest.mark.parametrize("metric", ["roc_auc", "mse", "weighted_f1", "unknown"])
def test_multiclass_unknown_metric_is_rejected(metric):
    with pytest.raises(ConfigError, match="Unknown metric|incompatible"):
        _config(optimization_metric=metric)


def test_multiclass_callable_metric_is_rejected():
    with pytest.raises(ConfigError, match="non-empty string"):
        _config(optimization_metric=lambda _target, _probabilities: 0.5)


@pytest.mark.parametrize("field", ["verbose", "verbosity", "logging_level", "silent"])
def test_multiclass_runtime_parameters_cannot_be_hidden(field):
    with pytest.raises(ConfigError, match="Task-level runtime parameters"):
        _config(model_params={field: 1})
    with pytest.raises(ConfigError, match="Task-level runtime parameters"):
        _config(search_space={field: [1]}, hyperopt=True)


def test_multiclass_absolute_feature_path_is_required(tmp_path):
    relative = Path("features.yaml")
    with pytest.raises(ConfigError, match="must be absolute"):
        _config(categorical_columns=relative)
