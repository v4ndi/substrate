"""Configuration contracts for binary response modelling."""

from dataclasses import asdict
from inspect import signature

import pytest

from avatar.automl import BinaryTaskConfig, ResponseTaskConfig
from avatar.automl.exceptions import ConfigError


def _values(**overrides):
    return {
        "env_type": "local",
        "backend": "boosting",
        "engine": "xgboost",
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ["x"],
        "hidden_state_columns": (),
        "model_layout": "global",
        "hyperopt": False,
        "output_dir": "outputs",
        "environment": {},
    } | overrides


def test_treatment_fields_belong_only_to_response_config() -> None:
    assert "treatment_column" not in signature(BinaryTaskConfig).parameters
    assert "inverse_treatment" not in signature(BinaryTaskConfig).parameters
    with pytest.raises(TypeError, match="treatment_column"):
        BinaryTaskConfig(**_values(treatment_column=None))
    with pytest.raises(ConfigError, match="Unknown BinaryTaskConfig.*treatment_column"):
        BinaryTaskConfig.from_mapping(_values(treatment_column=None))


def test_response_without_treatment_omits_inverse_semantics() -> None:
    config = ResponseTaskConfig(**_values())
    assert config.treatment_column is None
    assert config.inverse_treatment is None

    payload = asdict(config)
    payload.pop("inverse_treatment")
    assert ResponseTaskConfig.from_mapping(payload) == config
    with pytest.raises(ConfigError, match="must be omitted"):
        ResponseTaskConfig.from_mapping(payload | {"inverse_treatment": False})


@pytest.mark.parametrize("inverse", [False, True])
def test_response_with_treatment_requires_boolean_inverse(inverse: bool) -> None:
    config = ResponseTaskConfig(
        **_values(treatment_column="treatment", inverse_treatment=inverse)
    )
    assert config.treatment_column == "treatment"
    assert config.inverse_treatment is inverse

    payload = asdict(config)
    assert ResponseTaskConfig.from_mapping(payload) == config


def test_response_with_treatment_rejects_missing_or_null_inverse() -> None:
    values = _values(treatment_column="treatment")
    with pytest.raises(ConfigError, match="inverse_treatment=None"):
        ResponseTaskConfig(**values)
    with pytest.raises(ConfigError, match="inverse_treatment=None"):
        ResponseTaskConfig.from_mapping(values)


def test_response_python_yaml_parity(tmp_path) -> None:
    path = tmp_path / "response.yaml"
    path.write_text(
        """
env_type: local
backend: boosting
engine: xgboost
device: cpu
target_column: target
client_id_column: epk_id
group_column: group
treatment_column: treatment
inverse_treatment: false
date_column: report_month
categorical_columns: []
numerical_columns: [x]
hidden_state_columns: []
model_layout: global
hyperopt: false
output_dir: outputs
environment: {}
""",
        encoding="utf-8",
    )
    assert ResponseTaskConfig.from_yaml(path) == ResponseTaskConfig(
        **_values(treatment_column="treatment", inverse_treatment=False)
    )
