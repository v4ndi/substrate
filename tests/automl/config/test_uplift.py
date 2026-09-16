"""Configuration contracts for uplift AutoML."""

from __future__ import annotations

import pytest

from avatar.automl import UpliftTaskConfig
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
        "treatment_column": "treatment",
        "inverse_treatment": True,
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ["x"],
        "hidden_state_columns": (),
        "model_layout": "global",
        "hyperopt": False,
        "output_dir": "outputs",
        "environment": {},
        "estimate_propensity": False,
    } | overrides


def test_uplift_defaults_and_mapping_equivalence() -> None:
    direct = UpliftTaskConfig(**_values())
    mapped = UpliftTaskConfig.from_mapping({"task": _values()})
    assert direct == mapped
    assert direct.optimization_metric == "qini_auc"
    assert direct.estimate_propensity is False


@pytest.mark.parametrize("metric", ["qini_auc", "uplift_auc"])
def test_uplift_named_metrics(metric: str) -> None:
    config = UpliftTaskConfig(**_values(optimization_metric=metric))
    assert config.optimization_metric == metric


@pytest.mark.parametrize("metric", ["roc_auc", "unknown", "uplift_at_10"])
def test_uplift_rejects_invalid_optimization_metric(metric: str) -> None:
    with pytest.raises(
        ConfigError, match="Unknown metric|incompatible|only for evaluation"
    ):
        UpliftTaskConfig(**_values(optimization_metric=metric))


def test_uplift_rejects_callable_metric() -> None:
    with pytest.raises(ConfigError, match="non-empty string"):
        UpliftTaskConfig(**_values(optimization_metric=lambda _target, _score: 0.0))


def test_uplift_treatment_is_not_a_public_covariate() -> None:
    with pytest.raises(ConfigError, match="learner-controlled"):
        UpliftTaskConfig(**_values(categorical_columns=["treatment"]))


def test_uplift_requires_treatment_role() -> None:
    with pytest.raises(ConfigError, match="requires treatment_column"):
        UpliftTaskConfig(**_values(treatment_column=None))


def test_uplift_mapping_requires_explicit_estimate_propensity() -> None:
    values = _values()
    values.pop("estimate_propensity")
    with pytest.raises(ConfigError, match="estimate_propensity"):
        UpliftTaskConfig.from_mapping(values)
