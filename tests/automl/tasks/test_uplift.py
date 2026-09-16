"""Uplift task architecture, metrics and validation contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from avatar.automl import (
    BinaryTask,
    MulticlassTask,
    RegressionTask,
    UpliftTask,
    UpliftTaskConfig,
)
from avatar.automl.backends.boosting.binary import BinaryBoostingBackend
from avatar.automl.backends.boosting.regression import RegressionBoostingBackend
from avatar.automl.backends.boosting.uplift import UpliftBoostingBackend
from avatar.automl.exceptions import ConfigError, SchemaError
from avatar.automl.metrics import qini_auc_score, uplift_at_k, uplift_auc_score


def test_uplift_task_and_backend_are_independent_siblings() -> None:
    assert not issubclass(UpliftTask, BinaryTask | RegressionTask | MulticlassTask)
    assert not issubclass(
        UpliftBoostingBackend, BinaryBoostingBackend | RegressionBoostingBackend
    )


def test_uplift_metrics_are_finite_and_stable_for_ties() -> None:
    target = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    score = np.array([0.4, 0.4, 0.2, 0.2, 0.1, 0.1, -0.2, -0.2])
    assert np.isfinite(qini_auc_score(target, score, treatment))
    assert np.isfinite(uplift_auc_score(target, score, treatment))
    assert uplift_at_k(target, score, treatment, "overall", 0.5) == 0.0
    permutation = np.array([1, 0, 3, 2, 5, 4, 7, 6])
    assert qini_auc_score(target, score, treatment) == pytest.approx(
        qini_auc_score(target[permutation], score[permutation], treatment[permutation])
    )


@pytest.fixture
def task() -> UpliftTask:
    return UpliftTask(
        UpliftTaskConfig(
            env_type="local",
            backend="boosting",
            engine="xgboost",
            device="cpu",
            target_column="target",
            client_id_column="epk_id",
            group_column=None,
            treatment_column="treatment",
            inverse_treatment=True,
            date_column="report_month",
            categorical_columns=(),
            numerical_columns=["x"],
            hidden_state_columns=(),
            hyperopt=False,
            output_dir="outputs",
            environment={},
            estimate_propensity=False,
        )
    )


@pytest.mark.parametrize("column", ["target", "treatment"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 2])
def test_binary_role_validation(task: UpliftTask, column: str, value) -> None:
    frame = pl.DataFrame({"target": [0.0, 1.0], "treatment": [0.0, 1.0]}).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )
    with pytest.raises(SchemaError):
        (task._target if column == "target" else task._treatment)(frame)


@pytest.mark.parametrize(
    "model_layout", ["per_group", "global", "global_and_per_group"]
)
def test_model_layout_is_forbidden_without_group_column(
    task: UpliftTask, tmp_path, model_layout: str
) -> None:
    with pytest.raises(
        ConfigError, match=rf"model_layout={model_layout!r}.*group_column=None"
    ):
        replace(
            task.config, model_layout=model_layout, output_dir=tmp_path / model_layout
        )
