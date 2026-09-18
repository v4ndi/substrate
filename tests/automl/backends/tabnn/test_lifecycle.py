"""The full lifecycle on a network: train, predict, calibrate, evaluate, report.

Binary and multiclass proved the chain works. This is the rest of the public
surface on the other two supervised tasks, and it is the check that nothing
downstream of the backend — calibration, evaluation, the report — cares which
family produced the scores.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fmlib.automl import (
    RegressionTask,
    RegressionTaskConfig,
    ResponseTask,
    ResponseTaskConfig,
)

pytestmark = pytest.mark.slow

SMALL = {
    "max_epochs": 2,
    "batch_size": 128,
    "hidden_size": 16,
    "num_layers": 1,
    "num_heads": 2,
    "num_workers": 0,
    "evaluations_per_epoch": 1,
    "patience": 2,
}


def _write(directory, rows: int, seed: int, month: str = "2025-01-01"):
    rng = np.random.default_rng(seed)
    balance = rng.normal(size=rows)
    treatment = rng.integers(0, 2, rows)
    logit = 1.4 * balance + 0.8 * treatment
    directory.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series([month] * rows).str.to_date(),
        "segment": rng.choice(["a", "b", "c"], rows),
        "balance": balance,
        "treatment": treatment.astype(np.int8),
        "y_binary": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
        "y_reg": logit + rng.normal(scale=0.2, size=rows),
    })
    half = rows // 2
    frame.head(half).write_parquet(directory / "part-0.parquet")
    frame.tail(rows - half).write_parquet(directory / "part-1.parquet")
    return directory


@pytest.fixture
def data(tmp_path):
    return {
        "train": _write(tmp_path / "train", 600, 1),
        "valid": _write(tmp_path / "valid", 300, 2),
        "test": _write(tmp_path / "test", 300, 3, month="2025-03-01"),
        "calibration": _write(tmp_path / "calib", 300, 4, month="2025-02-01"),
    }


def _response_config(tmp_path):
    return ResponseTaskConfig(
        env_type="local",
        backend="tabnn",
        engine="tabular_transformer",
        device="cpu",
        target_column="y_binary",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hidden_state_columns=[],
        treatment_column="treatment",
        inverse_treatment=False,
        hyperopt=False,
        model_params=dict(SMALL),
        verbose=False,
        output_dir=tmp_path / "response",
        environment={},
    )


def test_response_runs_train_predict_calibrate_evaluate(tmp_path, data):
    task = ResponseTask(_response_config(tmp_path))
    training = task.train(data["train"], data["valid"])
    assert training.backend_name == "tabnn"

    task.predict(data["test"])
    task.predict(data["calibration"])
    calibrated = task.calibrate(data["test"], data["calibration"])
    assert calibrated.calibration_strategy == "beta_calibration"
    assert calibrated.scores.height == 300
    assert calibrated.scores["score"].is_between(0.0, 1.0).all()

    # Evaluating a CalibrationResult fills the *_calibrated fields, which is
    # what tells the two evaluations of one test path apart.
    evaluation = task.evaluate(data["test"], calibrated)
    assert evaluation.metrics_calibrated
    assert np.isfinite(list(evaluation.metrics_calibrated.values())).all()
    assert evaluation.metrics_raw is None

    raw = task.evaluate(data["test"])
    assert raw.metrics_raw
    assert np.isfinite(list(raw.metrics_raw.values())).all()


def test_the_treatment_column_is_an_ordinary_feature_of_a_response_model(
    tmp_path, data
):
    task = ResponseTask(_response_config(tmp_path))
    task.train(data["train"], data["valid"])
    entry = task._models[0]
    assert "treatment" in entry.schema.categorical
    assert entry.backend.task_name == "response"
    assert entry.backend.score_transform == "sigmoid"


def test_regression_runs_the_whole_lifecycle_and_writes_a_report(tmp_path, data):
    config = RegressionTaskConfig(
        env_type="local",
        backend="tabnn",
        engine="tabular_transformer",
        device="cpu",
        target_column="y_reg",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hidden_state_columns=[],
        hyperopt=False,
        model_params=dict(SMALL),
        optimization_metric="mse",
        verbose=False,
        output_dir=tmp_path / "regression",
        environment={},
    )
    task = RegressionTask(config)
    task.train(data["train"], data["valid"])

    prediction = task.predict(data["test"])
    assert prediction.scores.height == 300
    assert prediction.scores["score"].dtype.is_float()

    evaluation = task.evaluate(data["test"], prediction)
    assert "mse" in evaluation.metrics_raw
    reports = list((tmp_path / "regression").rglob("*.xlsx"))
    assert reports, "evaluation must leave a report behind"


def test_a_regression_model_scores_values_not_probabilities(tmp_path, data):
    config = RegressionTaskConfig(
        env_type="local",
        backend="tabnn",
        engine="tabular_transformer",
        device="cpu",
        target_column="y_reg",
        client_id_column="epk_id",
        group_column=None,
        date_column=None,
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hidden_state_columns=[],
        hyperopt=False,
        model_params=dict(SMALL),
        optimization_metric="mse",
        verbose=False,
        output_dir=tmp_path / "regression2",
        environment={},
    )
    task = RegressionTask(config)
    task.train(data["train"], data["valid"])
    assert task._models[0].backend.score_transform == "identity"
    scores = task.predict(data["test"]).scores["score"].to_numpy()
    assert np.isfinite(scores).all()


def test_the_artifact_survives_a_reload_for_every_supervised_task(tmp_path, data):
    task = ResponseTask(_response_config(tmp_path))
    task.train(data["train"], data["valid"])
    before = task.predict(data["test"])
    restored = ResponseTask.load(task.save(tmp_path / "artifact"))
    assert restored.predict(data["test"]).scores.equals(before.scores)
