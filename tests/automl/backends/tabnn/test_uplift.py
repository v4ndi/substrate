"""Uplift on a network: one S-Learner, scored twice, calibrated, evaluated.

Three columns where boosting reports ten. T- and X-metalearners are out of
scope, which is why the score columns are a function of the backend family
rather than one frozen tuple.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fmlib.automl import UpliftTask, UpliftTaskConfig
from fmlib.automl.uplift_scores import uplift_score_columns

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
    directory.mkdir(parents=True, exist_ok=True)
    balance = rng.normal(size=rows)
    treatment = rng.integers(0, 2, rows)
    # A real effect: treatment helps the high-balance half and not the rest.
    logit = 1.2 * balance + 1.5 * treatment * (balance > 0)
    frame = pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series([month] * rows).str.to_date(),
        "segment": rng.choice(["a", "b"], rows),
        "balance": balance,
        "treatment": treatment.astype(np.int8),
        "y": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
    })
    half = rows // 2
    frame.head(half).write_parquet(directory / "part-0.parquet")
    frame.tail(rows - half).write_parquet(directory / "part-1.parquet")
    return directory


@pytest.fixture
def data(tmp_path):
    return {
        "train": _write(tmp_path / "train", 800, 1),
        "valid": _write(tmp_path / "valid", 400, 2),
        "test": _write(tmp_path / "test", 400, 3, month="2025-03-01"),
        "calibration": _write(tmp_path / "calib", 400, 4, month="2025-02-01"),
    }


def _config(tmp_path, **overrides):
    values = {
        "env_type": "local",
        "backend": "tabnn",
        "engine": "tabular_transformer",
        "device": "cpu",
        "target_column": "y",
        "client_id_column": "epk_id",
        "group_column": None,
        "date_column": "report_month",
        "categorical_columns": ["segment"],
        "numerical_columns": ["balance"],
        "hidden_state_columns": [],
        "treatment_column": "treatment",
        "inverse_treatment": False,
        "estimate_propensity": False,
        "hyperopt": False,
        "model_params": dict(SMALL),
        "verbose": False,
        "output_dir": tmp_path / "outputs",
        "environment": {},
    }
    values.update(overrides)
    return UpliftTaskConfig(**values)


def test_the_column_set_is_a_function_of_the_backend_family():
    assert uplift_score_columns("tabnn") == (
        "score_s",
        "score_s_control",
        "score_s_treatment",
    )
    assert len(uplift_score_columns("boosting")) == 10


def test_an_uplift_network_trains_and_scores_three_columns(tmp_path, data):
    task = UpliftTask(_config(tmp_path))
    training = task.train(data["train"], data["valid"])

    assert training.backend_name == "tabnn"
    # The uplift report names a metric per learner, and there is one.
    assert set(training.validation_metrics) == {"global:s"}
    assert np.isfinite(training.validation_metrics["global:s"])

    prediction = task.predict(data["test"])
    columns = set(prediction.scores.columns)
    assert set(uplift_score_columns("tabnn")) <= columns
    assert not [name for name in columns if name.startswith(("score_t", "score_x"))]


def test_the_effect_is_exactly_treated_minus_control(tmp_path, data):
    """The model computes both arms itself, so the difference is its own."""
    task = UpliftTask(_config(tmp_path))
    task.train(data["train"], data["valid"])
    scores = task.predict(data["test"]).scores

    assert np.allclose(
        scores["score_s"].to_numpy(),
        scores["score_s_treatment"].to_numpy() - scores["score_s_control"].to_numpy(),
        atol=1e-10,
    )
    for column in ("score_s_control", "score_s_treatment"):
        assert scores[column].is_between(0.0, 1.0).all()


def test_the_treatment_is_the_learners_input_not_one_of_its_features(tmp_path, data):
    task = UpliftTask(_config(tmp_path))
    task.train(data["train"], data["valid"])
    entry = task._models[0]
    assert "treatment" not in entry.schema.categorical
    assert entry.backend.score_transform == "uplift"
    assert entry.backend.task_name == "uplift"


def test_uplift_runs_the_whole_lifecycle_including_calibration(tmp_path, data):
    task = UpliftTask(_config(tmp_path))
    task.train(data["train"], data["valid"])
    task.predict(data["test"])
    task.predict(data["calibration"])

    calibrated = task.calibrate(data["test"], data["calibration"])
    assert set(uplift_score_columns("tabnn")) <= set(calibrated.scores.columns)
    assert np.allclose(
        calibrated.scores["score_s"].to_numpy(),
        calibrated.scores["score_s_treatment"].to_numpy()
        - calibrated.scores["score_s_control"].to_numpy(),
        atol=1e-10,
    )

    evaluation = task.evaluate(data["test"], calibrated)
    assert evaluation.metrics_calibrated
    assert np.isfinite(list(evaluation.metrics_calibrated.values())).all()


def test_an_uplift_artifact_survives_a_reload(tmp_path, data):
    task = UpliftTask(_config(tmp_path))
    task.train(data["train"], data["valid"])
    before = task.predict(data["test"])
    restored = UpliftTask.load(task.save(tmp_path / "artifact"))
    assert restored.predict(data["test"]).scores.equals(before.scores)


def test_propensity_is_refused_because_there_is_no_x_learner(tmp_path):
    from fmlib.automl.exceptions import UnsupportedBackendError

    with pytest.raises(UnsupportedBackendError, match="S-Learner only"):
        _config(tmp_path, estimate_propensity=True)
