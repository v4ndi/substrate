"""Separate path-based calibration lifecycle contracts."""

import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from avatar.automl import (
    BinaryTask,
    BinaryTaskConfig,
    CalibrationResult,
    EvaluationResult,
    MulticlassTask,
    PredictionResult,
    RegressionTask,
    ResponseTask,
    ResponseTaskConfig,
    UpliftTask,
    UpliftTaskConfig,
)
from avatar.automl.backends.boosting.uplift import UPLIFT_SCORE_COLUMNS
from avatar.automl.exceptions import ArtifactIntegrityError, ConfigError, SchemaError


def _base(config_class, tmp_path, **updates):
    values = {
        "env_type": "local",
        "backend": "boosting",
        "engine": "xgboost",
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ("x",),
        "hidden_state_columns": (),
        "model_layout": "global_and_per_group",
        "hyperopt": False,
        "output_dir": tmp_path,
        "environment": {},
        "verbose": False,
    }
    if config_class in {ResponseTaskConfig, UpliftTaskConfig}:
        values.update(treatment_column="treatment", inverse_treatment=False)
    if config_class is UpliftTaskConfig:
        values["estimate_propensity"] = False
    values.update(updates)
    return config_class(**values)


def _persist_prediction(task, data_path, scores):
    operation = task._store.start_operation("predict", dataset_path=data_path, runtime={"env_type": "local"})
    result_path = task._store.persist_prediction(data_path, PredictionResult(scores))
    task._store.update_operation(operation, state="succeeded", result_path=str(result_path))
    return result_path


def _response_data(start, month, rows=8):
    index = np.arange(rows)
    return pl.DataFrame(
        {
            "epk_id": start + index,
            "report_month": [month] * rows,
            "group": np.where(index % 2 == 0, "a", "b"),
            "treatment": (index // 2) % 2,
            "x": np.linspace(-1.0, 1.0, rows),
            "target": (index % 4 >= 2).astype(int),
        }
    )


def _combined_scalar_scores(data, values):
    identity = data.select("epk_id", "report_month", "group")
    return pl.concat(
        [
            identity.with_columns(pl.Series("score", values), pl.lit("global").alias("model_layout")),
            identity.with_columns(pl.Series("score", values[::-1]), pl.lit("per_group").alias("model_layout")),
        ]
    )


def test_only_response_and_uplift_expose_calibration(tmp_path):
    assert "calibrate" not in BinaryTask.__dict__
    assert not hasattr(RegressionTask, "calibrate")
    assert not hasattr(MulticlassTask, "calibrate")
    assert hasattr(ResponseTask, "calibrate") and hasattr(UpliftTask, "calibrate")
    assert "calibration_paths" not in inspect.signature(ResponseTask.predict).parameters
    assert "calibration_windows" not in BinaryTaskConfig.__dataclass_fields__
    result = CalibrationResult(pl.DataFrame({"score": [0.5]}), "beta_calibration")
    task = BinaryTask(_base(BinaryTaskConfig, tmp_path))
    task._models = [object()]
    with pytest.raises(ConfigError, match="does not accept CalibrationResult"):
        task.evaluate(tmp_path / "missing.parquet", result)
    response = ResponseTask(_base(ResponseTaskConfig, tmp_path / "response"))
    with pytest.raises(ConfigError, match="test_path must be a parquet path"):
        response.calibrate(pl.DataFrame({"score": [0.5]}), tmp_path / "reference.parquet")


@pytest.mark.parametrize("strategy", ["beta_calibration", "isotonic_regression"])
def test_response_calibration_is_branch_local_path_based_and_persistent(tmp_path, strategy):
    task = ResponseTask(_base(ResponseTaskConfig, tmp_path))
    test_path, calibration_path = tmp_path / "test.parquet", tmp_path / "calibration.parquet"
    test_data = _response_data(100, "2026-03")
    calibration_data = _response_data(0, "2026-02")
    test_data.write_parquet(test_path)
    calibration_data.write_parquet(calibration_path)
    scores = _combined_scalar_scores(test_data, np.linspace(0.1, 0.8, test_data.height))
    calibration_scores = _combined_scalar_scores(
        calibration_data,
        np.array([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95]),
    )
    _persist_prediction(task, test_path, scores)
    _persist_prediction(task, calibration_path, calibration_scores)

    result = task.calibrate(test_path, calibration_path, calibration_strategy=strategy)

    assert isinstance(result, CalibrationResult)
    assert result.calibration_strategy == strategy
    assert result.scores["epk_id"].to_list() == scores["epk_id"].to_list()
    assert result.scores["model_layout"].to_list() == scores["model_layout"].to_list()
    assert_frame_equal(task.load_calibration(test_path).scores, result.scores)
    _, key = task._store.dataset(test_path, create=False)
    metadata = json.loads((task._store.calibration_dir(key) / "calibration.json").read_text())
    assert metadata["test_path"] == str(test_path.resolve())
    assert metadata["calibration_path"] == str(calibration_path.resolve())
    assert set(metadata["state"]["branches"]) == {"global", "per_group"}
    assert not (task.path / "artifact").exists()

    task._models = [object()]
    task._execute_evaluate = lambda test_path, selected, kind, metric_names: EvaluationResult.for_kind(
        kind,
        metrics={"selected": float(kind == "calibrated")},
        metrics_by_group=pl.DataFrame({"scope": ["group"], "selected": [kind]}),
    )
    raw_evaluation = task.evaluate(test_path)
    evaluation = task.evaluate(test_path, result)
    loaded = task.load_evaluation(test_path)
    _, truth_key = task._store.dataset(test_path, create=False)
    payload = json.loads((task._store.evaluation_dir(truth_key) / "evaluation_result.json").read_text())

    assert raw_evaluation.metrics_raw == {"selected": 0.0}
    assert raw_evaluation.metrics_calibrated is None
    assert evaluation.metrics_raw == {"selected": 0.0}
    assert evaluation.metrics_calibrated == {"selected": 1.0}
    assert loaded.metrics_raw == {"selected": 0.0}
    assert loaded.metrics_calibrated == {"selected": 1.0}
    assert loaded.metrics_by_group_raw["selected"].to_list() == ["raw"]
    assert loaded.metrics_by_group_calibrated["selected"].to_list() == ["calibrated"]
    assert payload["metrics_raw"] == {"selected": 0.0}
    assert payload["metrics_calibrated"] == {"selected": 1.0}
    assert set(payload["metric_tables"]) == {
        "metrics_by_group_raw",
        "metrics_by_group_calibrated",
        "metrics_by_class_raw",
        "metrics_by_class_calibrated",
    }

    calibrated_path = Path(metadata["result_path"])
    assert task.evaluate(test_path, calibrated_path).metrics_calibrated == {"selected": 1.0}
    unregistered_path = tmp_path / "unregistered.parquet"
    scores.write_parquet(unregistered_path)
    with pytest.raises(ConfigError, match="Cannot determine whether scores_path"):
        task.evaluate(test_path, unregistered_path)


def test_response_public_predict_then_calibrate_lifecycle(tmp_path, monkeypatch):
    paths = [tmp_path / f"{name}.parquet" for name in ("train", "valid", "calibration", "test")]
    for index, path in enumerate(paths):
        _response_data(index * 100, f"2026-0{index + 1}", rows=40).write_parquet(path)
    task = ResponseTask(
        _base(
            ResponseTaskConfig,
            tmp_path / "lifecycle",
            model_layout="global",
            model_params={"n_estimators": 4, "max_depth": 2},
        )
    )
    task.train(paths[0], paths[1])
    calibration_prediction = task.predict(paths[2])
    test_prediction = task.predict(paths[3])
    monkeypatch.setattr(task, "_execute_predict", lambda *args, **kwargs: pytest.fail("calibrate must not predict"))

    calibrated = task.calibrate(paths[3], paths[2], calibration_strategy="isotonic_regression")

    assert calibration_prediction.scores.height == 40
    assert test_prediction.scores.height == calibrated.scores.height == 40
    assert_frame_equal(task.load_calibration(paths[3]).scores, calibrated.scores)


def test_calibration_requires_predictions_for_both_dataset_paths(tmp_path):
    task = ResponseTask(_base(ResponseTaskConfig, tmp_path))
    test_path, calibration_path = tmp_path / "test.parquet", tmp_path / "calibration.parquet"
    _response_data(100, "2026-03").write_parquet(test_path)
    _response_data(0, "2026-02").write_parquet(calibration_path)
    with pytest.raises(ConfigError, match="must identify different datasets"):
        task.calibrate(test_path, test_path)
    with pytest.raises(ArtifactIntegrityError, match=r"succeeded predict\(test_path\)"):
        task.calibrate(test_path, calibration_path)
    _persist_prediction(
        task,
        test_path,
        _response_data(100, "2026-03").select("epk_id", "report_month").with_columns(pl.lit(0.5).alias("score")),
    )
    with pytest.raises(ArtifactIntegrityError, match=r"succeeded predict\(calibration_path\)"):
        task.calibrate(test_path, calibration_path)


def test_response_calibration_validates_strategy_branches_and_values(tmp_path):
    task = ResponseTask(_base(ResponseTaskConfig, tmp_path))
    test_path, calibration_path = tmp_path / "test.parquet", tmp_path / "calibration.parquet"
    with pytest.raises(ConfigError, match="Unsupported calibration_strategy"):
        task.calibrate(test_path, calibration_path, calibration_strategy="beta")
    test_data = _response_data(100, "2026-03")
    calibration_data = _response_data(0, "2026-02")
    test_data.write_parquet(test_path)
    calibration_data.write_parquet(calibration_path)
    _persist_prediction(
        task,
        test_path,
        test_data.select("epk_id", "report_month").with_columns(
            pl.lit(0.2).alias("score"), pl.lit("per_group").alias("model_layout")
        ),
    )
    _persist_prediction(
        task,
        calibration_path,
        calibration_data.select("epk_id", "report_month").with_columns(
            pl.lit(0.8).alias("score"), pl.lit("global").alias("model_layout")
        ),
    )
    with pytest.raises(SchemaError, match="missing model branches"):
        task.calibrate(test_path, calibration_path)


def test_uplift_calibration_fits_each_arm_inside_each_branch_and_recomputes_effects(tmp_path):
    task = UpliftTask(_base(UpliftTaskConfig, tmp_path))
    rows = 8
    treatment = np.tile([0, 0, 1, 1], 2)
    target = np.tile([0, 1, 0, 1], 2)
    calibration_data = pl.DataFrame(
        {
            "epk_id": np.arange(rows),
            "report_month": ["2026-02"] * rows,
            "group": ["a", "b"] * 4,
            "x": np.linspace(-1.0, 1.0, rows),
            "target": target,
            "treatment": treatment,
        }
    )
    test_data = calibration_data.drop("target", "treatment").with_columns(
        (pl.col("epk_id") + 100).alias("epk_id"), pl.lit("2026-03").alias("report_month")
    )
    calibration_path, test_path = tmp_path / "uplift-calibration.parquet", tmp_path / "uplift-test.parquet"
    calibration_data.write_parquet(calibration_path)
    test_data.write_parquet(test_path)

    def uplift_scores(data):
        base = np.linspace(0.05, 0.95, data.height)
        identity = data.select("epk_id", "report_month", "group")
        branches = []
        for layout in ("global", "per_group"):
            branch = identity.with_columns(
                *[
                    pl.Series(column, base + index * 0.001)
                    for index, column in enumerate(UPLIFT_SCORE_COLUMNS)
                    if column not in {"score_s", "score_t", "score_x"}
                ],
                pl.lit(layout).alias("model_layout"),
            ).with_columns(
                (pl.col("score_s_treatment") - pl.col("score_s_control")).alias("score_s"),
                (pl.col("score_t_treatment") - pl.col("score_t_control")).alias("score_t"),
                (pl.col("score_x_treatment") - pl.col("score_x_control")).alias("score_x"),
            )
            branches.append(branch)
        return pl.concat(branches)

    _persist_prediction(task, test_path, uplift_scores(test_data))
    _persist_prediction(task, calibration_path, uplift_scores(calibration_data))

    result = task.calibrate(test_path, calibration_path, calibration_strategy="isotonic_regression")

    for effect, control, treated in (
        ("score_s", "score_s_control", "score_s_treatment"),
        ("score_t", "score_t_control", "score_t_treatment"),
        ("score_x", "score_x_control", "score_x_treatment"),
    ):
        np.testing.assert_allclose(result.scores[effect], result.scores[treated] - result.scores[control])


def test_remote_calibration_plans_one_job_per_branch_and_finalizes_in_source_order(tmp_path, monkeypatch):
    task = ResponseTask(
        _base(
            ResponseTaskConfig,
            tmp_path,
            env_type="osiris",
            device="gpu",
            environment={"venv_path": tmp_path},
        )
    )
    test_path, calibration_path = tmp_path / "test.parquet", tmp_path / "calibration.parquet"
    test_data = _response_data(100, "2026-03")
    calibration_data = _response_data(0, "2026-02")
    test_data.write_parquet(test_path)
    calibration_data.write_parquet(calibration_path)
    _persist_prediction(task, test_path, _combined_scalar_scores(test_data, np.linspace(0.1, 0.8, 8)))
    _persist_prediction(
        task,
        calibration_path,
        _combined_scalar_scores(calibration_data, np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])),
    )
    submitted = []

    def submit(*, config, action, payload, run_dir):
        submitted.append(payload)
        return {"result_path": str(run_dir / "result.json"), "spec_path": str(run_dir / "run_spec.json")}

    monkeypatch.setattr(task._runner(), "submit", submit)
    assert task.calibrate(test_path, calibration_path) is None
    operation = task._store.latest("calibrate", test_path)
    assert len(submitted) == 2
    assert {payload["remote_layout"] for payload in submitted} == {"global", "per_group"}
    assert all("remote_group_value" not in payload and "treatment" not in payload for payload in submitted)

    jobs = []
    for payload, job in zip(submitted, operation["jobs"], strict=True):
        layout = payload["remote_layout"]
        output = task._execute_calibrate(
            payload["test_scores_path"],
            payload["calibration_scores_path"],
            payload["calibration_path"],
            "beta_calibration",
            remote_layout=layout,
            include_row_id=True,
        )
        Path(payload["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        output.result.scores.write_parquet(payload["output_path"])
        result_path = Path(job["result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "scores_path": payload["output_path"],
                    "calibration_strategy": "beta_calibration",
                    "state": output.state,
                }
            )
        )
        spec_path = Path(job["spec_path"])
        spec_path.write_text(json.dumps({"payload": payload}))
        jobs.append(job)
    operation = task._store.update_operation(operation, jobs=jobs)
    task._operation_runner()._finalize_remote_calibration(operation)
    task._store.update_operation(operation, state="succeeded")
    assert task._store.load_calibration(test_path).scores["epk_id"].to_list() == (
        test_data["epk_id"].to_list() * 2
    )
