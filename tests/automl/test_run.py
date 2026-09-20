import json
from pathlib import Path

import polars as pl
import pytest

from fmlib.automl import (
    BinaryTaskConfig,
    EvaluationResult,
    MulticlassTaskConfig,
    PredictionResult,
    RegressionTaskConfig,
    ResponseTaskConfig,
    UpliftTaskConfig,
)
from fmlib.automl.run import execute_spec
from fmlib.automl.tasks.calibration import CalibrationOutput
from fmlib.automl.types import CalibrationResult


@pytest.mark.parametrize(
    ("task_name", "config_class"),
    [
        ("binary", BinaryTaskConfig),
        ("regression", RegressionTaskConfig),
        ("multiclass", MulticlassTaskConfig),
        ("uplift", UpliftTaskConfig),
    ],
)
@pytest.mark.parametrize("env_type", ["local", "osiris"])
def test_worker_dispatch_selects_registered_task_identity_without_changing_existing_defaults(
    tmp_path, monkeypatch, task_name, config_class, env_type
):
    captured = {}

    class FakeTask:
        def __init__(self, config, *, _entity_path=None):
            captured["config"] = config
            captured["entity_path"] = _entity_path

        def _execute_train(self, *args, **kwargs):
            return type("Training", (), {"best_params": {}, "validation_metrics": {}})()

        def _execution_view(self, context):
            captured["runtime_config"] = context.config
            return self

        def _adopt(self, execution):
            captured["adopted"] = execution

        def save(self, path):
            return path

    import fmlib.automl.run as worker

    config_type, _ = worker._TASK_TYPES[task_name]
    monkeypatch.setitem(worker._TASK_TYPES, task_name, (config_type, FakeTask))
    result_path = tmp_path / "result.json"
    config = {
        "env_type": env_type,
        "backend": "boosting",
        "engine": "catboost",
        "device": "cpu" if env_type == "local" else "gpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": [],
        "numerical_columns": ["feature"],
        "hidden_state_columns": [],
        "model_layout": "global",
        "hyperopt": False,
        "output_dir": str(tmp_path / "outputs"),
        "environment": {},
    }
    if task_name == "uplift":
        config.update(treatment_column="treatment", inverse_treatment=True)
        config["estimate_propensity"] = False
    payload = {
        "run_id": "run",
        "task": task_name,
        "action": "train",
        "config": config,
        "payload": {
            "train_path": "train",
            "valid_path": "valid",
            "artifact_path": str(tmp_path / "artifact"),
            "entity_config": config
            | {"environment": {"log_dir": str(tmp_path / "entity-logs")}},
        },
        "result_path": str(result_path),
        "log_path": str(tmp_path / "worker.log"),
    }
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    execute_spec(spec)

    assert isinstance(captured["config"], config_class)
    assert captured["config"].environment.log_dir == str(tmp_path / "entity-logs")
    assert captured["runtime_config"].environment.log_dir is None
    assert captured["runtime_config"].env_type == env_type
    assert "adopted" in captured
    assert captured["entity_path"] == tmp_path / "artifact"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "succeeded"
    worker_log = (tmp_path / "worker.log").read_text(encoding="utf-8")
    assert f"env_type={env_type} device={config['device']}" in worker_log
    assert "Finished action=train duration_seconds=" in worker_log


def test_prediction_worker_runs_only_the_requested_model_part(tmp_path, monkeypatch):
    captured = {}

    class FakeTask:
        def __init__(self, config, *, _entity_path=None):
            captured["config"] = config
            captured["entity_path"] = _entity_path

        def _restore_artifact(self, path):
            captured["restored"] = path

        def _select_remote_prediction_part(self, scope, group_value):
            captured["selected"] = (scope, group_value)

        def _execution_view(self, context, *, prepare_backends=False):
            captured["runtime_config"] = context.config
            assert prepare_backends
            return self

        def _validate_prediction_layout(self):
            captured["validated"] = True

        def _execute_predict(self, test_path, **kwargs):
            captured["prediction"] = (test_path, kwargs)
            return PredictionResult(
                pl.DataFrame({
                    "epk_id": [7],
                    "group": ["channel-a"],
                    "score": [0.75],
                    "__fmlib_remote_row_id": [2],
                })
            )

        @staticmethod
        def _prediction_storage_frame(prediction):
            return prediction.scores

    import fmlib.automl.run as worker

    monkeypatch.setitem(worker._TASK_TYPES, "binary", (BinaryTaskConfig, FakeTask))
    artifact_path = tmp_path / "artifact"
    scores_path = tmp_path / "scores.parquet"
    result_path = tmp_path / "result.json"
    config = {
        "env_type": "local",
        "backend": "boosting",
        "engine": "catboost",
        "device": "gpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": [],
        "numerical_columns": ["feature"],
        "hidden_state_columns": [],
        "model_layout": "global_and_per_group",
        "hyperopt": False,
        "output_dir": str(tmp_path / "outputs"),
        "environment": {},
    }
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({
            "run_id": "run",
            "task": "binary",
            "action": "predict",
            "config": config,
            "requested_config": config | {"env_type": "osiris"},
            "payload": {
                "test_path": "/data/test",
                "artifact_path": str(artifact_path),
                "scores_path": str(scores_path),
                "remote_layout": "per_group",
                "remote_group_value": "channel-a",
            },
            "result_path": str(result_path),
            "log_path": str(tmp_path / "worker.log"),
        }),
        encoding="utf-8",
    )

    result = execute_spec(spec)

    assert captured["config"].model_layout == "per_group"
    assert captured["entity_path"] == artifact_path
    assert captured["restored"] == artifact_path
    assert captured["selected"] == ("per_group", "channel-a")
    assert captured["runtime_config"].model_layout == "per_group"
    assert captured["validated"] is True
    assert captured["prediction"] == (
        "/data/test",
        {
            "remote_group_value": "channel-a",
            "include_row_id": True,
            "include_group": True,
        },
    )
    assert result["scores_path"] == str(scores_path)
    assert pl.read_parquet(scores_path)["__fmlib_remote_row_id"].to_list() == [2]


@pytest.mark.parametrize("kind", ["raw", "calibrated"])
def test_evaluation_worker_preserves_explicit_score_kind(tmp_path, monkeypatch, kind):
    captured = {}

    class FakeTask:
        def __init__(self, config, *, _entity_path=None):
            captured["entity_path"] = _entity_path

        def _restore_artifact(self, path):
            captured["restored"] = path

        def _execution_view(self, context, *, prepare_backends=False):
            assert not prepare_backends
            return self

        def _execute_evaluate(
            self, test_path, scores_path, evaluation_kind, metric_names
        ):
            captured["evaluate"] = (
                test_path,
                scores_path,
                evaluation_kind,
                metric_names,
            )
            return EvaluationResult.for_kind(
                evaluation_kind,
                metrics={"roc_auc": 0.75},
                metrics_by_group=pl.DataFrame({"scope": ["group"], "roc_auc": [0.75]}),
            )

    import fmlib.automl.run as worker

    monkeypatch.setitem(worker._TASK_TYPES, "response", (ResponseTaskConfig, FakeTask))
    result_path = tmp_path / "result.json"
    artifact_path = tmp_path / "artifact"
    scores_path = tmp_path / f"scores_{kind}.parquet"
    config = {
        "env_type": "osiris",
        "backend": "boosting",
        "engine": "catboost",
        "device": "gpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "treatment_column": "treatment",
        "inverse_treatment": False,
        "categorical_columns": [],
        "numerical_columns": ["feature"],
        "hidden_state_columns": [],
        "model_layout": "global",
        "hyperopt": False,
        "output_dir": str(tmp_path / "outputs"),
        "environment": {},
    }
    spec = tmp_path / f"evaluate_{kind}.json"
    spec.write_text(
        json.dumps({
            "run_id": f"evaluate-{kind}",
            "task": "response",
            "action": "evaluate",
            "config": config,
            "payload": {
                "test_path": "/data/test",
                "artifact_path": str(artifact_path),
                "scores_path": str(scores_path),
                "evaluation_kind": kind,
                "metrics": ["roc_auc"],
            },
            "result_path": str(result_path),
            "log_path": str(tmp_path / "worker.log"),
        }),
        encoding="utf-8",
    )

    result = execute_spec(spec)

    assert captured["entity_path"] == artifact_path
    assert captured["restored"] == artifact_path
    assert captured["evaluate"] == ("/data/test", str(scores_path), kind, ("roc_auc",))
    assert result[f"metrics_{kind}"] == {"roc_auc": 0.75}
    other = "calibrated" if kind == "raw" else "raw"
    assert result[f"metrics_{other}"] is None
    assert result["metric_tables"][f"metrics_by_group_{kind}"] is not None


def test_calibration_worker_writes_its_scores_and_reports_where(tmp_path, monkeypatch):
    """The one remote action with no test, and the only one that writes a file.

    `calibrate` is the fourth branch of the remote entrypoint and was the
    uncovered one. It differs from the others in that the driver learns the
    result by *path*: the scores are written here and read back by whoever
    collects the job, so the path in the result has to be the path that was
    written.
    """
    captured = {}

    class FakeTask:
        def __init__(self, config, *, _entity_path=None):
            captured["entity_path"] = _entity_path

        def _execute_calibrate(
            self,
            test_scores_path,
            calibration_scores_path,
            calibration_path,
            calibration_strategy,
            *,
            remote_layout=None,
            include_row_id=False,
        ):
            captured["calibrate"] = (
                test_scores_path,
                calibration_scores_path,
                calibration_path,
                calibration_strategy,
                remote_layout,
                include_row_id,
            )
            return CalibrationOutput(
                result=CalibrationResult(
                    scores=pl.DataFrame({"epk_id": [1, 2], "score": [0.25, 0.75]}),
                    calibration_strategy=calibration_strategy,
                ),
                state={"fitted_on": "calibration"},
            )

    import fmlib.automl.run as worker

    monkeypatch.setitem(worker._TASK_TYPES, "response", (ResponseTaskConfig, FakeTask))
    output_path = tmp_path / "nested" / "calibrated.parquet"
    result_path = tmp_path / "result.json"
    config = {
        "env_type": "osiris",
        "backend": "boosting",
        "engine": "catboost",
        "device": "gpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "treatment_column": "treatment",
        "inverse_treatment": False,
        "categorical_columns": [],
        "numerical_columns": ["feature"],
        "hidden_state_columns": [],
        "model_layout": "global",
        "hyperopt": False,
        "output_dir": str(tmp_path / "outputs"),
        "environment": {},
    }
    spec = tmp_path / "calibrate.json"
    spec.write_text(
        json.dumps({
            "run_id": "calibrate-1",
            "task": "response",
            "action": "calibrate",
            "config": config,
            "payload": {
                "test_scores_path": "/data/test_scores",
                "calibration_scores_path": "/data/calibration_scores",
                "calibration_path": "/data/calibration",
                "calibration_strategy": "isotonic_regression",
                "remote_layout": "global",
                "output_path": str(output_path),
            },
            "result_path": str(result_path),
            "log_path": str(tmp_path / "worker.log"),
        }),
        encoding="utf-8",
    )

    result = execute_spec(spec)

    assert captured["entity_path"] == Path(config["output_dir"])
    assert captured["calibrate"] == (
        "/data/test_scores",
        "/data/calibration_scores",
        "/data/calibration",
        "isotonic_regression",
        "global",
        True,
    )
    assert result["status"] == "succeeded"
    assert result["calibration_strategy"] == "isotonic_regression"
    assert result["state"] == {"fitted_on": "calibration"}

    # The directory did not exist: the worker has to make it, or the job fails
    # after the work is done and the result is lost.
    assert result["scores_path"] == str(output_path)
    assert pl.read_parquet(output_path).to_dict(as_series=False) == {
        "epk_id": [1, 2],
        "score": [0.25, 0.75],
    }
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_an_unknown_action_is_refused_and_recorded_as_failed(tmp_path):
    """A spec the driver should never write still must not fail silently."""
    result_path = tmp_path / "result.json"
    spec = tmp_path / "bad_action.json"
    spec.write_text(
        json.dumps({
            "run_id": "bad-1",
            "task": "binary",
            "action": "summarise",
            "config": {
                "env_type": "osiris",
                "backend": "boosting",
                "engine": "catboost",
                "device": "gpu",
                "target_column": "target",
                "client_id_column": "epk_id",
                "group_column": None,
                "date_column": None,
                "categorical_columns": [],
                "numerical_columns": ["feature"],
                "hidden_state_columns": [],
                "hyperopt": False,
                "output_dir": str(tmp_path / "outputs"),
                "environment": {},
            },
            "payload": {},
            "result_path": str(result_path),
            "log_path": str(tmp_path / "worker.log"),
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported AutoML action: 'summarise'"):
        execute_spec(spec)

    # The failure is on disk, because the driver reads result.json and nothing
    # else: a job that raised and left no record is indistinguishable from one
    # that is still running.
    recorded = json.loads(result_path.read_text(encoding="utf-8"))
    assert recorded["status"] == "failed"
    assert recorded["error_type"] == "ValueError"
    assert "summarise" in recorded["error"]


def test_an_unknown_task_is_refused_before_any_work(tmp_path):
    spec = tmp_path / "bad_task.json"
    spec.write_text(
        json.dumps({
            "run_id": "bad-2",
            "task": "clustering",
            "action": "train",
            "config": {},
            "payload": {},
            "result_path": str(tmp_path / "result.json"),
            "log_path": str(tmp_path / "worker.log"),
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported AutoML task: 'clustering'"):
        execute_spec(spec)
