"""Operation settings stay isolated during execution, failure and persistence."""

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from avatar.automl import (
    BinaryTask,
    BinaryTaskConfig,
    MulticlassTask,
    MulticlassTaskConfig,
    RegressionTask,
    RegressionTaskConfig,
    ResponseTaskConfig,
    UpliftTask,
    UpliftTaskConfig,
)
from avatar.automl.backends.boosting.binary import BinaryBoostingBackend
from avatar.automl.backends.boosting.uplift import UpliftBoostingBackend
from avatar.automl.execution import ExecutionContext
from avatar.automl.metrics import resolve_evaluation_metrics


def _config(tmp_path, config_class=BinaryTaskConfig, engine="catboost", **updates):
    values = {
        "env_type": "local",
        "backend": "boosting",
        "engine": engine,
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "date_column": "report_month",
        "group_column": "group",
        "categorical_columns": (),
        "numerical_columns": ["x"],
        "hidden_state_columns": (),
        "model_layout": "global_and_per_group",
        "hyperopt": False,
        "output_dir": tmp_path / "output",
        "verbose": False,
        "model_params": {"iterations": 4, "depth": 2}
        if engine == "catboost"
        else {"n_estimators": 4, "max_depth": 2},
    }
    if config_class in {ResponseTaskConfig, UpliftTaskConfig}:
        values.update(treatment_column="treatment", inverse_treatment=False)
    if config_class is UpliftTaskConfig:
        values["estimate_propensity"] = True
    return config_class(**(values | updates))


def test_device_override_isolates_native_state_and_partial_composite_failure():
    class NativeModel:
        def __init__(self, fail=False):
            self.device = "cpu"
            self.fail = fail

        def set_params(self, *, device):
            self.device = device
            if self.fail:
                msg = "native device failed"
                raise RuntimeError(msg)

    component = BinaryBoostingBackend(
        engine="xgboost", params={}, random_state=42, device="cpu", model=NativeModel()
    )
    assert component.for_execution("cpu") is component
    isolated = component.for_execution("gpu")
    assert isolated.model is not component.model
    assert isolated.device == "gpu" and isolated.model.device == "cuda"
    assert component.device == component.model.device == "cpu"
    failing = BinaryBoostingBackend(
        engine="xgboost",
        params={},
        random_state=42,
        device="cpu",
        model=NativeModel(True),
    )
    composite = UpliftBoostingBackend(
        engine="xgboost",
        params={},
        random_state=42,
        device="cpu",
        components={"first": component, "second": failing},
    )
    with pytest.raises(RuntimeError, match="native device failed"):
        composite.for_execution("gpu")
    assert composite.device == component.device == failing.device == "cpu"
    assert component.model.device == failing.model.device == "cpu"


@pytest.mark.parametrize("engine", ["catboost", "xgboost"])
@pytest.mark.parametrize(
    ("task_class", "config_class", "hyperopt"),
    [
        (BinaryTask, BinaryTaskConfig, False),
        (RegressionTask, RegressionTaskConfig, False),
        (MulticlassTask, MulticlassTaskConfig, False),
        (UpliftTask, UpliftTaskConfig, False),
        (UpliftTask, UpliftTaskConfig, True),
    ],
)
def test_synthetic_runtime_isolation_and_artifact_round_trip(
    tmp_path, monkeypatch, engine, task_class, config_class, hyperopt
):
    paths = []
    for split, rows in enumerate((120, 72, 72)):
        index = np.arange(rows)
        x = np.random.default_rng(split).normal(size=rows)
        target = (index // 4) % (3 if task_class is MulticlassTask else 2)
        if task_class is RegressionTask:
            target = 0.7 * x + index / rows
        path = tmp_path / f"split-{split}.parquet"
        pl.DataFrame({
            "epk_id": index + split * 1000,
            "report_month": ["2026-01-01"] * rows,
            "group": np.where(index % 2, "a", "b"),
            "treatment": (index // 2) % 2,
            "target": target,
            "x": x,
        }).write_parquet(path)
        paths.append(path)
    updates = {}
    if hyperopt:
        params = (
            {"iterations": 4, "depth": 2}
            if engine == "catboost"
            else {"n_estimators": 4, "max_depth": 2}
        )
        updates = {
            "hyperopt": True,
            "n_trials": 2,
            "model_params": {},
            "search_space": {
                key: {"type": "int", "low": value, "high": value}
                for (key, value) in params.items()
            },
        }
    config = _config(tmp_path, config_class, engine, **updates)
    task = task_class(config)
    internal = task._internal_config
    training = task_class._execute_train

    def observe_training(execution, *args, **kwargs):
        assert task.config is config and task._internal_config is internal
        assert (
            execution is not task
            and execution.config.environment.log_dir == tmp_path / "runtime-logs"
        )
        return training(execution, *args, **kwargs)

    monkeypatch.setattr(task_class, "_execute_train", observe_training)
    task.train(paths[0], paths[1], environment={"log_dir": tmp_path / "runtime-logs"})
    stored = json.loads((task.path / "artifact" / "config.json").read_text())
    assert stored["environment"]["log_dir"] is None
    assert task.config is config and task._internal_config is internal
    events = []
    score_branch = task_class._score_prediction_branch
    combine = task_class._combine_layout_predictions

    def observe_scoring(branch, *args):
        events.append(("score", branch.config.model_layout))
        assert branch is not task
        return score_branch(branch, *args)

    def observe_assembly(execution, branches):
        events.append(("assemble", execution.config.model_layout))
        return combine(execution, branches)

    with monkeypatch.context() as scoped:
        scoped.setattr(task_class, "_score_prediction_branch", observe_scoring)
        scoped.setattr(task_class, "_combine_layout_predictions", observe_assembly)
        baseline = task.predict(paths[2])
    assert events == [
        ("score", "global"),
        ("score", "per_group"),
        ("assemble", "global_and_per_group"),
    ]
    assert baseline.scores.height == 144
    predict = task_class._execute_predict

    def fail_prediction(execution, *args, **kwargs):
        assert execution is not task and execution.config.model_layout == "global"
        assert (
            task.config is config and task.config.model_layout == "global_and_per_group"
        )
        assert all(
            left.backend is right.backend
            for left, right in zip(task._models, execution._models, strict=True)
        )
        nested = task._execution_view(
            ExecutionContext.from_config(config).derive(model_layout="per_group")
        )
        assert predict(nested, *args, **kwargs).scores.height == 72
        assert (
            execution.config.model_layout == "global"
            and task.config.model_layout == "global_and_per_group"
        )
        msg = "prediction failed inside operation"
        raise RuntimeError(msg)

    with monkeypatch.context() as scoped:
        scoped.setattr(task_class, "_execute_predict", fail_prediction)
        with pytest.raises(RuntimeError, match="prediction failed inside operation"):
            task.predict(paths[2], model_layout="global")
    assert task_class._execute_predict is predict
    group = task.predict(paths[2], model_layout="per_group")
    product = task.predict(paths[2], model_layout="global")
    assert group.scores.height == product.scores.height == 72
    assert task.config is config and task._internal_config is internal

    def reject_output(*args, **kwargs):
        pytest.fail("Evaluation computation must not write workbooks or render figures")

    with monkeypatch.context() as scoped:
        scoped.setattr(pl.DataFrame, "write_excel", reject_output)
        scoped.setattr(plt, "subplots", reject_output)
        data = task._evaluate_data(
            paths[2],
            baseline,
            "raw",
            tuple(
                metric.name
                for metric in resolve_evaluation_metrics(None, task.config.task_name)
            ),
        )
    assert data.result.figures_raw == data.result.excel_paths_raw == {}
    assert data.feature_importance is not None
    if task_class is MulticlassTask:
        assert set(data.confusions) == {
            "multiclass_confusion_matrix_global",
            "multiclass_confusion_matrix_per_group",
        }
        assert all(matrix.matrix.shape == (3, 3) for matrix in data.confusions.values())
    if task_class is UpliftTask:
        assert set(data.curves) == {
            "qini_curves_global",
            "qini_curves_per_group",
            "uplift_curves_global",
            "uplift_curves_per_group",
        }
        assert all(
            set(curve.learners) == {"s", "t", "x"} for curve in data.curves.values()
        )
    evaluation = task.evaluate(paths[2], baseline)
    assert evaluation.metrics_raw == pytest.approx(data.result.metrics_raw)
    assert evaluation.metrics_by_group_raw.equals(data.result.metrics_by_group_raw)
    columns = [name for name in baseline.scores.columns if name.startswith("score")]
    values = baseline.scores.select(columns).to_numpy()
    assert np.isfinite(values).all()
    entity_artifact_config_path = task.path / "artifact" / "config.json"
    entity_artifact_config = json.loads(entity_artifact_config_path.read_text())
    entity_artifact_config.update(
        env_type="osiris",
        device="gpu",
        environment={"venv_path": "/different/runtime/environment"},
    )
    entity_artifact_config_path.write_text(json.dumps(entity_artifact_config))
    for restored in (
        task_class.load(task.path),
        task_class.load(task.save(tmp_path / "snapshot")),
    ):
        actual = restored.predict(paths[2]).scores.select(columns).to_numpy()
        np.testing.assert_allclose(actual, values, atol=2e-7)
    if task_class is UpliftTask:
        production = tmp_path / "production.parquet"
        pl.read_parquet(paths[2]).drop("target", "treatment").write_parquet(production)
        np.testing.assert_allclose(
            task.predict(production).scores.select(columns).to_numpy(),
            values,
            atol=2e-7,
        )
