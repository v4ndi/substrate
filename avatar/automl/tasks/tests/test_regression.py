import ast
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from avatar.automl import RegressionTask, RegressionTaskConfig
from avatar.automl.backends.boosting import RegressionBoostingBackend, suggest_params
from avatar.automl.exceptions import ArtifactError, ConfigError, SchemaError


def test_regression_api_and_documentation_are_self_contained():
    repository = Path(__file__).resolve().parents[4]
    implementation_paths = [
        repository / "avatar/automl/calibrators/isotonic_regression.py",
        repository / "avatar/automl/tasks/base.py",
        repository / "avatar/automl/tasks/regression.py",
        repository / "avatar/automl/backends/boosting/regression.py",
        repository / "avatar/automl/config/tasks.py",
        repository / "examples/automl/configs/fmlib_regression.yaml",
        repository / "examples/automl/tests/configs/regression_one_trial.yaml",
        repository / "examples/automl/tests/configs/regression_inline_features.yaml",
    ]
    notebook_paths = [
        repository / "examples/automl/regression_pipeline.ipynb",
        repository / "examples/automl/tests/regression_config_sources.ipynb",
        repository / "examples/automl/tests/regression_engines.ipynb",
        repository / "examples/automl/tests/regression_model_scopes.ipynb",
    ]

    documentation = "\n".join(path.read_text(encoding="utf-8") for path in implementation_paths)
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        documentation += "\n" + "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )

    normalized = documentation.casefold()
    assert "autocampaign" not in normalized
    assert "legacy" not in normalized


def test_base_task_docstrings_are_task_neutral():
    automl_root = Path(__file__).resolve().parents[2]
    base_paths = [automl_root / "tasks/base.py", automl_root / "backends/boosting/base.py"]
    docstrings = []
    for base_path in base_paths:
        tree = ast.parse(base_path.read_text(encoding="utf-8"))
        docstrings.extend(
            docstring
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and (docstring := ast.get_docstring(node, clean=False)) is not None
        )

    normalized = "\n".join(docstrings).casefold()
    for task_specific_term in ("binary", "classification", "positive-class", "roc auc", "logloss"):
        assert task_specific_term not in normalized

    task_base_source = base_paths[0].read_text(encoding="utf-8")
    assert "roc_auc" not in task_base_source
    assert "_top_k_metrics" not in task_base_source
    assert "import optuna" not in task_base_source
    assert "default_search_space" not in task_base_source


@pytest.fixture(autouse=True)
def _isolate_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _write_splits(tmp_path, *, calibration=False, hidden=False):
    rng = np.random.default_rng(42)
    split_months = (
        {"train": ["2025-09-01", "2025-10-01"], "valid": ["2025-11-01", "2025-12-01"], "test": ["2026-01-01", "2026-02-01"]}
        if calibration
        else {"train": ["2025-11-01"], "valid": ["2025-12-01"], "test": ["2026-01-01"]}
    )
    paths = []
    for split, months in split_months.items():
        frames = []
        for month_index, month in enumerate(months):
            size = 50
            balance = rng.normal(size=size)
            group = np.where(np.arange(size) % 2, "channel_a", "channel_b")
            treatment = rng.integers(0, 2, size=size)
            target = 3.0 + 1.8 * balance + (group == "channel_a") * 0.7 + treatment * 0.2 + rng.normal(0, 0.2, size)
            values = {
                "epk_id": np.arange(size) + month_index * size,
                "target": target.astype(np.float32),
                "segment": np.where(balance > 0, "a", "b"),
                "balance": balance,
                "group": group,
                "treatment": treatment,
                "report_month": [month] * size,
            }
            if hidden:
                values["seq_hidden_state"] = pl.Series(
                    "seq_hidden_state",
                    np.column_stack((balance, balance**2)).tolist(),
                    dtype=pl.List(pl.Float64),
                )
            frames.append(pl.DataFrame(values))
        path = tmp_path / f"{split}.parquet"
        pl.concat(frames).write_parquet(path)
        paths.append(path)
    return paths


def _config(tmp_path, **overrides):
    values = {
        "env_type": "local",
        "backend": "boosting",
        "engine": "catboost",
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": ["segment"],
        "numerical_columns": ["balance"],
        "hidden_state_columns": (),
        "model_layout": "global",
        "hyperopt": False,
        "model_params": {"iterations": 16, "depth": 3},
        "verbose": False,
        "output_dir": "reports" if tmp_path is None else tmp_path / "reports",
        "environment": {},
    }
    values.update(overrides)
    if values["hyperopt"] and "model_params" not in overrides:
        values["model_params"] = {}
    return RegressionTaskConfig(**values)


@pytest.mark.parametrize(
    ("engine", "params"),
    [("catboost", {"iterations": 16, "depth": 3}), ("xgboost", {"n_estimators": 16, "max_depth": 3})],
)
def test_regression_lifecycle_metrics_reports_and_artifact(tmp_path, engine, params):
    pytest.importorskip(engine)
    train, valid, test = _write_splits(tmp_path, hidden=True)
    task = RegressionTask(_config(tmp_path, engine=engine, model_params=params, hidden_state_columns=["seq_hidden_state"]))

    training = task.train(train, valid)
    prediction = task.predict(test)
    open_figures = set(plt.get_fignums())
    evaluation = task.evaluate(test, prediction)
    artifact = task.save()
    restored = RegressionTask.load(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

    assert training.task_name == "regression"
    assert training.best_params["global"] == params
    assert {"group", "seq_hidden_state__0", "seq_hidden_state__1"} <= set(training.feature_names)
    assert "treatment" not in training.feature_names
    assert prediction.scores.columns == ["epk_id", "report_month", "score"]
    assert not hasattr(prediction, "raw_scores")
    assert "prediction" not in prediction.scores.columns
    assert set(evaluation.metrics_raw) == {"mse", "mae", "mape"}
    assert evaluation.metrics_raw["mse"] < 2.0
    assert set(evaluation.metrics_by_group_raw.columns) == {
        "scope",
        "report_month",
        "group",
        "n_samples",
        "mse",
        "mae",
        "mape",
    }
    assert set(evaluation.metrics_by_group_raw["scope"]) == {"group", "date_group"}
    assert evaluation.metrics_by_group_raw.schema["report_month"] == pl.String
    assert evaluation.excel_paths_raw["metrics_raw"].exists()
    assert evaluation.excel_paths_calibrated == {}
    assert not (tmp_path / "reports" / "metrics_calibrated.xlsx").exists()
    assert "regression_metrics_by_date_raw" in evaluation.figures_raw
    assert set(plt.get_fignums()) == open_figures
    _, dataset_key = task._store.dataset(test, create=False)
    assert (task.path / "evaluations" / dataset_key / "regression_metrics_by_date_raw.png").exists()
    with ZipFile(evaluation.excel_paths_raw["feature_importance_raw"]) as workbook:
        assert b"<autoFilter" in workbook.read("xl/tables/table1.xml")
    assert artifact == (tmp_path / "reports" / "artifacts" / "regression_model").resolve()
    assert np.allclose(prediction.scores["score"], restored.predict(test).scores["score"])
    assert manifest["task"] == "regression"
    assert "format_version" not in manifest
    assert manifest["models"][0]["hidden_dimensions"] == {"seq_hidden_state": 2}
    assert [item["path"] for item in manifest["source_manifests"]["train"]] == [str(train.resolve())]
    with pytest.raises(ArtifactError, match="already exists"):
        task.save(artifact)
    assert task.save(artifact, overwrite=True) == artifact
    with pytest.raises(ArtifactError, match="already exists"):
        task.save(artifact, overwrite=False)


@pytest.mark.parametrize(
    ("target", "message"),
    [([1.0, None], "null"), ([1.0, float("inf")], "infinite"), (["low", "high"], "numerical")],
)
def test_regression_target_validation(target, message):
    task = RegressionTask(_config(None))
    with pytest.raises(SchemaError, match=message):
        task._target(pl.DataFrame({"target": target}, strict=False))


def test_regression_target_preserves_numeric_dtype():
    task = RegressionTask(_config(None))
    target = task._target(pl.DataFrame({"target": pl.Series([1.0, 2.0], dtype=pl.Float32)}))
    assert target.dtype == np.float32


@pytest.mark.parametrize(
    ("model_layout", "expected"),
    [
        ("global", {"global"}),
        ("per_group", {"per_group:channel_a", "per_group:channel_b"}),
        ("global_and_per_group", {"global", "per_group:channel_a", "per_group:channel_b"}),
    ],
)
def test_regression_model_layouts_and_global_grouped_metrics(tmp_path, model_layout, expected):
    train, valid, test = _write_splits(tmp_path)
    task = RegressionTask(_config(tmp_path, model_layout=model_layout, model_params={"iterations": 5, "depth": 2}))
    result = task.train(train, valid)
    prediction = task.predict(test)
    evaluation = task.evaluate(test, prediction)
    assert set(result.best_params) == expected
    assert "group" in evaluation.metrics_by_group_raw.columns
    if model_layout == "global_and_per_group":
        assert prediction.scores.height == pl.read_parquet(test).height * 2
        assert set(prediction.scores["model_layout"].unique()) == {"global", "per_group"}
        assert {"global_mse", "per_group_mse"} <= set(evaluation.metrics_raw)
        assert set(evaluation.metrics_by_group_raw["model_layout"].unique()) == {"global", "per_group"}
    for item in task._models:
        assert ("group" in item.schema.categorical) is (item.layout == "global")
        assert "treatment" not in item.schema.categorical


def test_group_and_combined_layouts_reject_unknown_groups(tmp_path, monkeypatch):
    train, valid, test = _write_splits(tmp_path)
    group = RegressionTask(_config(tmp_path, model_layout="per_group", model_params={"iterations": 4}))
    group.train(train, valid)
    unknown = tmp_path / "unknown.parquet"
    pl.read_parquet(test).with_columns(pl.lit("new").alias("group")).write_parquet(unknown)
    with pytest.raises(SchemaError, match=r"Unknown group values.*'new'.*50 rows"):
        group.predict(unknown)

    both = RegressionTask(_config(tmp_path, model_layout="global_and_per_group", model_params={"iterations": 4}))
    both.train(train, valid)
    for item in both._models:
        value = 1.0 if item.layout == "global" else 2.0
        monkeypatch.setattr(item.backend, "predict_score", lambda frame, schema, value=value: np.full(frame.height, value))
    with pytest.raises(SchemaError, match=r"Unknown group values.*'new'.*50 rows"):
        both.predict(unknown)
    scores = both.predict(unknown, model_layout="global")
    assert scores.scores["score"].to_list() == [1.0] * scores.scores.height


@pytest.mark.parametrize("metric", ["mse", "mae"])
def test_mse_and_mae_hyperopt_minimize_without_refit(tmp_path, monkeypatch, metric):
    pytest.importorskip("optuna")
    train, valid, _ = _write_splits(tmp_path)
    fit_count = 0
    prepared_ids = set()
    objective_values = []
    original = RegressionBoostingBackend.fit_prepared

    def counted(backend, prepared):
        nonlocal fit_count
        fit_count += 1
        prepared_ids.add(id(prepared))
        result = original(backend, prepared)
        scores = backend.predict_prepared_score(prepared.valid_prediction_features)
        objective_values.append(task._optimization_metric(prepared.valid_target, scores))
        return result

    monkeypatch.setattr(RegressionBoostingBackend, "fit_prepared", counted)
    task = RegressionTask(
        _config(
            tmp_path,
            optimization_metric=metric,
            hyperopt=True,
            n_trials=2,
            search_space={"iterations": [3, 4], "depth": [2]},
        )
    )
    result = task.train(train, valid)
    assert fit_count == 2
    assert len(prepared_ids) == 1
    assert result.validation_metrics["global"] == min(objective_values)


class _Trial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, **kwargs):
        self.kwargs = kwargs
        return low

    def suggest_int(self, name, low, high, **kwargs):
        self.int_kwargs = kwargs
        return low


def test_regression_search_space_float_step_and_log_validation(tmp_path):
    task = RegressionTask(
        _config(
            tmp_path,
            hyperopt=True,
            search_space={"eta": {"type": "float", "low": 0.1, "high": 0.3, "step": 0.1}},
        )
    )
    trial = _Trial()
    assert suggest_params(
        trial,
        engine=task.config.engine,
        model_params=task.config.model_params,
        search_space=task.config.search_space,
    ) == {"eta": 0.1}
    assert trial.kwargs == {"step": 0.1, "log": False}
    invalid = RegressionTask(
        _config(
            tmp_path,
            hyperopt=True,
            search_space={"eta": {"type": "float", "low": 0.1, "high": 0.3, "step": 0.1, "log": True}},
        )
    )
    with pytest.raises(ConfigError, match="cannot combine"):
        suggest_params(
            _Trial(),
            engine=invalid.config.engine,
            model_params=invalid.config.model_params,
            search_space=invalid.config.search_space,
        )


def test_regression_uses_packaged_default_search_space(tmp_path, monkeypatch):
    import avatar.automl.backends.boosting.hyperopt as boosting_hyperopt

    monkeypatch.setattr(boosting_hyperopt, "default_search_space", lambda engine, **kwargs: {"depth": [5]})
    task = RegressionTask(_config(tmp_path, model_params={"iterations": 7}, search_space=None))

    assert suggest_params(
        _Trial(),
        engine=task.config.engine,
        model_params=task.config.model_params,
        search_space=task.config.search_space,
    ) == {"iterations": 7, "depth": 5}


def test_regression_integer_and_categorical_search_ranges_from_yaml(tmp_path):
    yaml_path = tmp_path / "search.yaml"
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
  search_space:
    depth: {type: int, low: 3, high: 7, step: 2}
    grow_policy: {type: categorical, choices: [SymmetricTree, Depthwise]}
output_dir: outputs
environment: {}
""",
        encoding="utf-8",
    )
    config = RegressionTaskConfig.from_yaml(yaml_path)
    trial = _Trial()

    params = suggest_params(
        trial,
        engine=config.engine,
        model_params=config.model_params,
        search_space=config.search_space,
    )

    assert params == {"depth": 3, "grow_policy": "SymmetricTree"}
    assert trial.int_kwargs == {"step": 2, "log": False}


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"type": "float", "low": 1.0}, "low.*high"),
        ({"type": "int", "low": 3, "high": 1}, "low <= high"),
        ({"type": "categorical", "choices": []}, "non-empty"),
        ({"type": "float", "low": 0.0, "high": 1.0, "log": True}, "low > 0"),
    ],
)
def test_regression_search_ranges_have_diagnostic_validation(definition, message):
    with pytest.raises(ConfigError, match=message):
        suggest_params(
            _Trial(),
            engine="catboost",
            model_params={},
            search_space={"parameter": definition},
        )


def test_regression_named_metric_artifact_round_trip(tmp_path):
    train, valid, test = _write_splits(tmp_path)

    task = RegressionTask(
        _config(tmp_path, optimization_metric="mae", model_params={"iterations": 6, "depth": 2})
    )
    task.train(train, valid)
    before = task.predict(test).scores["score"]
    artifact = task.save(tmp_path / "named_regression")
    restored = RegressionTask.load(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

    assert restored.config.optimization_metric == "mae"
    assert manifest["optimization_metric"] == "mae"
    assert np.allclose(before, restored.predict(test).scores["score"])


def test_regression_custom_aliases_and_recursive_hive_shards(tmp_path):
    train_root = tmp_path / "product=reg" / "split=train"
    valid_root = tmp_path / "product=reg" / "split=valid"
    test_root = tmp_path / "product=reg" / "split=test"
    rng = np.random.default_rng(7)
    for root, month in ((train_root, "2025-10-01"), (valid_root, "2025-11-01"), (test_root, "2025-12-01")):
        month_root = root / f"period={month}"
        month_root.mkdir(parents=True)
        for shard in range(2):
            feature = rng.normal(size=20)
            pl.DataFrame(
                {
                    "client": np.arange(20) + shard * 20,
                    "label": 2.0 * feature + rng.normal(scale=0.1, size=20),
                    "feature": feature,
                    "channel": np.where(np.arange(20) % 2, "a", "b"),
                }
            ).write_parquet(month_root / f"part-{shard}.parquet")
    task = RegressionTask(
        RegressionTaskConfig(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            target_column="label",
            client_id_column="client",
            date_column="period",
            group_column="channel",
            categorical_columns=(),
            numerical_columns=["feature"],
            hidden_state_columns=(),
            model_layout="global",
            hyperopt=False,
            model_params={"iterations": 6, "depth": 2},
            verbose=False,
            output_dir=tmp_path / "alias_reports",
            environment={},
        )
    )

    training = task.train(train_root, valid_root)
    prediction = task.predict(test_root)
    evaluation = task.evaluate(test_root, prediction)

    assert prediction.scores.columns == ["client", "period", "score"]
    assert prediction.scores.height == 40
    assert "channel" in training.feature_names
    assert evaluation.metrics_by_group_raw.columns[:3] == ["scope", "period", "channel"]
    assert set(evaluation.metrics_by_group_raw["scope"]) == {"group", "date_group"}
    assert len(task._source_manifests["train"]) == 2
