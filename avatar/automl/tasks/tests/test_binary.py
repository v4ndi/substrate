import json
from dataclasses import asdict
from zipfile import ZipFile

import numpy as np
import polars as pl
import pytest

from avatar.automl import BinaryTask, BinaryTaskConfig, PredictionResult, ResponseTask, ResponseTaskConfig
from avatar.automl.backends.boosting import BinaryBoostingBackend, suggest_params
from avatar.automl.exceptions import ArtifactError, ArtifactIntegrityError, ConfigError, NotFittedError, SchemaError
from avatar.automl.metrics import DEFAULT_EVALUATION_METRICS, binary_top_k_metrics


@pytest.fixture(autouse=True)
def _isolate_default_output_directory(tmp_path, monkeypatch):
    """Keep default output/log paths inside pytest's temporary directory."""
    monkeypatch.chdir(tmp_path)


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
    return BinaryTaskConfig(**values)


def _response_config(**overrides):
    values = asdict(_config()) | {"treatment_column": "treatment", "inverse_treatment": True}
    values.update(overrides)
    return ResponseTaskConfig(**values)


def _write_splits(tmp_path):
    generator = np.random.default_rng(42)
    paths = []
    for name, size in (("train", 120), ("valid", 60), ("test", 60)):
        feature = generator.normal(size=size)
        category = np.where(feature > 0, "a", "b")
        target = (feature + generator.normal(scale=0.4, size=size) > 0).astype(int)
        frame = pl.DataFrame(
            {
                "epk_id": np.arange(size),
                "target": target,
                "segment": category,
                "balance": feature,
                "group": np.where(feature > 0, "channel_a", "channel_b"),
                "treatment": generator.integers(0, 2, size=size),
                "report_month": ["2026-01-01"] * size,
            }
        )
        path = tmp_path / f"{name}.parquet"
        frame.write_parquet(path)
        paths.append(path)
    return paths


def _write_calibration_splits(tmp_path):
    generator = np.random.default_rng(73)
    paths = []
    split_months = {
        "train": ["2025-09-01", "2025-10-01"],
        "valid": ["2025-11-01", "2025-12-01"],
        "test": ["2026-01-01", "2026-02-01"],
    }
    for split, months in split_months.items():
        frames = []
        for month in months:
            size = 80
            feature = generator.normal(size=size)
            frames.append(
                pl.DataFrame(
                    {
                        "epk_id": np.arange(size) + len(frames) * size,
                        "target": (feature + generator.normal(scale=0.8, size=size) > 0).astype(int),
                        "segment": np.where(feature > 0, "a", "b"),
                        "balance": feature,
                        "group": np.where(np.arange(size) % 2, "channel_a", "channel_b"),
                        "treatment": generator.integers(0, 2, size=size),
                        "report_month": [month] * size,
                    }
                )
            )
        path = tmp_path / f"calibration_{split}.parquet"
        pl.concat(frames).write_parquet(path)
        paths.append(path)
    return paths


def test_binary_treatment_inversion_is_configurable():
    frame = pl.DataFrame({"treatment": [0, 1], "feature": [1.0, 2.0]})
    inverted = ResponseTask(
        _response_config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            inverse_treatment=True,
        )
    )._normalize_model_frame(frame)
    unchanged = ResponseTask(
        _response_config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            inverse_treatment=False,
        )
    )._normalize_model_frame(frame)

    assert inverted["treatment"].to_list() == [1, 0]
    assert unchanged.equals(frame)


@pytest.mark.parametrize(
    ("engine", "model_params"),
    [
        ("catboost", {"iterations": 20, "depth": 3}),
        ("xgboost", {"n_estimators": 20, "max_depth": 3}),
    ],
)
def test_binary_train_predict_evaluate_and_artifact_round_trip(tmp_path, engine, model_params):
    pytest.importorskip(engine)
    train_path, valid_path, test_path = _write_splits(tmp_path)
    config = _config(
        env_type="local",
        backend="boosting",
        engine=engine,
        device="cpu",
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        model_params=model_params,
        output_dir=tmp_path / "reports",
    )
    task = BinaryTask(config)

    training = task.train(train_path, valid_path)
    prediction = task.predict(test_path)
    evaluation = task.evaluate(test_path, prediction)
    selected_evaluation = task.evaluate(test_path, prediction, metrics=["precision@10"])
    second_evaluation = task.evaluate(test_path, prediction)
    default_artifact = task.save()
    artifact = task.save(tmp_path / "artifact")
    with pytest.raises(ArtifactError, match="already exists"):
        task.save(tmp_path / "artifact")
    assert task.save(tmp_path / "artifact", overwrite=True) == artifact
    sentinel = tmp_path / "reports" / "keep.txt"
    sentinel.write_text("preserve outputs", encoding="utf-8")
    with pytest.raises(ArtifactError, match="already exists"):
        task.save(tmp_path / "reports")
    assert sentinel.read_text(encoding="utf-8") == "preserve outputs"
    restored_prediction = BinaryTask.load(artifact).predict(test_path)
    loaded_entity = BinaryTask.load(task.path)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

    assert training.validation_metrics["global"] > 0.5
    assert training.best_params["global"] == model_params
    assert not hasattr(training, "trials")
    assert "group" in training.feature_names
    assert "treatment" not in training.feature_names
    assert prediction.scores.columns == ["epk_id", "report_month", "score"]
    assert prediction.scores["score"].is_between(0, 1).all()
    assert evaluation.metrics_raw["roc_auc"] > 0.5
    assert set(selected_evaluation.metrics_raw) == {"precision@10"}
    assert set(selected_evaluation.metrics_by_group_raw.columns) == {
        "scope",
        "report_month",
        "group",
        "n_samples",
        "precision@10",
    }
    assert task._store.latest("evaluate", test_path)["runtime"]["metrics"] == list(
        DEFAULT_EVALUATION_METRICS["binary"]
    )
    assert set(evaluation.metrics_raw) == {
        "roc_auc",
        "precision@5",
        "recall@5",
        "precision@10",
        "recall@10",
        "precision@20",
        "recall@20",
        "precision@25",
        "recall@25",
        "precision@50",
        "recall@50",
    }
    assert evaluation.metrics_by_group_raw is not None
    assert set(evaluation.metrics_by_group_raw.columns) == {
        "scope",
        "report_month",
        "group",
        "n_samples",
        *evaluation.metrics_raw,
    }
    assert evaluation.metrics_by_group_raw.schema["report_month"] == pl.String
    date_group = evaluation.metrics_by_group_raw.filter(pl.col("scope") == "date_group")
    assert date_group["report_month"].to_list() == ["2026-01-01", "2026-01-01"]
    assert set(evaluation.metrics_by_group_raw["scope"]) == {"group", "date_group"}
    assert "roc_auc_by_date_raw" in evaluation.figures_raw
    assert evaluation.excel_paths_raw["metrics_raw"].exists()
    _, dataset_key = task._store.dataset(test_path, create=False)
    evaluation_dir = task.path / "evaluations" / dataset_key
    assert evaluation.excel_paths_raw["metrics_raw"].parent == evaluation_dir
    assert second_evaluation.excel_paths_raw["metrics_raw"].parent == evaluation_dir
    assert task.load_evaluation(test_path).metrics_raw == second_evaluation.metrics_raw
    assert task.config.output_dir == tmp_path / "reports"
    assert default_artifact == (tmp_path / "reports" / "artifacts" / "binary_model").resolve()
    importance = task._models[0].backend.feature_importance(task._models[0].schema)
    assert importance is not None
    assert importance["importance"].to_list() == sorted(importance["importance"].to_list(), reverse=True)
    with ZipFile(evaluation.excel_paths_raw["feature_importance_raw"]) as workbook:
        table_xml = workbook.read("xl/tables/table1.xml")
    assert b"<autoFilter" in table_xml
    assert np.allclose(prediction.scores["score"], restored_prediction.scores["score"])
    assert loaded_entity.id == task.id
    assert not hasattr(loaded_entity.config, "inverse_treatment")
    assert "format_version" not in manifest
    assert "trials" not in manifest
    assert [item["path"] for item in manifest["source_manifests"]["train"]] == [str(train_path.resolve())]
    assert [item["path"] for item in manifest["source_manifests"]["valid"]] == [str(valid_path.resolve())]
    assert manifest["source_manifests"]["train"][0]["size"] == train_path.stat().st_size
    assert manifest["source_manifests"]["train"][0]["modified_ns"] == train_path.stat().st_mtime_ns

    operation_count = len(task._store.operations(action="evaluate"))
    invalid_selections = (
        ([], "metrics"),
        (["roc_auc", "roc_auc"], "metrics"),
        ("roc_auc", "metrics"),
        (["not_registered"], "Unknown metric"),
        (["mse"], "incompatible"),
    )
    for invalid, message in invalid_selections:
        with pytest.raises(ConfigError, match=message):
            task.evaluate(test_path, prediction, metrics=invalid)
    assert len(task._store.operations(action="evaluate")) == operation_count


def test_evaluate_warns_and_aligns_repeated_client_month_keys(tmp_path, caplog):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    test = pl.read_parquet(test_path).with_columns((pl.col("epk_id") // 2).alias("epk_id"))
    test.write_parquet(test_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2, "depth": 2},
            output_dir=tmp_path / "duplicate_key_reports",
            verbose=False,
        )
    )
    task.train(train_path, valid_path)
    prediction = task.predict(test_path)

    result = task.evaluate(test_path, prediction)
    stored_result = task.evaluate(test_path, PredictionResult(prediction.scores))

    assert result.metrics_raw == stored_result.metrics_raw
    assert caplog.text.count("Evaluation contains duplicate rows by ['epk_id', 'report_month']") == 2


def test_unfitted_task_rejects_predict_evaluate_and_save(tmp_path):
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            numerical_columns=["balance"],
        )
    )

    with pytest.raises(NotFittedError):
        task.predict(tmp_path / "test.parquet")
    with pytest.raises(NotFittedError):
        task.evaluate(tmp_path / "test.parquet", tmp_path / "scores.parquet")
    with pytest.raises(NotFittedError):
        task.save(tmp_path / "artifact")


def test_failed_artifact_save_rolls_back_fitted_state_and_training_result(tmp_path, monkeypatch):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2, "depth": 2},
            output_dir=tmp_path / "failed-save",
        )
    )

    def fail_save(*_args, **_kwargs):
        msg = "native model save failed"
        raise TypeError(msg)

    monkeypatch.setattr(task, "save", fail_save)

    with pytest.raises(TypeError, match="native model save failed"):
        task.train(train_path, valid_path)

    assert task.is_fitted is False
    assert task.status()["state"].to_list() == ["failed"]
    with pytest.raises(ArtifactIntegrityError, match="Training result is not available"):
        task.training_result()
    with pytest.raises(NotFittedError):
        task.predict(test_path)


def test_predict_preserves_input_row_order_and_ignores_extra_columns(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 8, "depth": 2},
        )
    )
    task.train(train_path, valid_path)
    original = pl.read_parquet(test_path)
    reordered_path = tmp_path / "reordered.parquet"
    reordered = original.with_row_index("source_order").sort("balance", descending=True).with_columns(pl.lit(1).alias("extra"))
    reordered.select("extra", "balance", "treatment", "epk_id", "report_month", "group", "segment", "target").write_parquet(
        reordered_path
    )

    original_scores = task.predict(test_path).scores.sort("epk_id")
    reordered_scores = task.predict(reordered_path).scores

    assert reordered_scores["epk_id"].to_list() == reordered["epk_id"].to_list()
    assert np.allclose(
        original_scores["score"].to_numpy(),
        reordered_scores.sort("epk_id")["score"].to_numpy(),
    )


def test_predict_rejects_missing_fitted_feature(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 6, "depth": 2},
        )
    )
    task.train(train_path, valid_path)
    missing_feature_path = tmp_path / "missing_feature.parquet"
    pl.read_parquet(test_path).drop("balance").write_parquet(missing_feature_path)

    with pytest.raises(SchemaError, match="numerical_columns='balance'"):
        task.predict(missing_feature_path)


def test_evaluate_requires_target_column_in_test_data(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 8, "depth": 2},
            output_dir=tmp_path / "reports",
        )
    )
    task.train(train_path, valid_path)
    test = pl.read_parquet(test_path)
    no_target_path = tmp_path / "test_without_target.parquet"
    test.drop("target").write_parquet(no_target_path)

    with pytest.raises(SchemaError, match="target_column='target'"):
        task.evaluate(no_target_path, task.predict(no_target_path))


def test_binary_training_allows_disabled_group(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    paths = []
    for source in (train_path, valid_path, test_path):
        path = tmp_path / f"without_roles_{source.name}"
        pl.read_parquet(source).drop("group", "treatment").write_parquet(path)
        paths.append(path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            group_column=None,
            model_layout=None,
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 8, "depth": 2},
            output_dir=tmp_path / "reports_without_roles",
        )
    )

    training = task.train(paths[0], paths[1])
    evaluation = task.evaluate(paths[2], task.predict(paths[2]))

    assert training.feature_names == ("segment", "balance")
    assert evaluation.metrics_by_group_raw.columns == ["scope", "report_month", "n_samples", *evaluation.metrics_raw]
    assert evaluation.metrics_by_group_raw["scope"].unique().to_list() == ["date"]


@pytest.mark.parametrize("group_column", [None, "group"])
def test_binary_lifecycle_without_date_has_no_synthetic_date_and_expected_slices(tmp_path, group_column):
    source_paths = _write_splits(tmp_path)
    paths = []
    for index, source in enumerate(source_paths):
        path = tmp_path / f"without_date_{index}.parquet"
        frame = pl.read_parquet(source).drop("report_month")
        if group_column is None:
            frame = frame.drop("group")
        frame.write_parquet(path)
        paths.append(path)
    task = BinaryTask(
        _config(
            date_column=None,
            group_column=group_column,
            model_layout=None if group_column is None else "global",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 4, "depth": 2},
            output_dir=tmp_path / f"without_date_{group_column}",
        )
    )

    task.train(paths[0], paths[1])
    prediction = task.predict(paths[2])
    evaluation = task.evaluate(paths[2], prediction)

    assert prediction.scores.columns == ["epk_id", "score"]
    if group_column is None:
        assert evaluation.metrics_by_group_raw is None
        assert "binary_metrics_by_group_raw" not in evaluation.figures_raw
    else:
        assert evaluation.metrics_by_group_raw["scope"].unique().to_list() == ["group"]
        assert "binary_metrics_by_group_raw" in evaluation.figures_raw


@pytest.mark.parametrize("role_column", ["group"])
def test_binary_training_requires_each_configured_role_column(tmp_path, role_column):
    train_path, valid_path, _ = _write_splits(tmp_path)
    missing_path = tmp_path / f"train_without_{role_column}.parquet"
    pl.read_parquet(train_path).drop(role_column).write_parquet(missing_path)
    task = BinaryTask(
        _config(
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2, "depth": 2},
            output_dir=tmp_path / f"missing_{role_column}",
        )
    )

    with pytest.raises(SchemaError, match=rf"{role_column}_column='{role_column}'"):
        task.train(missing_path, valid_path)


def test_missing_custom_treatment_column_uses_public_name_in_error(tmp_path):
    train_path, valid_path, _ = _write_splits(tmp_path)
    missing_path = tmp_path / "train_without_custom_treatment.parquet"
    pl.read_parquet(train_path).drop("treatment").write_parquet(missing_path)
    task = ResponseTask(
        _response_config(
            treatment_column="target_attr_3",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2, "depth": 2},
            output_dir=tmp_path / "missing_custom_treatment",
        )
    )

    with pytest.raises(SchemaError, match="treatment_column='target_attr_3'"):
        task.train(missing_path, valid_path)


@pytest.mark.parametrize(
    ("model_layout", "expected_names"),
    [
        ("global", {"global"}),
        ("per_group", {"per_group:channel_a", "per_group:channel_b"}),
        ("global_and_per_group", {"global", "per_group:channel_a", "per_group:channel_b"}),
    ],
)
def test_model_layout_trains_the_requested_global_and_group_models(tmp_path, model_layout, expected_names):
    train_path, valid_path, _ = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            model_layout=model_layout,
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 4, "depth": 2},
            verbose=False,
            output_dir=tmp_path / f"reports_{model_layout}",
        )
    )

    result = task.train(train_path, valid_path)
    artifact = task.save(tmp_path / f"artifact_{model_layout}")
    restored = BinaryTask.load(artifact)
    model_names = {task._model_name(item.layout, item.group_value) for item in task._models}
    restored_names = {restored._model_name(item.layout, item.group_value) for item in restored._models}

    assert model_names == expected_names
    assert restored_names == expected_names
    assert set(result.best_params) == expected_names
    for item in task._models:
        assert "treatment" not in item.schema.categorical
        assert ("group" in item.schema.categorical) is (item.layout == "global")


def test_combined_layout_rejects_unknown_group_before_scoring(tmp_path, monkeypatch):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            model_layout="global_and_per_group",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 4, "depth": 2},
            verbose=False,
        )
    )
    task.train(train_path, valid_path)
    constants = {"global": 0.1, "per_group:channel_a": 0.2, "per_group:channel_b": 0.3}
    for item in task._models:
        value = constants[task._model_name(item.layout, item.group_value)]
        monkeypatch.setattr(
            item.backend,
            "predict_score",
            lambda frame, _schema, value=value: np.full(frame.height, value),
        )
    test = pl.read_parquet(test_path).with_columns(
        pl.when(pl.col("epk_id") == 0).then(pl.lit("new_channel")).otherwise(pl.col("group")).alias("group")
    )
    routed_path = tmp_path / "routed.parquet"
    test.write_parquet(routed_path)

    with pytest.raises(SchemaError, match=r"Unknown group values.*'new_channel'.*1 rows"):
        task.predict(routed_path)

    global_prediction = task.predict(routed_path, model_layout="global")
    assert global_prediction.scores.height == test.height
    assert global_prediction.scores["score"].to_list() == [0.1] * test.height


def test_collapsed_combined_layout_rejects_unknown_group_after_artifact_load(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    for path in (train_path, valid_path, test_path):
        pl.read_parquet(path).with_columns(pl.lit("only").alias("group")).write_parquet(path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            model_layout="global_and_per_group",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 4, "depth": 2},
            verbose=False,
        )
    )
    task.train(train_path, valid_path)
    artifact = task.save(tmp_path / "collapsed")
    restored = BinaryTask.load(artifact)
    unknown_path = tmp_path / "unknown-collapsed.parquet"
    pl.read_parquet(test_path).with_columns(pl.lit("unknown").alias("group")).write_parquet(unknown_path)

    assert task._models[0].single_group_value == "only"
    assert restored._models[0].single_group_value == "only"
    for fitted in (task, restored):
        with pytest.raises(SchemaError, match=r"Unknown group values.*'unknown'.*60 rows"):
            fitted.predict(unknown_path)
        assert fitted.predict(unknown_path, model_layout="global").scores.height == 60


def test_both_artifact_supports_runtime_scope_overrides(tmp_path, monkeypatch):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            model_layout="global_and_per_group",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 3, "depth": 2},
            verbose=False,
        )
    )
    task.train(train_path, valid_path)
    constants = {"global": 0.1, "per_group:channel_a": 0.2, "per_group:channel_b": 0.3}
    observed_devices = []
    for item in task._models:
        value = constants[task._model_name(item.layout, item.group_value)]

        def score(frame, _schema, *, backend=item.backend, value=value):
            observed_devices.append(backend.device)
            return np.full(frame.height, value)

        monkeypatch.setattr(item.backend, "predict_score", score)

    product = task.predict(test_path, model_layout="global")
    group = task.predict(test_path, model_layout="per_group")
    both = task.predict(test_path, model_layout="global_and_per_group")

    assert product.scores["score"].to_list() == [0.1] * product.scores.height
    assert both.scores.filter(pl.col("model_layout") == "global")["score"].to_list() == product.scores["score"].to_list()
    assert both.scores.filter(pl.col("model_layout") == "per_group")["score"].to_list() == group.scores["score"].to_list()
    assert set(observed_devices) == {"cpu"}
    assert task.config.model_layout == "global_and_per_group"
    assert task.config.device == "cpu"
    assert all(item.backend.device == "cpu" for item in task._models)


def test_runtime_scope_rejects_models_missing_from_artifact(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            model_layout="global",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2},
            verbose=False,
        )
    )
    task.train(train_path, valid_path)

    with pytest.raises(ConfigError, match="requires unavailable model branches.*per_group"):
        task.predict(test_path, model_layout="per_group")


def test_runtime_environment_is_resolved_independently_from_training_config():
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            environment={"venv_path": "/shared/fmlib/env"},
        )
    )

    remote = task._resolve_runtime_config(
        env_type="osiris",
        environment={"venv_path": "/shared/fmlib/predict-env"},
    )

    assert (remote.env_type, remote.device, remote.resolved_device) == ("osiris", "gpu", "gpu")
    assert str(remote.environment.venv_path) == "/shared/fmlib/predict-env"
    assert (task.config.env_type, task.config.device) == ("local", "cpu")
    assert str(task.config.environment.venv_path) == "/shared/fmlib/env"


def test_remote_runtime_defaults_to_gpu_when_switched_to_local():
    task = BinaryTask(
        _config(
            env_type="osiris",
            backend="boosting",
            engine="catboost",
            device="gpu",
            environment={"venv_path": "/shared/fmlib/env"},
        )
    )

    local = task._resolve_runtime_config(env_type="local")

    assert (local.env_type, local.device, local.resolved_device) == ("local", "gpu", "gpu")
    assert (task.config.env_type, task.config.device) == ("osiris", "gpu")


def test_predict_rejects_device_for_remote_environment(monkeypatch):
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
        )
    )

    monkeypatch.setattr(task, "_require_fitted", lambda: None)
    with pytest.raises(ConfigError, match="device can be passed only"):
        task.predict("test.parquet", env_type="osiris", device="gpu")


def test_gpu_catboost_logs_cpu_prediction_policy(monkeypatch, caplog):
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="gpu",
        )
    )
    assert task._prediction_device == "cpu"

    def stop_after_policy_log(*_args, **_kwargs):
        msg = "stop after policy log"
        raise RuntimeError(msg)

    monkeypatch.setattr("avatar.automl.tasks.training.ParquetSource.resolve", stop_after_policy_log)
    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="stop after policy log"):
        task._execute_train("train.parquet", "valid.parquet")

    assert "training uses GPU; prediction always uses CPU" in caplog.text
    assert "GPU evaluation is not implemented for models with categorical features" in caplog.text


def test_per_group_layout_rejects_unknown_group_at_prediction(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            model_layout="per_group",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 4, "depth": 2},
            verbose=False,
        )
    )
    task.train(train_path, valid_path)
    unknown_path = tmp_path / "unknown.parquet"
    pl.read_parquet(test_path).with_columns(pl.lit("unknown").alias("group")).write_parquet(unknown_path)

    with pytest.raises(SchemaError, match=r"Unknown group values.*'unknown'.*60 rows"):
        task.predict(unknown_path)


def test_group_scope_requires_non_null_group_in_train_and_validation(tmp_path):
    train_path, valid_path, _ = _write_splits(tmp_path)
    no_group_path = tmp_path / "no_group.parquet"
    pl.read_parquet(train_path).drop("group").write_parquet(no_group_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            model_layout="per_group",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2},
        )
    )

    with pytest.raises(SchemaError, match="group_column='group'.*model_layout='per_group'"):
        task.train(no_group_path, valid_path)


@pytest.mark.parametrize("value", [None, 2, "treated"])
def test_treatment_must_be_non_null_binary_when_present(value):
    task = ResponseTask(
        _response_config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
        )
    )
    treatment = pl.Series("treatment", [0, value], strict=False)

    with pytest.raises(SchemaError, match="Treatment column"):
        task._normalize_model_frame(pl.DataFrame({"treatment": treatment}))


def test_predict_rejects_feature_dtype_drift(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 4, "depth": 2},
            verbose=False,
        )
    )
    task.train(train_path, valid_path)
    drift_path = tmp_path / "drift.parquet"
    pl.read_parquet(test_path).with_columns(pl.col("balance").cast(pl.Float32)).write_parquet(drift_path)

    with pytest.raises(SchemaError, match="balance: expected Float64, got Float32"):
        task.predict(drift_path)


def test_report_month_is_required_for_every_path_operation(tmp_path):
    train_path, valid_path, _ = _write_splits(tmp_path)
    missing_path = tmp_path / "missing_report_month.parquet"
    pl.read_parquet(train_path).drop("report_month").write_parquet(missing_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2},
        )
    )

    with pytest.raises(SchemaError, match="date_column='report_month'"):
        task.train(missing_path, valid_path)


def test_custom_column_aliases_are_used_for_train_predict_and_evaluate(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    renamed_paths = []
    rename_map = {
        "epk_id": "client",
        "target": "label",
        "group": "channel",
        "treatment": "treatment_flag",
        "report_month": "period",
    }
    for source in (train_path, valid_path, test_path):
        path = tmp_path / f"renamed_{source.name}"
        pl.read_parquet(source).rename(rename_map).write_parquet(path)
        renamed_paths.append(path)
    task = ResponseTask(
        _response_config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            target_column="label",
            client_id_column="client",
            group_column="channel",
            treatment_column="treatment_flag",
            date_column="period",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 8, "depth": 2},
            output_dir=tmp_path / "reports_aliases",
        )
    )

    training = task.train(renamed_paths[0], renamed_paths[1])
    prediction = task.predict(renamed_paths[2])
    evaluation = task.evaluate(renamed_paths[2], prediction)

    assert {"channel", "treatment_flag"}.issubset(training.feature_names)
    assert prediction.scores.columns == ["client", "period", "score"]
    assert evaluation.metrics_by_group_raw.columns[0:3] == ["scope", "period", "channel"]
    assert set(evaluation.metrics_by_group_raw["scope"]) == {"group", "date_group"}
    with ZipFile(evaluation.excel_paths_raw["feature_importance_raw"]) as workbook:
        shared_strings = workbook.read("xl/sharedStrings.xml")
    assert b"channel" in shared_strings
    assert b"treatment_flag" in shared_strings


def test_evaluate_rejects_scores_without_required_role_columns(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 4, "depth": 2},
            verbose=False,
            output_dir=tmp_path / "reports_missing_scores_roles",
        )
    )
    task.train(train_path, valid_path)
    invalid_scores = PredictionResult(pl.DataFrame({"epk_id": [1], "score": [0.5]}))

    with pytest.raises(SchemaError, match="report_month"):
        task.evaluate(test_path, invalid_scores)


def test_named_metric_artifact_round_trip_and_overwrite_can_be_disabled(tmp_path):
    train_path, valid_path, test_path = _write_splits(tmp_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            optimization_metric="roc_auc",
            model_params={"iterations": 6, "depth": 2},
        )
    )
    task.train(train_path, valid_path)
    named_artifact = task.save(tmp_path / "named")
    named_restored = BinaryTask.load(named_artifact)

    assert named_restored.config.optimization_metric == "roc_auc"
    assert np.allclose(task.predict(test_path).scores["score"], named_restored.predict(test_path).scores["score"])

    regular_task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 6, "depth": 2},
        )
    )
    regular_task.train(train_path, valid_path)
    artifact = regular_task.save(tmp_path / "artifact")
    with pytest.raises(ArtifactError, match="already exists"):
        regular_task.save(artifact, overwrite=False)


def test_binary_target_must_contain_zero_and_one(tmp_path):
    train_path, valid_path, _ = _write_splits(tmp_path)
    train = pl.read_parquet(train_path).with_columns(
        pl.when(pl.col("target") == 1).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("target")
    )
    train.write_parquet(train_path)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            model_params={"iterations": 2},
            output_dir=tmp_path / "reports_invalid_target",
        )
    )

    with pytest.raises(SchemaError, match="exactly 0 and 1"):
        task.train(train_path, valid_path)


def test_hyperopt_reuses_best_trial_model_without_an_extra_fit(tmp_path, monkeypatch):
    pytest.importorskip("catboost")
    pytest.importorskip("optuna")
    train_path, valid_path, _ = _write_splits(tmp_path)
    fit_count = 0
    original_fit = BinaryBoostingBackend.fit_prepared

    def counted_fit(backend, prepared):
        nonlocal fit_count
        fit_count += 1
        return original_fit(backend, prepared)

    monkeypatch.setattr(BinaryBoostingBackend, "fit_prepared", counted_fit)
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            categorical_columns=["segment"],
            numerical_columns=["balance"],
            hyperopt=True,
            n_trials=2,
            verbose=False,
            optimization_metric="roc_auc",
            search_space={"iterations": {"type": "int", "low": 2, "high": 3}},
            output_dir=tmp_path / "hyperopt_reports",
        )
    )

    result = task.train(train_path, valid_path)
    train_log = next((tmp_path / "hyperopt_reports" / "logs").glob("*.train.log")).read_text(encoding="utf-8")

    assert fit_count == 2
    assert result.validation_metrics["global"] > 0.5
    assert "[train 1/5] completed duration_seconds=" in train_log
    assert "[train 5/5] completed duration_seconds=" in train_log
    assert "[train 4/5][model 1/1] model=global completed duration_seconds=" in train_log
    assert "[boosting prepare 1/1] engine=catboost completed duration_seconds=" in train_log
    assert "[optuna 2/2] engine=catboost search completed duration_seconds=" in train_log


class _RecordingTrial:
    def __init__(self):
        self.float_kwargs = None
        self.int_kwargs = None
        self.categorical_kwargs = []

    def suggest_float(self, name, low, high, **kwargs):
        self.float_kwargs = {"name": name, "low": low, "high": high, **kwargs}
        return 0.3

    def suggest_int(self, name, low, high, **kwargs):
        self.int_kwargs = {"name": name, "low": low, "high": high, **kwargs}
        return low

    def suggest_categorical(self, name, choices):
        self.categorical_kwargs.append({"name": name, "choices": list(choices)})
        return choices[0]


def test_float_search_space_forwards_step():
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            hyperopt=True,
            search_space={"learning_rate": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}},
        )
    )
    trial = _RecordingTrial()

    assert suggest_params(
        trial,
        engine=task.config.engine,
        model_params=task.config.model_params,
        search_space=task.config.search_space,
    ) == {"learning_rate": 0.3}
    assert trial.float_kwargs == {
        "name": "learning_rate",
        "low": 0.1,
        "high": 0.5,
        "step": 0.1,
        "log": False,
    }


def test_hyperopt_rejects_fixed_model_params():
    with pytest.raises(ConfigError, match="model_params cannot be configured when hyperopt=True"):
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            hyperopt=True,
            model_params={"depth": 9, "fixed": "value"},
            search_space={
                "depth": {"type": "int", "low": 3, "high": 7, "step": 2},
                "bootstrap_type": {"type": "categorical", "choices": ["Bayesian", "Bernoulli"]},
            },
        )


def test_float_search_space_rejects_step_with_log_scale():
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            hyperopt=True,
            search_space={"learning_rate": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1, "log": True}},
        )
    )

    with pytest.raises(ConfigError, match="cannot combine log=True with step=0.1"):
        suggest_params(
            _RecordingTrial(),
            engine=task.config.engine,
            model_params=task.config.model_params,
            search_space=task.config.search_space,
        )


def test_integer_search_space_rejects_non_unit_step_with_log_scale():
    task = BinaryTask(
        _config(
            env_type="local",
            backend="boosting",
            engine="catboost",
            device="cpu",
            hyperopt=True,
            search_space={"iterations": {"type": "int", "low": 10, "high": 100, "step": 10, "log": True}},
        )
    )

    with pytest.raises(ConfigError, match="cannot combine log=True with step=10"):
        suggest_params(
            _RecordingTrial(),
            engine=task.config.engine,
            model_params=task.config.model_params,
            search_space=task.config.search_space,
        )


def test_top_k_metrics_use_percent_of_highest_scores():
    target = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    scores = np.arange(20, 0, -1, dtype=float)

    metrics = binary_top_k_metrics(target, scores)

    assert metrics == {
        "precision@5": 1.0,
        "recall@5": 0.25,
        "precision@10": 0.5,
        "recall@10": 0.25,
        "precision@20": 0.5,
        "recall@20": 0.5,
        "precision@25": 0.6,
        "recall@25": 0.75,
        "precision@50": 0.3,
        "recall@50": 0.75,
    }
