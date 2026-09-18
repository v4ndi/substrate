import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from fmlib.automl import MulticlassTask, MulticlassTaskConfig, PredictionResult
from fmlib.automl.backends.boosting import MulticlassBoostingBackend
from fmlib.automl.exceptions import ArtifactError, ConfigError, SchemaError
from fmlib.automl.metrics import binary_top_k_metrics, multiclass_class_metrics


def test_multiclass_documentation_is_self_contained_and_base_is_task_neutral():
    repository = Path(__file__).resolve().parents[3]
    documentation_paths = [
        repository / "fmlib/automl/tasks/multiclass.py",
        repository / "fmlib/automl/backends/boosting/multiclass.py",
        repository / "fmlib/automl/config/tasks.py",
        repository / "examples/automl/configs/fmlib_multiclass.yaml",
        repository / "examples/automl/tests/configs/multiclass_one_trial.yaml",
        repository / "examples/automl/tests/configs/multiclass_inline_features.yaml",
    ]
    notebook_paths = [
        repository / "examples/automl/multiclass_pipeline.ipynb",
        repository / "examples/automl/tests/multiclass_config_sources.ipynb",
        repository / "examples/automl/tests/multiclass_engines.ipynb",
        repository / "examples/automl/tests/multiclass_model_scopes.ipynb",
    ]
    documentation = "\n".join(
        path.read_text(encoding="utf-8") for path in documentation_paths
    )
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        documentation += "\n" + "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        if path.parent.name == "tests":
            code_cells = [
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]
            assert code_cells
            assert all(
                cell["execution_count"] is not None and cell["outputs"]
                for cell in code_cells
            )
            assert not any(
                output.get("output_type") == "error"
                for cell in code_cells
                for output in cell["outputs"]
            )
    normalized = documentation.casefold()
    assert "legacy" not in normalized
    assert "autocampaign" not in normalized

    common_source = "\n".join(
        (repository / relative).read_text(encoding="utf-8")
        for relative in (
            "fmlib/automl/tasks/base.py",
            "fmlib/automl/backends/boosting/base.py",
        )
    ).casefold()
    for task_specific in (
        "positive_class",
        "roc_auc",
        "mean_squared_error",
        "beta_calibration",
    ):
        assert task_specific not in common_source


def _write_splits(
    tmp_path, *, labels=(0, 1, 2), calibration=False, group=True, hidden=False
):
    rng = np.random.default_rng(42)
    months = (
        {
            "train": ["2025-09-01", "2025-10-01"],
            "valid": ["2025-11-01", "2025-12-01"],
            "test": ["2026-01-01", "2026-02-01"],
        }
        if calibration
        else {"train": ["2025-11-01"], "valid": ["2025-12-01"], "test": ["2026-01-01"]}
    )
    paths = []
    for split, split_months in months.items():
        frames = []
        for month_index, month in enumerate(split_months):
            size = 90
            encoded = np.tile(np.arange(len(labels)), size // len(labels))
            rng.shuffle(encoded)
            feature = encoded + rng.normal(scale=0.35, size=size)
            values = {
                "epk_id": np.arange(size) + month_index * size,
                "target": np.asarray(labels, dtype=object)[encoded].tolist(),
                "segment": np.where(
                    feature < 0.7, "low", np.where(feature > 1.3, "high", "middle")
                ),
                "balance": feature,
                "treatment": np.arange(size) % 2,
                "report_month": [month] * size,
            }
            if group:
                values["group"] = np.where(
                    np.arange(size) % 2, "channel_a", "channel_b"
                )
            if hidden:
                values["seq_hidden_state"] = pl.Series(
                    "seq_hidden_state",
                    np.column_stack((feature, feature**2)).astype(np.float32).tolist(),
                    dtype=pl.Array(pl.Float32, 2),
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
        "model_params": {"iterations": 12, "depth": 3},
        "verbose": False,
        "output_dir": tmp_path / "reports",
        "environment": {},
    }
    values.update(overrides)
    if values["hyperopt"] and "model_params" not in overrides:
        values["model_params"] = {}
    return MulticlassTaskConfig(**values)


@pytest.mark.parametrize(
    ("engine", "params"),
    [
        ("catboost", {"iterations": 12, "depth": 3}),
        ("xgboost", {"n_estimators": 12, "max_depth": 3}),
    ],
)
def test_multiclass_lifecycle_probability_contract_reports_and_artifact(
    tmp_path, engine, params
):
    pytest.importorskip(engine)
    train, valid, test = _write_splits(
        tmp_path, labels=("class z", "class a", "class m"), hidden=True
    )
    task = MulticlassTask(
        _config(
            tmp_path,
            engine=engine,
            model_params=params,
            hidden_state_columns=["seq_hidden_state"],
        )
    )

    training = task.train(train, valid)
    prediction = task.predict(test)
    open_figures = set(plt.get_fignums())
    evaluation = task.evaluate(test, prediction)
    class_wise = task.evaluate(test, prediction, metrics=["class_wise_roc_auc"])
    aggregate_only = task.evaluate(test, prediction, metrics=["roc_auc_ovr_macro"])
    artifact = task.save()
    restored = MulticlassTask.load(artifact)
    restored_prediction = restored.predict(test)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

    score_columns = ["score_0", "score_1", "score_2"]
    probabilities = prediction.scores.select(score_columns).to_numpy()
    assert training.task_name == "multiclass"
    assert set(plt.get_fignums()) == open_figures
    assert {"group", "seq_hidden_state__0", "seq_hidden_state__1"} <= set(
        training.feature_names
    )
    assert "treatment" not in training.feature_names
    assert prediction.scores.columns == ["epk_id", "report_month", *score_columns]
    assert prediction.class_order == ("class a", "class m", "class z")
    assert not hasattr(prediction, "raw_scores")
    assert "prediction" not in prediction.scores.columns
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    aggregate_metrics = {
        "roc_auc_ovr_macro",
        "log_loss",
        "accuracy",
        "f1_macro",
        "f1_weighted",
    }
    class_metrics = {
        f"{metric}_class_{index}"
        for index in range(3)
        for metric in (
            "roc_auc",
            "n_positives",
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
        )
    }
    assert set(evaluation.metrics_raw) == aggregate_metrics | class_metrics
    assert set(class_wise.metrics_raw) == {
        *(f"roc_auc_class_{index}" for index in range(3)),
        *(f"n_positives_class_{index}" for index in range(3)),
    }
    assert class_wise.metrics_by_class_raw.columns == [
        "class_index",
        "class_label",
        "roc_auc",
        "n_positives",
    ]
    assert set(aggregate_only.metrics_raw) == {"roc_auc_ovr_macro"}
    assert aggregate_only.metrics_by_class_raw is None
    assert (
        sum(evaluation.metrics_raw[f"n_positives_class_{index}"] for index in range(3))
        == prediction.scores.height
    )
    assert all(
        0 <= evaluation.metrics_raw[f"precision@10_class_{index}"] <= 1
        for index in range(3)
    )
    assert set(evaluation.metrics_by_group_raw.columns) == {
        "scope",
        "report_month",
        "group",
        "n_samples",
        *evaluation.metrics_raw,
    }
    assert evaluation.excel_paths_raw["metrics_raw"].exists()
    assert evaluation.excel_paths_raw["metrics_by_class_raw"].exists()
    assert evaluation.excel_paths_calibrated == {}
    assert evaluation.metrics_by_class_raw.columns == [
        "class_index",
        "class_label",
        "roc_auc",
        "n_positives",
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
    ]
    assert evaluation.metrics_by_class_raw["class_label"].to_list() == list(
        prediction.class_order
    )
    assert evaluation.metrics_by_class_raw.schema["class_label"] == pl.String
    assert (
        evaluation.metrics_by_class_raw["n_positives"].sum() == prediction.scores.height
    )
    assert not hasattr(evaluation, "calibrated_metrics_by_class")
    assert "multiclass_metrics_by_date_raw" in evaluation.figures_raw
    assert "multiclass_confusion_matrix_raw" in evaluation.figures_raw
    with ZipFile(evaluation.excel_paths_raw["feature_importance_raw"]) as workbook:
        assert b"<autoFilter" in workbook.read("xl/tables/table1.xml")
    assert (
        artifact == (tmp_path / "reports" / "artifacts" / "multiclass_model").resolve()
    )
    assert restored.class_order == prediction.class_order
    assert restored_prediction.class_order == prediction.class_order
    assert restored_prediction.scores.columns == prediction.scores.columns
    assert np.allclose(restored_prediction.scores.select(score_columns), probabilities)
    assert manifest["task"] == "multiclass"
    assert manifest["class_order"] == list(prediction.class_order)
    assert manifest["target_schema"] == {
        "dtype": "String",
        "class_order": list(prediction.class_order),
    }
    assert "format_version" not in manifest
    assert [item["path"] for item in manifest["source_manifests"]["train"]] == [
        str(train.resolve())
    ]
    with pytest.raises(ArtifactError, match="already exists"):
        task.save(artifact)
    assert task.save(artifact, overwrite=True) == artifact
    with pytest.raises(ArtifactError, match="already exists"):
        task.save(artifact, overwrite=False)


@pytest.mark.parametrize("labels", [(10, 30, 20), ("z", "a", "m"), (2.5, -1.25, 0.0)])
def test_supported_class_order_is_deterministic_and_dtype_is_preserved(
    tmp_path, labels
):
    train, valid, test = _write_splits(tmp_path, labels=labels)
    task = MulticlassTask(_config(tmp_path, model_params={"iterations": 4}))
    task.train(train, valid)
    evaluation = task.evaluate(test, task.predict(test))
    manifest = json.loads((task.save() / "manifest.json").read_text(encoding="utf-8"))

    assert task.class_order == tuple(sorted(labels))
    expected_dtype = (
        "Int64"
        if isinstance(labels[0], int)
        else "Float64"
        if isinstance(labels[0], float)
        else "String"
    )
    assert manifest["target_schema"]["dtype"] == expected_dtype
    assert [item["label"] for item in manifest["target_label_encoding"]] == sorted(
        labels
    )
    assert str(evaluation.metrics_by_class_raw.schema["class_label"]) == expected_dtype


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ([0, 1, 1], "at least three"),
        ([0, 1, None], "null"),
        ([0.0, 1.0, float("nan")], "NaN"),
        ([0.0, 1.0, float("inf")], "infinite"),
        ([[0], [1], [2]], "scalar"),
        ([False, True, False], "string, integer or finite float"),
    ],
)
def test_multiclass_target_validation(tmp_path, target, message):
    task = MulticlassTask(_config(tmp_path))
    frame = pl.DataFrame({"target": target}, strict=False)
    with pytest.raises(SchemaError, match=message):
        task._prepare_training_state(frame, frame)


def test_per_class_top_k_metrics_match_binary_fmlib_contract(tmp_path):
    task = MulticlassTask(_config(tmp_path))
    task._class_order = ("a", "b", "c")
    target = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
        [0.5, 0.3, 0.2],
    ])

    metrics = multiclass_class_metrics(
        target, probabilities, task._class_order, grouped=False
    )

    for index in range(3):
        binary_target = (target == index).astype(np.int8)
        expected = binary_top_k_metrics(binary_target, probabilities[:, index])
        assert metrics[f"roc_auc_class_{index}"] == 1.0
        assert metrics[f"n_positives_class_{index}"] == int(binary_target.sum())
        for name, value in expected.items():
            assert metrics[f"{name}_class_{index}"] == value


def test_unknown_validation_and_evaluation_labels_are_rejected(tmp_path):
    train, valid, test = _write_splits(tmp_path)
    invalid_valid = tmp_path / "invalid_valid.parquet"
    pl.read_parquet(valid).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(99))
        .otherwise(pl.col("target"))
        .alias("target")
    ).write_parquet(invalid_valid)
    with pytest.raises(SchemaError, match="absent from training class_order"):
        MulticlassTask(_config(tmp_path)).train(train, invalid_valid)

    task = MulticlassTask(_config(tmp_path, model_params={"iterations": 4}))
    task.train(train, valid)
    invalid_test = tmp_path / "invalid_test.parquet"
    pl.read_parquet(test).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(99))
        .otherwise(pl.col("target"))
        .alias("target")
    ).write_parquet(invalid_test)
    prediction = task.predict(invalid_test)
    with pytest.raises(SchemaError, match="absent from training class_order"):
        task.evaluate(invalid_test, prediction)


def test_group_model_part_with_incomplete_multiclass_target_has_named_error(tmp_path):
    train, valid, _ = _write_splits(tmp_path)
    train_frame = pl.read_parquet(train).with_columns(
        pl.when(pl.col("target") == 2)
        .then(pl.lit("channel_b"))
        .otherwise(pl.col("group"))
        .alias("group")
    )
    train_frame.write_parquet(train)
    task = MulticlassTask(
        _config(tmp_path, model_layout="per_group", model_params={"iterations": 4})
    )
    with pytest.raises(SchemaError, match="per_group:channel_a.*classes are missing"):
        task.train(train, valid)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("global", {"global"}),
        ("per_group", {"per_group:channel_a", "per_group:channel_b"}),
        (
            "global_and_per_group",
            {"global", "per_group:channel_a", "per_group:channel_b"},
        ),
    ],
)
def test_multiclass_model_layouts_and_feature_semantics(tmp_path, scope, expected):
    train, valid, test = _write_splits(tmp_path)
    task = MulticlassTask(
        _config(tmp_path, model_layout=scope, model_params={"iterations": 4})
    )
    result = task.train(train, valid)
    prediction = task.predict(test)
    evaluation = task.evaluate(test, prediction)

    assert set(result.best_params) == expected
    assert prediction.class_order == (0, 1, 2)
    if scope == "global_and_per_group":
        assert prediction.scores.height == pl.read_parquet(test).height * 2
        assert set(prediction.scores["model_layout"].unique()) == {
            "global",
            "per_group",
        }
        assert {"global_accuracy", "per_group_accuracy"} <= set(evaluation.metrics_raw)
    for item in task._models:
        assert ("group" in item.schema.categorical) is (item.layout == "global")
        assert "treatment" not in item.schema.categorical


def test_group_and_combined_layouts_reject_unknown_groups(tmp_path, monkeypatch):
    train, valid, test = _write_splits(tmp_path)
    unknown = tmp_path / "unknown.parquet"
    pl.read_parquet(test).with_columns(
        pl.lit("new_channel").alias("group")
    ).write_parquet(unknown)
    group = MulticlassTask(
        _config(tmp_path, model_layout="per_group", model_params={"iterations": 4})
    )
    group.train(train, valid)
    with pytest.raises(
        SchemaError, match=r"Unknown group values.*'new_channel'.*90 rows"
    ):
        group.predict(unknown)

    both = MulticlassTask(
        _config(
            tmp_path,
            model_layout="global_and_per_group",
            model_params={"iterations": 4},
        )
    )
    both.train(train, valid)
    for item in both._models:
        values = (
            np.array([0.8, 0.1, 0.1])
            if item.layout == "global"
            else np.array([0.1, 0.8, 0.1])
        )
        monkeypatch.setattr(
            item.backend,
            "predict_score",
            lambda frame, _schema, values=values: np.tile(values, (frame.height, 1)),
        )
    with pytest.raises(
        SchemaError, match=r"Unknown group values.*'new_channel'.*90 rows"
    ):
        both.predict(unknown)
    prediction = both.predict(unknown, model_layout="global")
    assert np.allclose(
        prediction.scores.select("score_0", "score_1", "score_2"), [0.8, 0.1, 0.1]
    )


@pytest.mark.parametrize("scope", ["per_group", "global", "global_and_per_group"])
def test_model_layout_is_forbidden_without_group_column(tmp_path, scope):
    with pytest.raises(
        ConfigError, match=rf"model_layout={scope!r}.*group_column=None"
    ):
        _config(
            tmp_path,
            group_column=None,
            model_layout=scope,
            model_params={"iterations": 4},
        )


def test_without_group_evaluation_contains_month_only_rows(tmp_path):
    train, valid, test = _write_splits(tmp_path, group=False)
    task = MulticlassTask(
        _config(
            tmp_path,
            group_column=None,
            model_layout=None,
            model_params={"iterations": 4},
        )
    )
    task.train(train, valid)
    evaluation = task.evaluate(test, task.predict(test))
    assert evaluation.metrics_by_group_raw.columns[:8] == [
        "scope",
        "report_month",
        "n_samples",
        "roc_auc_ovr_macro",
        "log_loss",
        "accuracy",
        "f1_macro",
        "f1_weighted",
    ]
    assert evaluation.metrics_by_group_raw["scope"].unique().to_list() == ["date"]
    assert {
        f"{metric}_class_{index}"
        for index in range(3)
        for metric in (
            "roc_auc",
            "n_positives",
            "precision@5",
            "recall@5",
            "precision@50",
            "recall@50",
        )
    } <= set(evaluation.metrics_by_group_raw.columns)


def test_grouped_slice_with_missing_class_has_null_auc_and_other_metrics(
    tmp_path, caplog
):
    train, valid, test = _write_splits(tmp_path)
    test_frame = pl.read_parquet(test).with_columns(
        pl.when(pl.col("target") == 2)
        .then(pl.lit("channel_b"))
        .otherwise(pl.col("group"))
        .alias("group")
    )
    test_frame.write_parquet(test)
    task = MulticlassTask(_config(tmp_path, model_params={"iterations": 4}))
    task.train(train, valid)
    with caplog.at_level("WARNING"):
        evaluation = task.evaluate(test, task.predict(test))
    channel_a = evaluation.metrics_by_group_raw.filter(pl.col("group") == "channel_a")
    assert channel_a["roc_auc_ovr_macro"].null_count() == channel_a.height
    assert channel_a["roc_auc_class_2"].null_count() == channel_a.height
    assert channel_a["n_positives_class_2"].sum() == 0
    assert channel_a["accuracy"].is_not_null().all()
    assert "will be null" in caplog.text


def test_prediction_result_class_order_mismatch_and_score_shape_are_rejected(tmp_path):
    train, valid, test = _write_splits(tmp_path)
    task = MulticlassTask(_config(tmp_path, model_params={"iterations": 4}))
    task.train(train, valid)
    prediction = task.predict(test)
    mismatched = PredictionResult(prediction.scores, class_order=(2, 1, 0))
    with pytest.raises(SchemaError, match="class_order mismatch"):
        task.evaluate(test, mismatched)
    missing = PredictionResult(
        prediction.scores.drop("score_2"), class_order=prediction.class_order
    )
    with pytest.raises(SchemaError, match="score_2"):
        task.evaluate(test, missing)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("roc_auc_ovr_macro", "max"),
        ("accuracy", "max"),
        ("f1_macro", "max"),
        ("log_loss", "min"),
    ],
)
def test_multiclass_hyperopt_direction_no_refit_and_best_fitted_trial(
    tmp_path, monkeypatch, metric, expected
):
    pytest.importorskip("optuna")
    train, valid, _ = _write_splits(tmp_path)
    fit_count = 0
    fitted = []
    objective_values = []
    prepared_ids = set()
    original = MulticlassBoostingBackend.fit_prepared

    def counted(backend, prepared):
        nonlocal fit_count
        fit_count += 1
        prepared_ids.add(id(prepared))
        result = original(backend, prepared)
        fitted.append(backend)
        scores = backend.predict_prepared_score(prepared.valid_prediction_features)
        objective_values.append(
            task._optimization_metric(prepared.valid_target, scores)
        )
        return result

    monkeypatch.setattr(MulticlassBoostingBackend, "fit_prepared", counted)
    task = MulticlassTask(
        _config(
            tmp_path,
            optimization_metric=metric,
            hyperopt=True,
            n_trials=2,
            search_space={"iterations": [3, 4], "depth": [2]},
        )
    )
    result = task.train(train, valid)
    values = pl.Series(objective_values)
    expected_value = values.max() if expected == "max" else values.min()
    assert fit_count == 2
    assert len(prepared_ids) == 1
    assert task._models[0].backend in fitted
    assert result.validation_metrics["global"] == expected_value


def test_named_metric_artifact_load_is_inference_safe(tmp_path):
    train, valid, test = _write_splits(tmp_path)

    task = MulticlassTask(
        _config(
            tmp_path, optimization_metric="accuracy", model_params={"iterations": 4}
        )
    )
    task.train(train, valid)
    before = task.predict(test)
    artifact = task.save(tmp_path / "named")
    restored = MulticlassTask.load(artifact)
    after = restored.predict(test)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))

    assert restored.config.optimization_metric == "accuracy"
    assert manifest["optimization_metric"] == "accuracy"
    assert np.allclose(
        before.scores.select("score_0", "score_1", "score_2"),
        after.scores.select("score_0", "score_1", "score_2"),
    )
