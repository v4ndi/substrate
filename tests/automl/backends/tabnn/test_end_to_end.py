"""train -> save -> load -> predict, through the public AutoML API, on a network.

One public API, two backend families: the same calls a boosting task takes,
with ``backend="tabnn"``. Everything the run needs -- encoding, assembly, the
trial, the artifact -- happens underneath.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fmlib.automl import (
    BinaryTask,
    BinaryTaskConfig,
    MulticlassTask,
    MulticlassTaskConfig,
)

SMALL = {
    "max_epochs": 2,
    "batch_size": 64,
    "hidden_size": 16,
    "num_layers": 1,
    "num_heads": 2,
    "num_workers": 0,
    "evaluations_per_epoch": 1,
    "patience": 2,
}


def _write(directory, rows: int, seed: int, classes: int = 2):
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(rows, 4)).astype(np.float32)
    balance = rng.normal(size=rows)
    logit = 1.5 * balance + hidden[:, 0]
    frame = pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "segment": rng.choice(["a", "b", "c"], rows),
        "balance": balance,
        "seq_hidden_state": [row.tolist() for row in hidden],
        "y_binary": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
        "y_mc": np.digitize(logit, [-0.5, 0.8]).astype(np.int8),
    }).with_columns(pl.col("seq_hidden_state").cast(pl.List(pl.Float32)))
    directory.mkdir(parents=True, exist_ok=True)
    half = rows // 2
    frame.head(half).write_parquet(directory / "part-0.parquet")
    frame.tail(rows - half).write_parquet(directory / "part-1.parquet")
    return directory


@pytest.fixture
def data(tmp_path):
    return {
        "train": _write(tmp_path / "train", 600, 1),
        "valid": _write(tmp_path / "valid", 300, 2),
        "test": _write(tmp_path / "test", 300, 3),
    }


def _binary_config(tmp_path, **overrides):
    values = {
        "env_type": "local",
        "backend": "tabnn",
        "engine": "tabular_transformer",
        "device": "cpu",
        "target_column": "y_binary",
        "client_id_column": "epk_id",
        "group_column": None,
        "date_column": "report_month",
        "categorical_columns": ["segment"],
        "numerical_columns": ["balance"],
        "hidden_state_columns": ["seq_hidden_state"],
        "hyperopt": False,
        "model_params": dict(SMALL),
        "verbose": False,
        "output_dir": tmp_path / "outputs",
        "environment": {},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


@pytest.mark.slow
def test_binary_trains_saves_loads_and_predicts(tmp_path, data):
    task = BinaryTask(_binary_config(tmp_path))
    training = task.train(data["train"], data["valid"])

    assert training.backend_name == "tabnn"
    assert training.engine_name == "tabular_transformer"
    assert set(training.validation_metrics) == {"global"}
    assert np.isfinite(training.validation_metrics["global"])

    before = task.predict(data["test"])
    artifact = task.save(tmp_path / "artifact")
    restored = BinaryTask.load(artifact)
    after = restored.predict(data["test"])

    assert before.scores.height == 300
    assert after.scores.equals(before.scores)


@pytest.mark.slow
def test_the_saved_artifact_holds_no_pickle(tmp_path, data):
    task = BinaryTask(_binary_config(tmp_path))
    task.train(data["train"], data["valid"])
    artifact = task.save(tmp_path / "artifact")

    written = sorted(path.name for path in artifact.rglob("*") if path.is_file())
    assert "model.safetensors" in written
    assert not [name for name in written if name.endswith((".pkl", ".pickle", ".bin"))]
    for name in ("preprocessor.yaml", "train_config.yaml"):
        text = next(artifact.rglob(name)).read_text()
        assert "!!python" not in text


@pytest.mark.slow
def test_a_multiclass_model_scores_a_probability_matrix(tmp_path, data):
    config = MulticlassTaskConfig(
        env_type="local",
        backend="tabnn",
        engine="tabular_transformer",
        device="cpu",
        target_column="y_mc",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hidden_state_columns=["seq_hidden_state"],
        hyperopt=False,
        model_params=dict(SMALL),
        verbose=False,
        output_dir=tmp_path / "outputs",
        environment={},
    )
    task = MulticlassTask(config)
    task.train(data["train"], data["valid"])
    prediction = task.predict(data["test"])

    assert len(prediction.class_order) == 3
    scores = prediction.scores.select(pl.exclude("epk_id")).to_numpy()
    assert scores.shape[1] >= 3
    probabilities = scores[:, -3:]
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.slow
def test_the_processed_data_is_encoded_once_and_reused(tmp_path, data, capsys):
    config = _binary_config(tmp_path)
    BinaryTask(config).train(data["train"], data["valid"])
    processed_root = config.resolved_processed_data_path
    keys = [path.name for path in processed_root.iterdir() if path.is_dir()]
    assert len(keys) == 1

    BinaryTask(_binary_config(tmp_path / "second")).train(data["train"], data["valid"])
    # A second task with the same sources and the same preprocessing writes to
    # its own output_dir, so it is a second directory -- but the key is equal,
    # which is what makes a shared processed_data_path reuse anything.
    other_root = tmp_path / "second" / "outputs" / "processed"
    assert [path.name for path in other_root.iterdir() if path.is_dir()] == keys
