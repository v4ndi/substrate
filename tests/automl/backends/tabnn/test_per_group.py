"""A model per group, reading its own partition of one shared encoding.

The layout is the point. A batch comes from one file and there is no shuffle
buffer across files, so a per-group model reading its own subdirectory gets
group-homogeneous batches on purpose — that model is only about that group. The
same property is why a global model must not read this layout, and why the two
cannot be produced at once.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.exceptions import ConfigError

pytestmark = pytest.mark.slow

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


def _write(directory, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    balance = rng.normal(size=rows)
    group = rng.choice(["retail", "corp"], rows)
    logit = 1.5 * balance + np.where(group == "corp", 0.7, -0.7)
    frame = pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "channel": group,
        "segment": rng.choice(["a", "b"], rows),
        "balance": balance,
        "y": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
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
        "test": _write(tmp_path / "test", 300, 3),
    }


def _config(tmp_path, layout, **overrides):
    values = {
        "env_type": "local",
        "backend": "tabnn",
        "engine": "tabular_transformer",
        "device": "cpu",
        "target_column": "y",
        "client_id_column": "epk_id",
        "group_column": "channel",
        "date_column": None,
        "categorical_columns": ["segment"],
        "numerical_columns": ["balance"],
        "hidden_state_columns": [],
        "model_layout": layout,
        "hyperopt": False,
        "model_params": dict(SMALL),
        "verbose": False,
        "output_dir": tmp_path / "outputs",
        "environment": {},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


def test_a_model_per_group_reads_its_own_partition(tmp_path, data):
    task = BinaryTask(_config(tmp_path, "per_group"))
    training = task.train(data["train"], data["valid"])

    assert set(training.validation_metrics) == {"per_group:corp", "per_group:retail"}
    processed = next(
        path
        for path in task.config.resolved_processed_data_path.iterdir()
        if path.is_dir()
    )
    partitions = sorted(path.name for path in (processed / "train").iterdir())
    assert partitions == ["channel=corp", "channel=retail"]
    assert all(
        list(path.glob("*.parquet")) for path in (processed / "train").iterdir()
    )


def test_every_row_still_reaches_exactly_one_partition(tmp_path, data):
    task = BinaryTask(_config(tmp_path, "per_group"))
    task.train(data["train"], data["valid"])
    processed = next(
        path
        for path in task.config.resolved_processed_data_path.iterdir()
        if path.is_dir()
    )
    encoded = pl.read_parquet(processed / "train")
    assert encoded.height == 600
    assert sorted(encoded["epk_id"].to_list()) == sorted(
        pl.read_parquet(data["train"])["epk_id"].to_list()
    )


def test_prediction_routes_each_row_to_its_group_model(tmp_path, data):
    task = BinaryTask(_config(tmp_path, "per_group"))
    task.train(data["train"], data["valid"])
    prediction = task.predict(data["test"])

    assert prediction.scores.height == 300
    assert prediction.scores["score"].is_between(0.0, 1.0).all()
    restored = BinaryTask.load(task.save(tmp_path / "artifact"))
    assert restored.predict(data["test"]).scores.equals(prediction.scores)


def test_the_two_layouts_produce_different_encodings(tmp_path, data):
    """Different physical layout, different key: neither reuses the other."""
    flat = BinaryTask(_config(tmp_path / "flat", "global"))
    flat.train(data["train"], data["valid"])
    partitioned = BinaryTask(_config(tmp_path / "grouped", "per_group"))
    partitioned.train(data["train"], data["valid"])

    flat_keys = {
        path.name
        for path in flat.config.resolved_processed_data_path.iterdir()
        if path.is_dir()
    }
    grouped_keys = {
        path.name
        for path in partitioned.config.resolved_processed_data_path.iterdir()
        if path.is_dir()
    }
    assert flat_keys.isdisjoint(grouped_keys)


def test_both_layouts_at_once_is_still_refused(tmp_path):
    with pytest.raises(ConfigError, match="separate tasks"):
        _config(tmp_path, "global_and_per_group")
