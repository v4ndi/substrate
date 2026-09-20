"""What prediction routing refuses, exercised through the public API.

Routing decides which fitted model part scores which row. Every uncovered
branch here is a way of getting that wrong *without an error*: a null group, a
group nobody trained on, a column that is not in the frame. The rows would
either be scored by the wrong model or silently dropped, and the metrics would
look ordinary either way.

These go through `task.predict`, because that is where a user meets them and
the message is the whole product of the failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.exceptions import SchemaError

GROUPS = ("retail", "corp")


def _write(
    path: Path, rows: int, seed: int, groups=GROUPS, nulls: bool = False
) -> Path:
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=rows)
    group = [str(value) for value in rng.choice(list(groups), rows)]
    if nulls:
        group[0] = None
    pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "group": pl.Series(group, dtype=pl.String),
        "feature": feature,
        "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-2 * feature))).astype(
            np.int8
        ),
    }).write_parquet(path)
    return path


def _task(tmp_path: Path, layout: str) -> BinaryTask:
    return BinaryTask(
        BinaryTaskConfig(
            output_dir=tmp_path / f"out_{layout}",
            env_type="local",
            device="cpu",
            backend="boosting",
            engine="catboost",
            target_column="target",
            client_id_column="epk_id",
            group_column="group",
            date_column="report_month",
            categorical_columns=(),
            numerical_columns=["feature"],
            hidden_state_columns=(),
            model_layout=layout,
            hyperopt=False,
            verbose=False,
            random_state=42,
            model_params={"iterations": 10, "depth": 2, "thread_count": 1},
        )
    )


@pytest.fixture
def trained(tmp_path):
    """One task per layout, trained on both groups."""
    train = _write(tmp_path / "train.parquet", 400, 1)
    built = {}
    for layout in ("per_group", "global_and_per_group"):
        task = _task(tmp_path, layout)
        task.train(train, train)
        built[layout] = task
    return built, tmp_path


@pytest.mark.parametrize("layout", ["per_group", "global_and_per_group"])
def test_a_null_group_is_refused_rather_than_routed_somewhere(trained, layout):
    """A null cannot match a trained group, so it would be dropped in silence."""
    tasks, tmp_path = trained
    test = _write(tmp_path / f"null_{layout}.parquet", 100, 3, nulls=True)

    with pytest.raises(SchemaError, match="contains null values"):
        tasks[layout].predict(test)


@pytest.mark.parametrize("layout", ["per_group", "global_and_per_group"])
def test_an_unknown_group_is_refused_and_counted(trained, layout):
    """The message carries the row count: how much data this is about."""
    tasks, tmp_path = trained
    test = _write(
        tmp_path / f"unknown_{layout}.parquet", 100, 4, groups=("retail", "sme")
    )

    with pytest.raises(SchemaError, match=r"Unknown group values.*sme.*\d+ rows"):
        tasks[layout].predict(test)


@pytest.mark.parametrize("layout", ["per_group", "global_and_per_group"])
def test_a_missing_group_column_is_refused(trained, layout):
    """Scoring data without the routing column cannot be routed at all."""
    tasks, tmp_path = trained
    source = _write(tmp_path / f"full_{layout}.parquet", 100, 5)
    stripped = tmp_path / f"stripped_{layout}.parquet"
    pl.read_parquet(source).drop("group").write_parquet(stripped)

    with pytest.raises(SchemaError):
        tasks[layout].predict(stripped)


def test_a_group_seen_in_training_still_scores(trained):
    """The control: the refusals above must not be refusing everything."""
    tasks, tmp_path = trained
    test = _write(tmp_path / "good.parquet", 100, 6)

    prediction = tasks["per_group"].predict(test)

    assert prediction.scores.height == 100
    assert np.isfinite(prediction.scores["score"].to_numpy()).all()


def test_a_global_only_model_ignores_the_group_column_entirely(tmp_path):
    """No routing means no routing failures: an unknown group is just a column."""
    train = _write(tmp_path / "train.parquet", 400, 1)
    task = _task(tmp_path, "global")
    task.train(train, train)

    test = _write(tmp_path / "unknown.parquet", 100, 7, groups=("sme",))
    prediction = task.predict(test)

    assert prediction.scores.height == 100


def test_per_group_scoring_preserves_the_input_row_order(trained):
    """Rows are scored per partition and written back by index, not by order.

    If the write-back ever used position instead of the stored row id, every
    group's scores would land on another group's rows and the metrics would
    still look like metrics.
    """
    tasks, tmp_path = trained
    test = _write(tmp_path / "order.parquet", 200, 8)
    frame = pl.read_parquet(test)

    prediction = tasks["per_group"].predict(test)

    assert prediction.scores["epk_id"].to_list() == frame["epk_id"].to_list()

    # Scoring each group on its own must give those rows the same numbers.
    for group in GROUPS:
        subset = tmp_path / f"only_{group}.parquet"
        frame.filter(pl.col("group") == group).write_parquet(subset)
        alone = tasks["per_group"].predict(subset)
        together = prediction.scores.filter(
            pl.col("epk_id").is_in(alone.scores["epk_id"].implode())
        )
        np.testing.assert_allclose(
            alone.scores.sort("epk_id")["score"].to_numpy(),
            together.sort("epk_id")["score"].to_numpy(),
        )
