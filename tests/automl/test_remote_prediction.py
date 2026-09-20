"""Remote prediction, end to end, against the executing stand-in.

`train` on the cluster had tests. `predict` did not — not one, in either
layout. That path is not a smaller version of the local one: it fans out into a
job per model part, each job scores its own slice in its own process, and the
driver merges the pieces back by a row id that only exists to make the merge
possible. Every one of those steps can go wrong quietly. Scores landing on the
wrong rows still look like scores.

So the assertion is parity against **the same artifact**: the fan-out and the
merge must reassemble exactly what one process would have produced from the
model the remote run actually trained. Row for row, value for value.

Not against a locally trained model. `EnvironmentRunner.submit` forces
`device="gpu"` on the remote config, this machine has a real GPU, and CatBoost
trained on a GPU is a different model -- measured at 0.2 apart in probability
here, which is not rounding. Comparing those two would be comparing two
computations and would say nothing about the merge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from automl.scheduler_stub import LocalScheduler
from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.environment import EnvironmentRunner

GROUPS = ("retail", "corp", "sme")


def _write(path: Path, rows: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=rows)
    other = rng.normal(size=rows)
    logit = 1.4 * feature - 0.5 * other
    pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "group": pl.Series([str(v) for v in rng.choice(list(GROUPS), rows)]),
        "feature": feature,
        "other": other,
        "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
    }).write_parquet(path)
    return path


def _config(tmp_path: Path, *, env_type: str, layout: str, tag: str):
    venv = tmp_path / "venv"
    venv.mkdir(exist_ok=True)
    return BinaryTaskConfig(
        output_dir=tmp_path / f"out_{tag}",
        env_type=env_type,
        device="gpu" if env_type == "osiris" else "cpu",
        backend="boosting",
        engine="catboost",
        target_column="target",
        client_id_column="epk_id",
        group_column="group",
        date_column="report_month",
        categorical_columns=(),
        numerical_columns=["feature", "other"],
        hidden_state_columns=(),
        model_layout=layout,
        hyperopt=False,
        verbose=False,
        random_state=42,
        model_params={"iterations": 15, "depth": 3, "thread_count": 1},
        environment={"venv_path": str(venv), "pool": "common"},
    )


@pytest.mark.parametrize("layout", ["global", "per_group", "global_and_per_group"])
def test_the_remote_merge_reassembles_what_one_process_would_produce(tmp_path, layout):
    """The fan-out and the merge, checked against the artifact they used."""
    train = _write(tmp_path / "train.parquet", 600, 1)
    test = _write(tmp_path / "test.parquet", 300, 3)

    scheduler = LocalScheduler()
    remote = BinaryTask(
        _config(tmp_path, env_type="osiris", layout=layout, tag=f"remote_{layout}")
    )
    remote._environment_runner = EnvironmentRunner(scheduler)
    remote.train(train, train)
    remote.status(wait=True)
    artifact = remote.save(tmp_path / f"artifact_{layout}")

    assert remote.predict(test) is None, "a remote operation returns no result inline"
    remote.status(wait=True)
    actual = remote.load_prediction(test)

    # One process, the same weights, the same device: the only difference left
    # is the fan-out and the merge, which is what this is about.
    expected = BinaryTask.load(artifact).predict(test, env_type="local", device="gpu")

    assert actual.class_order == expected.class_order
    assert actual.scores.columns == expected.scores.columns
    assert actual.scores.height == expected.scores.height

    order = ["epk_id"] + (["model_layout"] if "model_layout" in actual.scores else [])
    left = actual.scores.sort(order)
    right = expected.scores.sort(order)
    assert left["epk_id"].to_list() == right["epk_id"].to_list()
    np.testing.assert_allclose(
        left["score"].to_numpy(), right["score"].to_numpy(), rtol=0, atol=0
    )


def test_a_per_group_prediction_is_one_job_per_trained_group(tmp_path):
    """Fan-out is per model part, and the parts are the groups seen in training."""
    train = _write(tmp_path / "train.parquet", 600, 1)
    test = _write(tmp_path / "test.parquet", 300, 3)

    scheduler = LocalScheduler()
    task = BinaryTask(
        _config(tmp_path, env_type="osiris", layout="per_group", tag="fanout")
    )
    task._environment_runner = EnvironmentRunner(scheduler)
    task.train(train, train)
    task.status(wait=True)
    train_jobs = len(scheduler.jobs)

    task.predict(test)
    task.status(wait=True)

    predict_jobs = len(scheduler.jobs) - train_jobs
    assert predict_jobs == len(GROUPS), (
        f"expected one predict job per group, got {predict_jobs}"
    )
    # Every job carried its own spec, so no two scored the same slice.
    specs = {job.spec_path for job in scheduler.jobs[train_jobs:]}
    assert len(specs) == predict_jobs
