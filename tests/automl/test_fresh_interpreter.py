"""T2: what a second process gets when the first one is gone.

Every end-to-end test in this repo so far trains, saves, loads and predicts
inside one interpreter. That proves the objects agree with each other; it does
not prove the *files* say enough. Import order, a path that only resolved
because the working directory happened to be right, a value that survived in
memory but not in JSON, package data missing from the artifact -- none of it
shows up until something else reads what was written.

So each test here writes in this process and reads in a subprocess started from
scratch, and compares the numbers. The child is given a path and nothing else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fmlib.automl import (
    BinaryTask,
    BinaryTaskConfig,
    ResponseTask,
    ResponseTaskConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

TABNN_PARAMS = {
    "max_epochs": 1,
    "batch_size": 128,
    "hidden_size": 16,
    "num_layers": 1,
    "num_heads": 2,
    "num_workers": 0,
    "evaluations_per_epoch": 1,
    "patience": 1,
}


def _write(path: Path, rows: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=rows)
    other = rng.normal(size=rows)
    logit = 1.5 * feature - 0.6 * other
    pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "segment": rng.choice(["a", "b", "c"], rows),
        "feature": feature,
        "other": other,
        "treatment": rng.integers(0, 2, rows).astype(np.int8),
        "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
    }).write_parquet(path)
    return path


def _run_child(script: str, *arguments: str) -> dict:
    """Run a snippet in a brand-new interpreter and return the JSON it printed."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        environment.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    # A working directory the artifact has no reason to know about: anything
    # the child resolves relative to "here" resolves wrongly, and says so.
    completed = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=os.sep,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, (
        f"child exited {completed.returncode}\n{completed.stderr[-4000:]}"
    )
    payload = completed.stdout.strip().splitlines()[-1]
    return json.loads(payload)


SCORE_FROM_ARTIFACT = """
import json, sys
from fmlib.automl import BinaryTask

artifact, test_path = sys.argv[1], sys.argv[2]
task = BinaryTask.load(artifact)
prediction = task.predict(test_path)
evaluation = task.evaluate(test_path, prediction)
print(json.dumps({
    "scores": [float(v) for v in prediction.scores["score"]],
    "columns": list(prediction.scores.columns),
    "class_order": None if prediction.class_order is None else list(prediction.class_order),
    "metrics": {k: float(v) for k, v in (evaluation.metrics_raw or {}).items()},
}))
"""

REATTACH_TO_ENTITY = """
import json, sys
from fmlib.automl import BinaryTask

entity, test_path = sys.argv[1], sys.argv[2]
task = BinaryTask.load(entity)
stored = task.load_prediction(test_path)
print(json.dumps({
    "id": task.id,
    "fitted": task.is_fitted,
    "scores": [float(v) for v in stored.scores["score"]],
}))
"""

REATTACH_CALIBRATION = """
import json, sys
from fmlib.automl import ResponseTask

entity, test_path = sys.argv[1], sys.argv[2]
task = ResponseTask.load(entity)
stored = task.load_calibration(test_path)
print(json.dumps({
    "strategy": stored.calibration_strategy,
    "columns": list(stored.scores.columns),
    "scores": [float(v) for v in stored.scores["score"]],
}))
"""


def _binary_config(tmp_path: Path, backend: str, **overrides) -> BinaryTaskConfig:
    values: dict = {
        "output_dir": tmp_path / "out",
        "env_type": "local",
        "device": "cpu",
        "backend": backend,
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": None,
        "date_column": "report_month",
        "categorical_columns": ["segment"],
        "numerical_columns": ["feature", "other"],
        "hidden_state_columns": (),
        "hyperopt": False,
        "verbose": False,
        "random_state": 42,
    }
    if backend == "boosting":
        values["engine"] = "catboost"
        values["model_params"] = {"iterations": 30, "depth": 3, "thread_count": 1}
    else:
        values["engine"] = "tabular_transformer"
        values["model_params"] = dict(TABNN_PARAMS)
    values.update(overrides)
    return BinaryTaskConfig(**values)


@pytest.mark.parametrize(
    "backend", ["boosting", pytest.param("tabnn", marks=pytest.mark.slow)]
)
def test_the_artifact_scores_the_same_from_a_cold_interpreter(tmp_path, backend):
    """A saved artifact must carry everything needed to score, and nothing less."""
    train = _write(tmp_path / "train.parquet", 600, 1)
    valid = _write(tmp_path / "valid.parquet", 300, 2)
    test = _write(tmp_path / "test.parquet", 300, 3)

    task = BinaryTask(_binary_config(tmp_path, backend))
    task.train(train, valid)
    artifact = task.save(tmp_path / "artifact")

    here = task.predict(test)
    here_evaluation = task.evaluate(test, here)
    there = _run_child(SCORE_FROM_ARTIFACT, str(artifact), str(test))

    assert there["columns"] == list(here.scores.columns)
    assert there["class_order"] == (
        None if here.class_order is None else list(here.class_order)
    )
    np.testing.assert_array_equal(
        np.asarray(there["scores"]), here.scores["score"].to_numpy()
    )
    assert there["metrics"] == pytest.approx(dict(here_evaluation.metrics_raw or {}))


def test_an_entity_is_reattachable_from_a_cold_interpreter(tmp_path):
    """The documented recovery after a driver dies, exercised as documented.

    `Task.load(entity_path)` is what `examples/automl/README.md` tells a user to
    run when the process that submitted the work is gone. Note that building a
    task from the same config instead creates a *new* entity with a new id --
    every construction does -- so the path is the handle, not the config.
    """
    train = _write(tmp_path / "train.parquet", 600, 1)
    test = _write(tmp_path / "test.parquet", 300, 3)

    task = BinaryTask(_binary_config(tmp_path, "boosting"))
    task.train(train, train)
    here = task.predict(test)

    there = _run_child(REATTACH_TO_ENTITY, str(task.path), str(test))

    assert there["id"] == task.id
    assert there["fitted"] is True
    np.testing.assert_array_equal(
        np.asarray(there["scores"]), here.scores["score"].to_numpy()
    )

    # The same config in a new process is a new lifecycle, not a resumed one.
    fresh = BinaryTask(_binary_config(tmp_path, "boosting"))
    assert fresh.id != task.id


@pytest.mark.parametrize("strategy", ["isotonic_regression", "beta_calibration"])
def test_calibration_survives_a_cold_interpreter(tmp_path, strategy):
    """A fitted calibrator is part of the entity, not of the process that fitted it."""
    train = _write(tmp_path / "train.parquet", 800, 1)
    test = _write(tmp_path / "test.parquet", 400, 3)
    calibration = _write(tmp_path / "calibration.parquet", 400, 4)

    config = ResponseTaskConfig(
        output_dir=tmp_path / "out",
        env_type="local",
        device="cpu",
        backend="boosting",
        engine="catboost",
        target_column="target",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=["segment"],
        numerical_columns=["feature", "other"],
        hidden_state_columns=(),
        hyperopt=False,
        verbose=False,
        random_state=42,
        model_params={"iterations": 30, "depth": 3, "thread_count": 1},
        treatment_column="treatment",
        inverse_treatment=False,
    )
    task = ResponseTask(config)
    task.train(train, train)
    task.predict(test)
    task.predict(calibration)
    here = task.calibrate(test, calibration, calibration_strategy=strategy)

    there = _run_child(REATTACH_CALIBRATION, str(task.path), str(test))

    assert there["strategy"] == here.calibration_strategy == strategy
    # Calibrated values replace `score` in place; the column set does not change.
    assert there["columns"] == list(here.scores.columns)
    np.testing.assert_array_equal(
        np.asarray(there["scores"]), here.scores["score"].to_numpy()
    )


LOAD_ONLY = """
import json, sys
from fmlib.automl import BinaryTask
try:
    BinaryTask.load(sys.argv[1])
except Exception as error:
    print(json.dumps({"ok": False, "error": type(error).__name__}))
else:
    print(json.dumps({"ok": True, "error": None}))
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    "removed", ["backend.json", "model.safetensors", "preprocessor.yaml"]
)
def test_an_incomplete_artifact_refuses_to_load(tmp_path, removed):
    """A file missing from an artifact must stop the load, not change the answer.

    The failure mode this rules out is the quiet one: a reader that falls back
    to a default when part of the artifact is absent scores *something*, and
    nobody finds out until the numbers are compared against a run nobody kept.
    """
    train = _write(tmp_path / "train.parquet", 600, 1)
    valid = _write(tmp_path / "valid.parquet", 300, 2)

    task = BinaryTask(_binary_config(tmp_path, "tabnn"))
    task.train(train, valid)
    artifact = Path(task.save(tmp_path / "artifact"))

    assert _run_child(LOAD_ONLY, str(artifact))["ok"] is True

    victims = sorted(artifact.rglob(removed))
    assert victims, f"the artifact has no {removed}"
    for victim in victims:
        victim.unlink()

    assert _run_child(LOAD_ONLY, str(artifact))["ok"] is False
