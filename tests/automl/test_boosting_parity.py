"""Bit-level parity gate for the boosting AutoML path.

Six configurations cover every task class, both engines, all three model
layouts and both hyperopt modes. Each one runs the full public lifecycle --
``train -> save -> load -> predict -> evaluate`` -- and is reduced to a
fingerprint: ``best_params``, validation metrics, feature names, class order,
evaluation metrics and a sha256 over the predicted scores. The fingerprints
are compared against ``boosting_parity_baseline.json``, committed next to this
file.

The gate exists because every stage of the TabNN work touches shared code:
the package rename, the backend-neutral seams, the lazy data descriptors, the
hidden-state schema. None of them is allowed to move a boosting number.

The suite is marked ``slow`` and deselected by default::

    .venv/bin/python -m pytest -m slow tests/automl/test_boosting_parity.py

To re-record the baseline after an *intentional* change, run the same command
with ``AUTOML_PARITY_RECORD=1``. Re-recording is a reviewable diff: if the
change was supposed to be behaviour-preserving, that diff must be empty.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from avatar import automl

pytestmark = pytest.mark.slow

BASELINE_PATH = Path(__file__).with_name("boosting_parity_baseline.json")
RECORD = os.environ.get("AUTOML_PARITY_RECORD") == "1"

_RECORDED: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# Deterministic synthetic data                                                 #
# --------------------------------------------------------------------------- #
def _build_frame(rows: int, seed: int) -> pl.DataFrame:
    """One split of the synthetic dataset.

    The target is a logistic function of the features so that every task has
    real signal to fit; ``treatment`` interacts with ``cat_1`` so uplift is not
    identically zero; ``seq_hidden_state`` is a fixed-width float32 list so the
    hidden-state path is exercised too.
    """
    rng = np.random.default_rng(seed)
    num_1 = rng.normal(size=rows)
    num_2 = rng.normal(size=rows)
    num_3 = rng.gamma(2.0, 1.0, size=rows)
    cat_1 = rng.choice(["a", "b", "c"], size=rows)
    cat_2 = rng.choice(["x", "y"], size=rows)
    group = rng.choice(["retail", "corp"], size=rows)
    treatment = rng.integers(0, 2, size=rows)
    hidden = rng.normal(size=(rows, 4)).astype(np.float32)
    logit = (
        1.2 * num_1
        - 0.8 * num_2
        + 0.3 * num_3
        + 0.7 * (cat_1 == "a")
        - 0.5 * (cat_2 == "y")
        + 0.6 * treatment * (cat_1 == "a")
        + hidden[:, 0]
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(size=rows) < prob).astype(np.int8)
    target_reg = logit + rng.normal(scale=0.3, size=rows)
    target_mc = np.digitize(logit, [-0.5, 0.8]).astype(np.int8)
    months = rng.choice(["2025-01-01", "2025-02-01", "2025-03-01"], size=rows)
    return pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 1_000_000,
        "report_month": pl.Series(months).str.to_date(),
        "group": group,
        "cat_1": cat_1,
        "cat_2": cat_2,
        "num_1": num_1,
        "num_2": num_2,
        "num_3": num_3,
        "seq_hidden_state": [row.tolist() for row in hidden],
        "treatment": treatment.astype(np.int8),
        "y_binary": target,
        "y_reg": target_reg,
        "y_mc": target_mc,
    }).with_columns(pl.col("seq_hidden_state").cast(pl.List(pl.Float32)))


@pytest.fixture(scope="session")
def parity_data(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("boosting_parity_data")
    paths = {}
    for name, rows, seed in (("train", 4000, 1), ("valid", 1500, 2), ("test", 1500, 3)):
        path = root / f"{name}.parquet"
        _build_frame(rows, seed).write_parquet(path)
        paths[name] = path
    return paths


# --------------------------------------------------------------------------- #
# The matrix                                                                   #
# --------------------------------------------------------------------------- #
_COMMON = dict(
    env_type="local",
    device="cpu",
    client_id_column="epk_id",
    categorical_columns=["cat_1", "cat_2"],
    numerical_columns=["num_1", "num_2", "num_3"],
    hidden_state_columns=["seq_hidden_state"],
    backend="boosting",
    verbose=False,
    random_state=42,
)

CASES: dict[str, tuple[type, type, dict]] = {
    "binary_catboost_both": (
        automl.BinaryTask,
        automl.BinaryTaskConfig,
        dict(
            _COMMON,
            engine="catboost",
            target_column="y_binary",
            group_column="group",
            date_column="report_month",
            model_layout="global_and_per_group",
            hyperopt=True,
            n_trials=2,
        ),
    ),
    "binary_xgboost_per_group": (
        automl.BinaryTask,
        automl.BinaryTaskConfig,
        dict(
            _COMMON,
            engine="xgboost",
            target_column="y_binary",
            group_column="group",
            date_column="report_month",
            model_layout="per_group",
            hyperopt=False,
            model_params={"n_estimators": 40, "max_depth": 3},
        ),
    ),
    "response_catboost_global": (
        automl.ResponseTask,
        automl.ResponseTaskConfig,
        dict(
            _COMMON,
            engine="catboost",
            target_column="y_binary",
            group_column=None,
            date_column="report_month",
            hyperopt=False,
            model_params={"iterations": 60, "depth": 4},
            treatment_column="treatment",
            inverse_treatment=False,
        ),
    ),
    "regression_xgboost_hyperopt": (
        automl.RegressionTask,
        automl.RegressionTaskConfig,
        dict(
            _COMMON,
            engine="xgboost",
            target_column="y_reg",
            group_column=None,
            date_column=None,
            hyperopt=True,
            n_trials=2,
        ),
    ),
    "multiclass_catboost_global": (
        automl.MulticlassTask,
        automl.MulticlassTaskConfig,
        dict(
            _COMMON,
            engine="catboost",
            target_column="y_mc",
            group_column=None,
            date_column="report_month",
            hyperopt=False,
            model_params={"iterations": 60, "depth": 4},
        ),
    ),
    "uplift_catboost_global": (
        automl.UpliftTask,
        automl.UpliftTaskConfig,
        dict(
            _COMMON,
            engine="catboost",
            target_column="y_binary",
            group_column=None,
            date_column="report_month",
            hyperopt=False,
            model_params={"iterations": 60, "depth": 4},
            treatment_column="treatment",
            inverse_treatment=False,
            estimate_propensity=True,
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Fingerprinting                                                               #
# --------------------------------------------------------------------------- #
def _digest(frame: pl.DataFrame) -> dict:
    """Shape plus a rounded sha256 per numeric column.

    Rounding to six decimals before hashing keeps the fingerprint stable under
    the last-bit noise of threaded boosting while still catching any change
    that a human would call a change.
    """
    out: dict = {"rows": frame.height, "columns": frame.columns}
    for name, dtype in frame.schema.items():
        if dtype.is_numeric():
            values = np.round(frame[name].to_numpy().astype(float), 6)
            out[f"sha:{name}"] = hashlib.sha256(values.tobytes()).hexdigest()[:16]
            out[f"mean:{name}"] = round(float(np.nanmean(values)), 8)
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.floating | float):
        return round(float(value), 8)
    if isinstance(value, np.integer | int | bool | str) or value is None:
        return value
    return str(value)


def _run_case(name: str, output_dir: Path, data: dict[str, Path]) -> dict:
    task_class, config_class, values = CASES[name]
    config = config_class(output_dir=output_dir, **values)
    task = task_class(config)

    training = task.train(data["train"], data["valid"])
    artifact = task.save(output_dir / "artifact")
    restored = task_class.load(artifact)
    prediction = restored.predict(data["test"])
    evaluation = restored.evaluate(data["test"])

    return {
        "best_params": _jsonable(dict(training.best_params)),
        "validation_metrics": _jsonable(dict(training.validation_metrics)),
        "feature_names": list(training.feature_names),
        "backend": training.backend_name,
        "engine": training.engine_name,
        "class_order": _jsonable(prediction.class_order),
        "scores": _jsonable(_digest(prediction.scores)),
        "metrics_raw": _jsonable(dict(evaluation.metrics_raw or {})),
        "metrics_by_group_raw": (
            None
            if evaluation.metrics_by_group_raw is None
            else _jsonable(_digest(evaluation.metrics_by_group_raw))
        ),
    }


def _baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.fail(
            f"{BASELINE_PATH.name} is missing; record it with "
            "AUTOML_PARITY_RECORD=1 pytest -m slow tests/automl/test_boosting_parity.py"
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def _write_recorded():
    yield
    if not RECORD:
        return
    merged = (
        json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if BASELINE_PATH.exists()
        else {}
    )
    merged.update(_RECORDED)
    BASELINE_PATH.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_boosting_parity(case, parity_data, tmp_path):
    record = _run_case(case, tmp_path / case, parity_data)
    if RECORD:
        _RECORDED[case] = record
        return
    assert record == _baseline().get(case)
