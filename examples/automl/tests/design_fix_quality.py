"""Deterministic quality checkpoint for the AutoML metric registry.

The checkpoint reuses fixed datasets and the after_point_8 reference. Dataset
generation remains available only for maintaining the original migration
harness and is never called by the metric-registry checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl

ROWS_PER_TASK = 200_000
SPLITS = {
    "train": (0, 100_000, "2026-01-01"),
    "valid": (100_000, 140_000, "2026-02-01"),
    "calibration": (140_000, 170_000, "2026-03-01"),
    "test": (170_000, 200_000, "2026-04-01"),
}
TASKS = ("binary", "response", "regression", "multiclass", "uplift")
OPTIMIZATION_METRICS = {
    "binary": "roc_auc",
    "response": "roc_auc",
    "regression": "mse",
    "multiclass": "roc_auc_ovr_macro",
    "uplift": "qini_auc",
}
EVALUATION_METRICS = {
    "binary": (
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
    ),
    "response": (
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
    ),
    "regression": ("mse", "mae", "mape"),
    "multiclass": (
        "roc_auc_ovr_macro",
        "log_loss",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "class_wise_roc_auc",
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
    ),
    "uplift": (
        "qini_auc",
        "uplift_auc",
        "uplift_at_10",
        "uplift_at_20",
        "uplift_at_50",
        "treatment_roc_auc",
        "control_roc_auc",
    ),
}
SEMANTIC_CONFIG = {
    "backend": "boosting",
    "engine": "xgboost",
    "device": "cpu",
    "env_type": "local",
    "hyperopt": True,
    "n_trials": 3,
    "random_state": 42,
    "model_layout": "global_and_per_group",
    "search_space": {
        "n_estimators": [32],
        "max_depth": [2, 3],
        "eta": [0.1],
        "lambda": [1.0],
        "alpha": [0.0],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "n_jobs": [1],
    },
    "verbose": False,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def _fingerprint_files(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.parquet")):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _base_frame(seed: int) -> tuple[pl.DataFrame, np.random.Generator]:
    rng = np.random.default_rng(seed)
    row = np.arange(ROWS_PER_TASK)
    n1 = rng.normal(size=ROWS_PER_TASK)
    n2 = rng.normal(size=ROWS_PER_TASK)
    n3 = rng.normal(size=ROWS_PER_TASK)
    n3[row % 97 == 0] = np.nan
    group = np.asarray(["group_a", "group_b", "group_c", "group_d"])[row % 4]
    cat = np.asarray(["cat_0", "cat_1", "cat_2", "cat_3", "cat_4"])[(row // 3) % 5]
    segment = np.asarray(["s0", "s1", "s2"])[(row // 7) % 3]
    frame = pl.DataFrame({
        "epk_id": (row // 2).astype(str),
        "group": group,
        "category": cat,
        "segment": segment,
        "n1": n1,
        "n2": n2,
        "n3": n3,
    })
    return frame, rng


def _task_frame(task: str, seed: int) -> pl.DataFrame:
    frame, rng = _base_frame(seed)
    n1 = frame["n1"].to_numpy()
    n2 = frame["n2"].to_numpy()
    group_effect = np.asarray([0.5, -0.25, 0.15, -0.4])[np.arange(ROWS_PER_TASK) % 4]
    category_effect = np.asarray([0.4, -0.3, 0.2, -0.1, 0.0])[
        (np.arange(ROWS_PER_TASK) // 3) % 5
    ]
    noise = rng.normal(scale=0.7, size=ROWS_PER_TASK)
    if task == "regression":
        target = 2.0 * n1 - 1.3 * n2 + group_effect + category_effect + noise
        return frame.with_columns(pl.Series("target", target))
    if task == "multiclass":
        logits = np.column_stack((
            1.2 * n1 + category_effect,
            -0.9 * n1 + 0.8 * n2 + group_effect,
            -0.6 * n2 - category_effect,
        ))
        logits += rng.normal(scale=0.5, size=logits.shape)
        labels = np.asarray(["class_a", "class_b", "class_c"])[
            np.argmax(logits, axis=1)
        ]
        return frame.with_columns(pl.Series("target", labels))
    treatment = rng.binomial(1, 0.5, size=ROWS_PER_TASK)
    if task == "binary":
        logit = 1.1 * n1 - 0.8 * n2 + group_effect + category_effect + noise
    elif task == "response":
        logit = (
            1.0 * n1
            - 0.7 * n2
            + 0.55 * treatment
            + group_effect
            + category_effect
            + noise
        )
        frame = frame.with_columns(pl.Series("treatment", treatment))
    elif task == "uplift":
        heterogeneous_effect = 0.9 * (n1 > 0).astype(float) - 0.35 * (n2 > 0).astype(
            float
        )
        logit = (
            -0.2
            + 0.7 * n1
            - 0.5 * n2
            + group_effect
            + treatment * heterogeneous_effect
            + noise
        )
        frame = frame.with_columns(pl.Series("treatment", treatment))
    else:
        raise ValueError(task)
    probability = 1.0 / (1.0 + np.exp(-logit))
    return frame.with_columns(pl.Series("target", rng.binomial(1, probability)))


def generate(root: Path) -> dict[str, str]:
    data_root = root / "data"
    fingerprints: dict[str, str] = {}
    for task_index, task in enumerate(TASKS):
        task_root = data_root / task
        if not task_root.exists():
            frame = _task_frame(task, 10_000 + task_index)
            for split, (start, stop, date) in SPLITS.items():
                split_root = task_root / split
                split_root.mkdir(parents=True, exist_ok=True)
                part = frame.slice(start, stop - start).with_columns(
                    pl.lit(date).alias("date")
                )
                midpoint = part.height // 2
                part.slice(0, midpoint).write_parquet(split_root / "part-000.parquet")
                part.slice(midpoint).write_parquet(split_root / "part-001.parquet")
        fingerprints[task] = _fingerprint_files(task_root)
    payload = {"rows_per_task": ROWS_PER_TASK, "fingerprints": fingerprints}
    (root / "dataset_fingerprints.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return fingerprints


def _current_config(task: str, root: Path, checkpoint: str):
    from avatar.automl import (
        BinaryTaskConfig,
        MulticlassTaskConfig,
        RegressionTaskConfig,
        ResponseTaskConfig,
        UpliftTaskConfig,
    )

    classes = {
        "binary": BinaryTaskConfig,
        "response": ResponseTaskConfig,
        "regression": RegressionTaskConfig,
        "multiclass": MulticlassTaskConfig,
        "uplift": UpliftTaskConfig,
    }
    values = {
        **SEMANTIC_CONFIG,
        "optimization_metric": OPTIMIZATION_METRICS[task],
        "target_column": "target",
        "client_id_column": "epk_id",
        "date_column": "date",
        "group_column": "group",
        "categorical_columns": ["category", "segment"],
        "numerical_columns": ["n1", "n2", "n3"],
        "hidden_state_columns": (),
        "output_dir": root / "entities" / checkpoint / task,
    }
    if task == "response":
        values.update(treatment_column="treatment", inverse_treatment=False)
    if task == "uplift":
        values.update(
            treatment_column="treatment",
            inverse_treatment=False,
            estimate_propensity=False,
        )
    return classes[task](**values)


def _canonical_scores(frame: pl.DataFrame) -> pl.DataFrame:
    renames = {}
    if "model_scope" in frame.columns:
        renames["model_scope"] = "model_layout"
    result = frame.rename(renames)
    if "model_layout" in result.columns:
        result = result.with_columns(
            pl.col("model_layout").replace({"product": "global", "group": "per_group"})
        )
    keys = [
        name
        for name in ("epk_id", "date", "model_layout", "group")
        if name in result.columns
    ]
    occurrence = "__occurrence"
    if keys:
        result = result.with_columns(
            pl.int_range(pl.len()).over(keys).alias(occurrence)
        ).sort([*keys, occurrence])
    return result


def _canonical_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in metrics.items():
        if key.startswith("product_"):
            key = f"global_{key.removeprefix('product_')}"
        elif key.startswith("group_"):
            key = f"per_group_{key.removeprefix('group_')}"
        result[key] = value
    return result


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, sort_keys=True))


def _semantic_config(config: Any) -> dict[str, Any]:
    result = _json_value(asdict(config))
    result.pop("output_dir")
    return result


def _reference_semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["optimization_metric"] = result.pop("metric")
    result.pop("output_dir")
    return result


def _config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["/usr/bin/git", "status", "--short", "--untracked-files=no"],
            cwd=repository,
            text=True,
        ).strip()
    )
    return {"commit": commit, "tracked_worktree_dirty": dirty}


def _numeric_deltas(
    current: dict[str, Any], reference: dict[str, Any], label: str
) -> dict[str, float]:
    if set(current) != set(reference):
        msg = f"{label} keys changed: {sorted(current)} != {sorted(reference)}"
        raise AssertionError(msg)
    return {
        name: abs(float(current[name]) - float(reference[name])) for name in current
    }


def _comparison(
    reference_root: Path,
    scores: pl.DataFrame,
    metrics: dict[str, Any],
    training: dict[str, Any],
    config: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    reference_scores = pl.read_parquet(reference_root / f"{task}_scores.parquet")
    if (
        scores.columns != reference_scores.columns
        or scores.height != reference_scores.height
    ):
        msg = (
            f"{task}: score schema/rows changed: {scores.columns}/{scores.height} != "
            f"{reference_scores.columns}/{reference_scores.height}"
        )
        raise AssertionError(msg)
    score_columns = [name for name in scores.columns if name.startswith("score")]
    key_columns = [name for name in scores.columns if name not in score_columns]
    if key_columns and not scores.select(key_columns).equals(
        reference_scores.select(key_columns)
    ):
        msg = f"{task}: ordered identity/model-branch keys changed"
        raise AssertionError(msg)
    score_delta = float(
        np.max(
            np.abs(
                scores.select(score_columns).to_numpy()
                - reference_scores.select(score_columns).to_numpy()
            )
        )
    )
    reference_payload = json.loads(
        (reference_root / f"{task}.json").read_text(encoding="utf-8")
    )
    metric_deltas = _numeric_deltas(
        metrics, reference_payload["metrics"], f"{task} metrics"
    )
    validation_deltas = _numeric_deltas(
        training["validation_metrics"],
        reference_payload["training"]["validation_metrics"],
        f"{task} validation metrics",
    )
    metric_delta = max(metric_deltas.values(), default=0.0)
    validation_delta = max(validation_deltas.values(), default=0.0)
    if training["best_params"] != reference_payload["training"]["best_params"]:
        msg = f"{task}: selected best parameters changed"
        raise AssertionError(msg)
    reference_config = _reference_semantic_config(reference_payload["config"])
    if config != reference_config:
        msg = f"{task}: semantic config changed"
        raise AssertionError(msg)
    if max(score_delta, metric_delta, validation_delta) > 1e-8:
        msg = (
            f"{task}: score_delta={score_delta}, metric_delta={metric_delta}, "
            f"validation_delta={validation_delta}"
        )
        raise AssertionError(msg)
    return {
        "max_abs_score_delta": score_delta,
        "max_abs_metric_delta": metric_delta,
        "max_abs_validation_metric_delta": validation_delta,
        "metric_deltas": metric_deltas,
        "validation_metric_deltas": validation_deltas,
        "best_params_match": True,
        "config_fingerprint": _config_fingerprint(config),
        "reference_config_fingerprint": _config_fingerprint(reference_config),
    }


def run_checkpoint(root: Path, checkpoint: str) -> None:
    from avatar.automl import (
        BinaryTask,
        MulticlassTask,
        RegressionTask,
        ResponseTask,
        UpliftTask,
    )

    classes = {
        "binary": BinaryTask,
        "response": ResponseTask,
        "regression": RegressionTask,
        "multiclass": MulticlassTask,
        "uplift": UpliftTask,
    }
    checkpoint_root = root / "checkpoints" / checkpoint
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    fingerprint_payload = json.loads(
        (root / "dataset_fingerprints.json").read_text(encoding="utf-8")
    )
    fingerprints = {task: _fingerprint_files(root / "data" / task) for task in TASKS}
    reference_root = root / "checkpoints" / "after_point_8"
    reference_summary = json.loads(
        (reference_root / "summary.json").read_text(encoding="utf-8")
    )
    if (
        fingerprints != fingerprint_payload["fingerprints"]
        or fingerprints != reference_summary["dataset_fingerprints"]
    ):
        msg = "Dataset fingerprints differ from after_point_8 reference"
        raise AssertionError(msg)
    summary: dict[str, Any] = {
        "checkpoint": checkpoint,
        "reference_checkpoint": "after_point_8",
        "artifacts": {
            "checkpoint_root": str(checkpoint_root.resolve()),
            "notebook_source": str(Path(__file__).with_suffix(".ipynb").resolve()),
            "executed_notebook": str(
                (checkpoint_root / "design_fix_quality.executed.ipynb").resolve()
            ),
            "operation_logs_root": str((root / "entities" / checkpoint).resolve()),
        },
        "dataset_fingerprints": fingerprints,
        "semantic_config": SEMANTIC_CONFIG,
        "evaluation_metrics": EVALUATION_METRICS,
        "git": _git_state(),
        "tasks": {},
    }
    started = perf_counter()
    for task_name in TASKS:
        task_started = perf_counter()
        config = _current_config(task_name, root, checkpoint)
        task = classes[task_name](config)
        task_root = root / "data" / task_name
        training = task.train(task_root / "train", task_root / "valid")
        prediction = task.predict(task_root / "test")
        assert prediction is not None
        evaluation = task.evaluate(
            task_root / "test", prediction, metrics=EVALUATION_METRICS[task_name]
        )
        assert evaluation is not None
        scores = _canonical_scores(prediction.scores)
        assert evaluation.metrics_raw is not None
        metrics = _canonical_metrics(dict(evaluation.metrics_raw))
        training_payload = _json_value(asdict(training))
        config_payload = _semantic_config(config)
        comparison = _comparison(
            reference_root,
            scores,
            metrics,
            training_payload,
            config_payload,
            task_name,
        )
        scores.write_parquet(checkpoint_root / f"{task_name}_scores.parquet")
        task_payload = {
            "duration_seconds": perf_counter() - task_started,
            "config": _json_value(asdict(config)),
            "training": training_payload,
            "metrics": metrics,
            "score_rows": scores.height,
            "score_columns": scores.columns,
            "comparison": comparison,
        }
        (checkpoint_root / f"{task_name}.json").write_text(
            json.dumps(task_payload, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["tasks"][task_name] = task_payload
    summary["duration_seconds"] = perf_counter() - started
    (checkpoint_root / "summary.json").write_text(
        json.dumps(summary, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "checkpoint"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        generate(args.root)
    else:
        if not args.name:
            parser.error("checkpoint requires --name")
        run_checkpoint(args.root, args.name)


if __name__ == "__main__":
    main()
