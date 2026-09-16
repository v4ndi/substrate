"""Terminal parity runner for boosting regression and autocampaignxfm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from avatar.automl import RegressionTask, RegressionTaskConfig
from avatar.automl.exceptions import ConfigError


@dataclass(frozen=True)
class RegressionParityCase:
    """Resolved native configurations and comparison tolerances."""

    autocampaign_config_path: Path
    fmlib_config_path: Path
    fmlib_config: RegressionTaskConfig
    train_path: Path
    valid_path: Path
    test_path: Path
    output_dir: Path
    engine: str
    model_scope: str
    target_column: str
    client_id_column: str
    report_month_column: str
    group_column: str | None
    treatment_column: str | None
    inverse_treatment: bool
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    hidden_state_columns: tuple[str, ...]
    n_trials: int
    random_state: int
    metric: str
    max_relative_metric_difference: float = 0.15
    max_score_ks_statistic: float = 0.1
    max_score_quantile_difference: float = 0.25


def _absolute(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        msg = f"{name} must be absolute; got {path}"
        raise ConfigError(msg)
    return path.resolve()


def load_case(
    autocampaign_config_path: str | Path, fmlib_config_path: str | Path
) -> RegressionParityCase:
    """Load and cross-check independent native regression configurations."""
    if not OmegaConf.has_resolver("date"):
        OmegaConf.register_new_resolver("date", date.fromisoformat)
    reference_path = Path(autocampaign_config_path).resolve()
    payload = OmegaConf.to_container(OmegaConf.load(reference_path), resolve=True)
    fmlib_path = Path(fmlib_config_path).resolve()
    config = RegressionTaskConfig.from_yaml(fmlib_path)
    data, train, model = payload["data"], payload["train"], payload["model"]
    scope = "product" if payload["evaluate"]["is_product"] else "group"
    expected = {
        "engine": model["boosting_type"],
        "target_column": data["target_column"],
        "report_month_column": data.get("report_month_column", "report_month"),
        "group_column": data.get("group_column"),
        "treatment_column": data.get("treatment_control_group_column"),
        "inverse_treatment": bool(data["inverse_treatment"]),
        "model_scope": scope,
        "n_trials": int(train["trials"]),
        "metric": train["optimization_type"],
    }
    mismatches = {
        name: (value, getattr(config, name))
        for name, value in expected.items()
        if value != getattr(config, name)
    }
    if payload.get("task_type") != "reg" or mismatches:
        msg = f"Regression parity configs disagree: task_type={payload.get('task_type')!r}, fields={mismatches}"
        raise ConfigError(msg)
    inputs = data["input_dir"]
    return RegressionParityCase(
        autocampaign_config_path=reference_path,
        fmlib_config_path=fmlib_path,
        fmlib_config=config,
        train_path=_absolute(inputs["train"], "data.input_dir.train"),
        valid_path=_absolute(inputs["valid"], "data.input_dir.valid"),
        test_path=_absolute(inputs["test"], "data.input_dir.test"),
        output_dir=Path(config.output_dir).resolve().parent,
        engine=config.engine,
        model_scope=scope,
        target_column=config.target_column,
        client_id_column=config.client_id_column,
        report_month_column=config.report_month_column,
        group_column=config.group_column,
        treatment_column=config.treatment_column,
        inverse_treatment=config.inverse_treatment,
        categorical_columns=tuple(config.categorical_columns),
        numerical_columns=tuple(config.numerical_columns),
        hidden_state_columns=tuple(config.hidden_state_columns),
        n_trials=config.n_trials,
        random_state=config.random_state,
        metric=str(config.metric),
    )


def _metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(target, score)),
        "mae": float(mean_absolute_error(target, score)),
        "mape": float(mean_absolute_percentage_error(target, score)),
    }


def run_fmlib(case: RegressionParityCase) -> dict[str, Any]:
    """Run fmlib training/prediction/evaluation and persist normalized results."""
    output = case.output_dir / "fmlib"
    output.mkdir(parents=True, exist_ok=True)
    task = RegressionTask(replace(case.fmlib_config, output_dir=output))
    started = perf_counter()
    training = task.train(case.train_path, case.valid_path)
    train_seconds = perf_counter() - started
    started = perf_counter()
    prediction = task.predict(case.test_path)
    predict_seconds = perf_counter() - started
    evaluation = task.evaluate(case.test_path, prediction)
    raw_scores = (
        prediction.raw_scores
        if prediction.raw_scores is not None
        else prediction.scores
    )
    calibrated_scores = prediction.scores if prediction.raw_scores is not None else None
    scores_path = output / "scores.parquet"
    raw_scores.write_parquet(scores_path)
    calibrated_scores_path = output / "scores_calibrated.parquet"
    if calibrated_scores is not None:
        calibrated_scores.write_parquet(calibrated_scores_path)
    result = {
        "pipeline": "fmlib",
        "selected_params": training.best_params,
        "validation_objective": training.validation_metrics,
        "test_metrics": evaluation.metrics,
        "calibrated_test_metrics": evaluation.calibrated_metrics,
        "timing_seconds": {
            "train": train_seconds,
            "predict": predict_seconds,
            "total": train_seconds + predict_seconds,
        },
        "scores_path": str(scores_path),
        "calibrated_scores_path": str(calibrated_scores_path)
        if calibrated_scores is not None
        else None,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _distribution(
    reference: pl.DataFrame, candidate: pl.DataFrame, case: RegressionParityCase
) -> dict[str, Any]:
    keys = [case.client_id_column, case.report_month_column]
    reference = reference.with_columns(pl.col(case.report_month_column).cast(pl.String))
    candidate = candidate.with_columns(pl.col(case.report_month_column).cast(pl.String))
    joined = reference.rename({"score": "reference"}).join(
        candidate.rename({"score": "candidate"}), on=keys, validate="1:1"
    )
    left, right = joined["reference"].to_numpy(), joined["candidate"].to_numpy()
    grid = np.asarray([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    scale = max(float(np.std(left)), np.finfo(float).eps)
    quantile_difference = float(
        np.max(np.abs(np.quantile(left, grid) - np.quantile(right, grid))) / scale
    )
    combined = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(np.sort(left), combined, side="right") / len(left)
    right_cdf = np.searchsorted(np.sort(right), combined, side="right") / len(right)
    ks = float(np.max(np.abs(left_cdf - right_cdf)))
    return {
        "rows": joined.height,
        "ks_statistic": ks,
        "normalized_max_quantile_difference": quantile_difference,
        "passed": ks <= case.max_score_ks_statistic
        and quantile_difference <= case.max_score_quantile_difference,
    }


def compare_results(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    case: RegressionParityCase,
) -> dict[str, Any]:
    """Compare all regression metrics, score distributions, params and elapsed time."""

    def compare_metrics(reference_metrics, candidate_metrics):
        rows = {}
        for name in ("mse", "mae", "mape"):
            left, right = float(reference_metrics[name]), float(candidate_metrics[name])
            relative = abs(left - right) / max(abs(left), np.finfo(float).eps)
            rows[name] = {
                "autocampaignxfm": left,
                "fmlib": right,
                "relative_difference": relative,
            }
        return rows

    metric_rows = compare_metrics(reference["test_metrics"], candidate["test_metrics"])
    calibrated_metric_rows = None
    if reference.get("calibrated_test_metrics") and candidate.get(
        "calibrated_test_metrics"
    ):
        calibrated_metric_rows = compare_metrics(
            reference["calibrated_test_metrics"], candidate["calibrated_test_metrics"]
        )
    distribution = _distribution(
        pl.read_parquet(reference["scores_path"]),
        pl.read_parquet(candidate["scores_path"]),
        case,
    )
    calibrated_distribution = None
    if reference.get("calibrated_scores_path") and candidate.get(
        "calibrated_scores_path"
    ):
        calibrated_distribution = _distribution(
            pl.read_parquet(reference["calibrated_scores_path"]),
            pl.read_parquet(candidate["calibrated_scores_path"]),
            case,
        )
    all_metric_rows = list(metric_rows.values()) + (
        list(calibrated_metric_rows.values())
        if calibrated_metric_rows is not None
        else []
    )
    metrics_passed = all(
        row["relative_difference"] <= case.max_relative_metric_difference
        for row in all_metric_rows
    )
    params_equal = reference["selected_params"] == candidate["selected_params"]
    return {
        "passed": metrics_passed
        and distribution["passed"]
        and (calibrated_distribution is None or calibrated_distribution["passed"]),
        "metrics": metric_rows,
        "calibrated_metrics": calibrated_metric_rows,
        "score_distribution": distribution,
        "calibrated_score_distribution": calibrated_distribution,
        "best_params_equal": params_equal,
        "best_param_note": (
            "Native categorical CatBoost features in fmlib can change objectives and the selected trial versus the "
            "numeric encoded matrix; fmlib intentionally keeps native categorical handling."
        ),
        "selected_params": {
            "autocampaignxfm": reference["selected_params"],
            "fmlib": candidate["selected_params"],
        },
        "timing_seconds": {
            "autocampaignxfm": reference["timing_seconds"],
            "fmlib": candidate["timing_seconds"],
        },
    }


def run_reference(
    case: RegressionParityCase, autocampaign_root: Path, autocampaign_python: Path
) -> dict[str, Any]:
    """Run the configured reference trainer/evaluator in its own environment."""
    environment = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((
        str(autocampaign_root.resolve()),
        str(repo_root),
    ))
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["MPLBACKEND"] = "Agg"
    subprocess.run(
        [
            str(autocampaign_python.resolve()),
            "-m",
            "tools.automl_parity.regression_reference",
            "--config",
            str(case.autocampaign_config_path),
            "--fmlib-config",
            str(case.fmlib_config_path),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
    )
    return json.loads(
        (case.output_dir / "autocampaignxfm" / "result.json").read_text(
            encoding="utf-8"
        )
    )


def run_comparison(
    autocampaign_config_path: str | Path,
    fmlib_config_path: str | Path,
    *,
    autocampaign_root: str | Path,
    autocampaign_python: str | Path = sys.executable,
) -> dict[str, Any]:
    """Run both complete workflows and write ``comparison.json``."""
    case = load_case(autocampaign_config_path, fmlib_config_path)
    reference = run_reference(case, Path(autocampaign_root), Path(autocampaign_python))
    candidate = run_fmlib(case)
    comparison = compare_results(reference, candidate, case)
    case.output_dir.mkdir(parents=True, exist_ok=True)
    (case.output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autocampaign-config", required=True, type=Path)
    parser.add_argument("--fmlib-config", required=True, type=Path)
    parser.add_argument("--autocampaign-root", required=True, type=Path)
    parser.add_argument(
        "--autocampaign-python", type=Path, default=Path(sys.executable)
    )
    args = parser.parse_args()
    result = run_comparison(
        args.autocampaign_config,
        args.fmlib_config,
        autocampaign_root=args.autocampaign_root,
        autocampaign_python=args.autocampaign_python,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
