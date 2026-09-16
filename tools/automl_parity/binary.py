"""Terminal runner for binary boosting parity with autocampaignxfm."""

from __future__ import annotations

import argparse
import json
import logging
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
from sklearn.metrics import roc_auc_score

from avatar.automl import BinaryTask, BinaryTaskConfig
from avatar.automl.data import ParquetSource
from avatar.automl.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryParityCase:
    """Resolved inputs shared by the fmlib and autocampaignxfm runners."""

    autocampaign_config_path: Path
    fmlib_config_path: Path
    fmlib_config: BinaryTaskConfig
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
    verbose: bool | int
    search_space: Mapping[str, Any] | None
    require_equal_best_params: bool
    max_roc_auc_difference: float
    max_score_ks_statistic: float
    max_score_quantile_difference: float


def _require_absolute_path(value: str | Path, field_name: str) -> Path:
    """Resolve one required absolute path from the parity YAML."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        msg = f"{field_name} must be an absolute path; got {path}"
        raise ConfigError(msg)
    return path.resolve()


def load_case(
    autocampaign_config_path: str | Path,
    fmlib_config_path: str | Path,
) -> BinaryParityCase:
    """Load independent autocampaignxfm and fmlib configurations."""
    if not OmegaConf.has_resolver("date"):
        OmegaConf.register_new_resolver("date", date.fromisoformat)
    config_path = Path(autocampaign_config_path).expanduser().resolve()
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, Mapping):
        msg = f"Parity YAML root must be a mapping: {config_path}"
        raise ConfigError(msg)
    data = payload["data"]
    train = payload["train"]
    model = payload["model"]
    resolved_fmlib_path = Path(fmlib_config_path).expanduser().resolve()
    fmlib_config = BinaryTaskConfig.from_yaml(resolved_fmlib_path)
    inputs = data["input_dir"]
    if not isinstance(inputs, Mapping):
        msg = "Binary parity requires data.input_dir with train, valid and test paths"
        raise ConfigError(msg)
    expected_fmlib_values = {
        "engine": str(model["boosting_type"]),
        "target_column": str(data["target_column"]),
        "report_month_column": str(data.get("report_month_column", "report_month")),
        "group_column": data.get("group_column"),
        "treatment_column": data.get("treatment_control_group_column"),
        "inverse_treatment": bool(data["inverse_treatment"]),
        "hidden_state_columns": tuple(data.get("hidden_state_columns") or ()),
        "n_trials": int(train["trials"]),
        "random_state": int(train.get("random_state", 42)),
    }
    reference_scope = "product" if bool(payload["evaluate"]["is_product"]) else "group"
    expected_fmlib_values["model_scope"] = reference_scope
    mismatches = {
        name: {"autocampaignxfm": expected, "fmlib": getattr(fmlib_config, name)}
        for name, expected in expected_fmlib_values.items()
        if getattr(fmlib_config, name) != expected
    }
    if mismatches:
        msg = f"Autocampaignxfm and fmlib configs disagree: {mismatches}"
        raise ConfigError(msg)
    output_dir = Path(fmlib_config.output_dir).expanduser().resolve().parent
    return BinaryParityCase(
        autocampaign_config_path=config_path,
        fmlib_config_path=resolved_fmlib_path,
        fmlib_config=fmlib_config,
        train_path=_require_absolute_path(inputs["train"], "data.input_dir.train"),
        valid_path=_require_absolute_path(inputs["valid"], "data.input_dir.valid"),
        test_path=_require_absolute_path(inputs["test"], "data.input_dir.test"),
        output_dir=output_dir,
        engine=str(model["boosting_type"]),
        model_scope=reference_scope,
        target_column=str(data["target_column"]),
        client_id_column=fmlib_config.client_id_column,
        report_month_column=str(
            data.get("report_month_column", fmlib_config.report_month_column)
        ),
        group_column=data.get("group_column"),
        treatment_column=data.get("treatment_control_group_column"),
        inverse_treatment=bool(data["inverse_treatment"]),
        categorical_columns=tuple(fmlib_config.categorical_columns),
        numerical_columns=tuple(fmlib_config.numerical_columns),
        hidden_state_columns=tuple(data.get("hidden_state_columns") or ()),
        n_trials=int(train["trials"]),
        random_state=fmlib_config.random_state,
        verbose=fmlib_config.verbose,
        search_space=fmlib_config.search_space,
        require_equal_best_params=False,
        max_roc_auc_difference=0.001,
        max_score_ks_statistic=0.02,
        max_score_quantile_difference=0.02,
    )


def task_config(case: BinaryParityCase, output_dir: Path) -> BinaryTaskConfig:
    """Build the fmlib configuration equivalent to the reference YAML."""
    return replace(case.fmlib_config, output_dir=output_dir)


def _jsonable(value: Any) -> Any:
    """Convert comparison payload values into JSON-compatible objects."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return repr(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one formatted UTF-8 JSON result file."""
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_fmlib(case: BinaryParityCase) -> dict[str, Any]:
    """Run fmlib search and scoring and persist a normalized result."""
    output_dir = case.output_dir / "fmlib"
    output_dir.mkdir(parents=True, exist_ok=True)
    task = BinaryTask(task_config(case, output_dir))

    started = perf_counter()
    training = task.train(case.train_path, case.valid_path)
    train_seconds = perf_counter() - started

    started = perf_counter()
    prediction = task.predict(case.test_path)
    predict_seconds = perf_counter() - started
    raw_scores = (
        prediction.raw_scores
        if prediction.raw_scores is not None
        else prediction.scores
    )
    calibrated_scores = prediction.scores if prediction.raw_scores is not None else None
    scores_path = output_dir / "scores.parquet"
    raw_scores.write_parquet(scores_path)
    calibrated_scores_path = output_dir / "scores_calibrated.parquet"
    if calibrated_scores is not None:
        calibrated_scores.write_parquet(calibrated_scores_path)

    test_frame = ParquetSource.resolve(case.test_path).read()
    scored_test = test_frame.with_columns(raw_scores["score"])
    calibrated_test = (
        test_frame.with_columns(calibrated_scores["score"])
        if calibrated_scores is not None
        else None
    )
    test_auc: dict[str, float] = {}
    calibrated_test_auc: dict[str, float] = {}
    resolved_model_params: dict[str, Any] = {}
    for item in task._models:
        model_name = task._model_name(item.scope, item.group_value)
        model_frame = scored_test
        if item.scope == "group":
            model_frame = scored_test.filter(
                pl.col(case.group_column) == item.group_value
            )
        test_auc[model_name] = float(
            roc_auc_score(
                model_frame[case.target_column].to_numpy(),
                model_frame["score"].to_numpy(),
            )
        )
        if calibrated_test is not None:
            calibrated_model_frame = calibrated_test
            if item.scope == "group":
                calibrated_model_frame = calibrated_test.filter(
                    pl.col(case.group_column) == item.group_value
                )
            calibrated_test_auc[model_name] = float(
                roc_auc_score(
                    calibrated_model_frame[case.target_column].to_numpy(),
                    calibrated_model_frame["score"].to_numpy(),
                )
            )
        resolved_model_params[model_name] = item.backend.model.get_params()
    result = {
        "pipeline": "fmlib",
        "selected_params": {
            name: dict(params) for name, params in training.best_params.items()
        },
        "resolved_model_params": resolved_model_params,
        "feature_names": list(training.feature_names),
        "validation_roc_auc": dict(training.validation_metrics),
        "test_roc_auc": test_auc,
        "calibrated_test_roc_auc": calibrated_test_auc,
        "timing_seconds": {
            "hyperparameter_search_and_train": train_seconds,
            "inference_model_fit": 0.0,
            "predict": predict_seconds,
            "total_train_before_predict": train_seconds,
        },
        "scores_path": str(scores_path),
        "calibrated_scores_path": str(calibrated_scores_path)
        if calibrated_scores is not None
        else None,
    }
    _write_json(output_dir / "result.json", result)
    return result


def _score_distribution_comparison(
    reference: pl.DataFrame,
    candidate: pl.DataFrame,
    case: BinaryParityCase,
) -> dict[str, Any]:
    """Validate the scored population and compare score distributions."""
    join_columns = [case.client_id_column]
    if (
        case.report_month_column in reference.columns
        and case.report_month_column in candidate.columns
    ):
        join_columns.append(case.report_month_column)
        reference = reference.with_columns(
            pl.col(case.report_month_column).cast(pl.String)
        )
        candidate = candidate.with_columns(
            pl.col(case.report_month_column).cast(pl.String)
        )
    joined = reference.rename({"score": "reference_score"}).join(
        candidate.rename({"score": "candidate_score"}),
        on=join_columns,
        how="inner",
        validate="1:1",
    )
    if joined.height != reference.height or joined.height != candidate.height:
        msg = (
            "Score rows do not match by identifiers: "
            f"reference={reference.height}, candidate={candidate.height}, joined={joined.height}"
        )
        raise AssertionError(msg)
    reference_scores = joined["reference_score"].to_numpy()
    candidate_scores = joined["candidate_score"].to_numpy()
    quantile_grid = np.asarray((0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99))
    reference_quantiles = np.quantile(reference_scores, quantile_grid)
    candidate_quantiles = np.quantile(candidate_scores, quantile_grid)
    quantile_differences = np.abs(reference_quantiles - candidate_quantiles)
    combined = np.sort(np.concatenate((reference_scores, candidate_scores)))
    reference_sorted = np.sort(reference_scores)
    candidate_sorted = np.sort(candidate_scores)
    reference_cdf = (
        np.searchsorted(reference_sorted, combined, side="right")
        / reference_scores.size
    )
    candidate_cdf = (
        np.searchsorted(candidate_sorted, combined, side="right")
        / candidate_scores.size
    )
    ks_statistic = float(np.max(np.abs(reference_cdf - candidate_cdf)))
    max_quantile_difference = float(quantile_differences.max())
    passed = (
        ks_statistic <= case.max_score_ks_statistic
        and max_quantile_difference <= case.max_score_quantile_difference
    )
    return {
        "passed": passed,
        "rows": joined.height,
        "ks_statistic": ks_statistic,
        "max_ks_statistic": case.max_score_ks_statistic,
        "max_quantile_absolute_difference": max_quantile_difference,
        "max_quantile_difference": case.max_score_quantile_difference,
        "summary": {
            "autocampaignxfm": {
                "mean": float(reference_scores.mean()),
                "std": float(reference_scores.std()),
                "min": float(reference_scores.min()),
                "max": float(reference_scores.max()),
            },
            "fmlib": {
                "mean": float(candidate_scores.mean()),
                "std": float(candidate_scores.std()),
                "min": float(candidate_scores.min()),
                "max": float(candidate_scores.max()),
            },
        },
        "quantiles": {
            f"{quantile:g}": {
                "autocampaignxfm": float(reference_value),
                "fmlib": float(candidate_value),
                "absolute_difference": float(difference),
            }
            for quantile, reference_value, candidate_value, difference in zip(
                quantile_grid,
                reference_quantiles,
                candidate_quantiles,
                quantile_differences,
                strict=True,
            )
        },
    }


def compare_results(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], case: BinaryParityCase
) -> dict[str, Any]:
    """Compare selected/native parameters, scores, metrics and elapsed time."""
    reference_params = dict(reference["selected_params"])
    candidate_params = dict(candidate["selected_params"])

    def model_differences(
        reference_values: Mapping[str, Any], candidate_values: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return parameter differences grouped by public model key."""
        differences: dict[str, Any] = {}
        for model_name in sorted(set(reference_values) | set(candidate_values)):
            reference_model = reference_values.get(model_name)
            candidate_model = candidate_values.get(model_name)
            if not isinstance(reference_model, Mapping) or not isinstance(
                candidate_model, Mapping
            ):
                if reference_model != candidate_model:
                    differences[model_name] = {
                        "autocampaignxfm": reference_model,
                        "fmlib": candidate_model,
                    }
                continue
            parameter_names = sorted(set(reference_model) | set(candidate_model))
            model_difference = {
                name: {
                    "autocampaignxfm": reference_model.get(name),
                    "fmlib": candidate_model.get(name),
                }
                for name in parameter_names
                if reference_model.get(name) != candidate_model.get(name)
            }
            if model_difference:
                differences[model_name] = model_difference
        return differences

    parameter_differences = model_differences(reference_params, candidate_params)
    reference_resolved = dict(reference["resolved_model_params"])
    candidate_resolved = dict(candidate["resolved_model_params"])
    resolved_parameter_differences = model_differences(
        reference_resolved, candidate_resolved
    )
    score_distribution = _score_distribution_comparison(
        pl.read_parquet(reference["scores_path"]),
        pl.read_parquet(candidate["scores_path"]),
        case,
    )
    calibrated_score_distribution = None
    if reference.get("calibrated_scores_path") and candidate.get(
        "calibrated_scores_path"
    ):
        calibrated_score_distribution = _score_distribution_comparison(
            pl.read_parquet(reference["calibrated_scores_path"]),
            pl.read_parquet(candidate["calibrated_scores_path"]),
            case,
        )
    params_equal = not parameter_differences
    resolved_params_equal = not resolved_parameter_differences
    reference_timing = reference["timing_seconds"]
    candidate_timing = candidate["timing_seconds"]

    def ratio(reference_value: float, candidate_value: float) -> float | None:
        """Return reference-to-candidate elapsed-time ratio when defined."""
        return float(reference_value / candidate_value) if candidate_value else None

    def metric_comparison(
        reference_values: Mapping[str, float], candidate_values: Mapping[str, float]
    ) -> dict[str, Any]:
        """Compare one metric for each product or channel model."""
        rows: dict[str, Any] = {}
        for model_name in sorted(set(reference_values) | set(candidate_values)):
            reference_value = reference_values.get(model_name)
            candidate_value = candidate_values.get(model_name)
            difference = (
                abs(float(reference_value) - float(candidate_value))
                if reference_value is not None and candidate_value is not None
                else None
            )
            rows[model_name] = {
                "autocampaignxfm": reference_value,
                "fmlib": candidate_value,
                "absolute_difference": difference,
            }
        return rows

    validation_metrics = metric_comparison(
        reference["validation_roc_auc"], candidate["validation_roc_auc"]
    )
    test_metrics = metric_comparison(
        reference["test_roc_auc"], candidate["test_roc_auc"]
    )
    calibrated_test_metrics = metric_comparison(
        reference.get("calibrated_test_roc_auc", {}),
        candidate.get("calibrated_test_roc_auc", {}),
    )
    metric_differences = [
        row["absolute_difference"]
        for row in (
            *validation_metrics.values(),
            *test_metrics.values(),
            *calibrated_test_metrics.values(),
        )
        if row["absolute_difference"] is not None
    ]
    metric_keys_equal = (
        set(reference["validation_roc_auc"]) == set(candidate["validation_roc_auc"])
        and set(reference["test_roc_auc"]) == set(candidate["test_roc_auc"])
        and set(reference.get("calibrated_test_roc_auc", {}))
        == set(candidate.get("calibrated_test_roc_auc", {}))
    )
    roc_auc_passed = (
        metric_keys_equal
        and max(metric_differences, default=float("inf")) <= case.max_roc_auc_difference
    )
    reference_features = reference.get("feature_names")
    candidate_features = candidate.get("feature_names")
    feature_names_equal = (
        reference_features == candidate_features
        if reference_features is not None and candidate_features is not None
        else None
    )
    passed = (
        (params_equal or not case.require_equal_best_params)
        and feature_names_equal is not False
        and score_distribution["passed"]
        and (
            calibrated_score_distribution is None
            or calibrated_score_distribution["passed"]
        )
        and roc_auc_passed
    )
    return {
        "passed": passed,
        "best_params_equal": params_equal,
        "best_param_differences": parameter_differences,
        "resolved_model_params_equal": resolved_params_equal,
        "resolved_model_param_differences": resolved_parameter_differences,
        "resolved_model_params": {
            "autocampaignxfm": reference_resolved,
            "fmlib": candidate_resolved,
        },
        "feature_names_equal": feature_names_equal,
        "score_distribution": score_distribution,
        "calibrated_score_distribution": calibrated_score_distribution,
        "metrics": {
            "passed": roc_auc_passed,
            "max_roc_auc_difference": case.max_roc_auc_difference,
            "validation_roc_auc": validation_metrics,
            "test_roc_auc": test_metrics,
            "calibrated_test_roc_auc": calibrated_test_metrics,
        },
        "timing_seconds": {
            "autocampaignxfm": reference_timing,
            "fmlib": candidate_timing,
            "autocampaignxfm_to_fmlib_ratio": {
                "total_train_before_predict": ratio(
                    reference_timing["total_train_before_predict"],
                    candidate_timing["total_train_before_predict"],
                ),
                "predict": ratio(
                    reference_timing["predict"], candidate_timing["predict"]
                ),
            },
        },
    }


def run_reference(
    case: BinaryParityCase, autocampaign_root: Path, autocampaign_python: Path
) -> dict[str, Any]:
    """Run the reference adapter in a subprocess with autocampaignxfm importable."""
    autocampaign_root = autocampaign_root.expanduser().resolve()
    autocampaign_python = autocampaign_python.expanduser().resolve()
    if not (autocampaign_root / "uplift_metalearner").is_dir():
        msg = f"autocampaignxfm package is missing under: {autocampaign_root}"
        raise ConfigError(msg)
    if not autocampaign_python.is_file():
        msg = f"autocampaign Python executable does not exist: {autocampaign_python}"
        raise ConfigError(msg)
    reference_output = case.output_dir / "autocampaignxfm"
    reference_output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    python_paths = [str(autocampaign_root), str(repo_root)]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONIOENCODING"] = "utf-8"
    matplotlib_directory = reference_output / ".matplotlib"
    matplotlib_directory.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(matplotlib_directory)
    subprocess.run(
        [
            str(autocampaign_python),
            "-m",
            "tools.automl_parity.reference",
            "--config",
            str(case.autocampaign_config_path),
            "--fmlib-config",
            str(case.fmlib_config_path),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
    )
    return json.loads((reference_output / "result.json").read_text(encoding="utf-8"))


def run_comparison(
    autocampaign_config_path: str | Path,
    fmlib_config_path: str | Path,
    *,
    autocampaign_root: str | Path,
    autocampaign_python: str | Path = sys.executable,
) -> dict[str, Any]:
    """Execute both pipelines and write the final comparison report."""
    case = load_case(autocampaign_config_path, fmlib_config_path)
    case.output_dir.mkdir(parents=True, exist_ok=True)
    reference = run_reference(case, Path(autocampaign_root), Path(autocampaign_python))
    candidate = run_fmlib(case)
    comparison = compare_results(reference, candidate, case)
    comparison_path = case.output_dir / "comparison.json"
    _write_json(comparison_path, comparison)
    logger.info(
        "Comparison:\n%s",
        json.dumps(_jsonable(comparison), ensure_ascii=False, indent=2),
    )
    logger.info("Comparison report: %s", comparison_path)
    return comparison


def main() -> None:
    """Run parity from terminal arguments and fail on a contract mismatch."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Compare fmlib binary boosting with autocampaignxfm"
    )
    parser.add_argument(
        "--autocampaign-config",
        required=True,
        type=Path,
        help="Native autocampaignxfm response YAML",
    )
    parser.add_argument(
        "--fmlib-config",
        required=True,
        type=Path,
        help="YAML loaded by BinaryTaskConfig.from_yaml",
    )
    parser.add_argument(
        "--autocampaign-root",
        required=True,
        type=Path,
        help="Path to the autocampaignxfm repository",
    )
    parser.add_argument(
        "--autocampaign-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for the autocampaignxfm subprocess",
    )
    args = parser.parse_args()
    comparison = run_comparison(
        args.autocampaign_config,
        args.fmlib_config,
        autocampaign_root=args.autocampaign_root,
        autocampaign_python=args.autocampaign_python,
    )
    if not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
