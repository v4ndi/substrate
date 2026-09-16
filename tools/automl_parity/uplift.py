"""Terminal comparison of fmlib uplift with the real reference trainer/evaluator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import polars as pl

from avatar.automl import UpliftTask, UpliftTaskConfig
from avatar.automl.exceptions import ConfigError
from avatar.automl.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

_LEARNERS = ("s", "t", "x")


def _metrics(target: np.ndarray, treatment: np.ndarray, effect: np.ndarray) -> dict[str, float]:
    return {
        "qini_auc": float(qini_auc_score(target, effect, treatment)),
        "uplift_auc": float(uplift_auc_score(target, effect, treatment)),
        **{f"uplift_at_{k}": float(uplift_at_k(target, effect, treatment, "overall", k / 100)) for k in (10, 20, 50)},
    }


def run_fmlib(config: UpliftTaskConfig, train: Path, valid: Path, test: Path, output: Path) -> dict[str, Any]:
    """Run the public fmlib lifecycle and persist normalized parity scores."""
    task = UpliftTask(replace(config, output_dir=output))
    started = perf_counter()
    training = task.train(train, valid)
    training_seconds = perf_counter() - started
    started = perf_counter()
    prediction = task.predict(test)
    evaluation = task.evaluate(test, prediction)
    evaluation_seconds = perf_counter() - started
    raw_scores = prediction.raw_scores if prediction.raw_scores is not None else prediction.scores
    calibrated_scores = prediction.scores if prediction.raw_scores is not None else None
    scores_path = output / "scores.parquet"
    raw_scores.write_parquet(scores_path)
    calibrated_scores_path = output / "scores_calibrated.parquet"
    if calibrated_scores is not None:
        calibrated_scores.write_parquet(calibrated_scores_path)
    return {
        "pipeline": "fmlib",
        "scores_path": str(scores_path),
        "calibrated_scores_path": str(calibrated_scores_path) if calibrated_scores is not None else None,
        "best_params": dict(training.best_params),
        "metrics": {
            learner: {
                name.removeprefix(f"{learner}_"): value
                for name, value in evaluation.metrics.items()
                if name.startswith(f"{learner}_")
            }
            for learner in _LEARNERS
        },
        "calibrated_metrics": (
            {
                learner: {
                    name.removeprefix(f"{learner}_"): value
                    for name, value in evaluation.calibrated_metrics.items()
                    if name.startswith(f"{learner}_")
                }
                for learner in _LEARNERS
            }
            if evaluation.calibrated_metrics is not None
            else None
        ),
        "timing_seconds": {"train": training_seconds, "evaluate": evaluation_seconds},
    }


def run_reference(
    reference_config: Path,
    fmlib_config: Path,
    *,
    autocampaign_root: Path,
    autocampaign_python: Path,
    output: Path,
) -> dict[str, Any]:
    """Launch the isolated adapter that invokes both real reference entrypoints."""
    environment = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(autocampaign_root), str(repo_root)))
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONIOENCODING"] = "utf-8"
    command = [
        str(autocampaign_python),
        "-m",
        "tools.automl_parity.uplift_reference",
        "--reference-config",
        str(reference_config),
        "--fmlib-config",
        str(fmlib_config),
        "--autocampaign-root",
        str(autocampaign_root),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=repo_root, env=environment, check=True)  # noqa: S603
    return json.loads((output / "result.json").read_text(encoding="utf-8"))


def _distribution(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    quantiles = np.linspace(0.05, 0.95, 19)
    reference_quantiles = np.quantile(reference, quantiles)
    candidate_quantiles = np.quantile(candidate, quantiles)
    return {
        "reference_mean": float(reference.mean()),
        "fmlib_mean": float(candidate.mean()),
        "reference_std": float(reference.std()),
        "fmlib_std": float(candidate.std()),
        "max_quantile_difference": float(np.max(np.abs(reference_quantiles - candidate_quantiles))),
    }


def compare(reference: Mapping[str, Any], candidate: Mapping[str, Any], config: UpliftTaskConfig) -> dict[str, Any]:
    """Compare metrics, score distributions, parameters and elapsed time."""
    join_columns = [config.client_id_column, config.report_month_column]

    def distributions_for(reference_path: str, candidate_path: str) -> dict[str, Any]:
        reference_scores = pl.read_parquet(reference_path).with_columns(pl.col(config.report_month_column).cast(pl.String))
        candidate_scores = pl.read_parquet(candidate_path).with_columns(pl.col(config.report_month_column).cast(pl.String))
        joined = reference_scores.join(candidate_scores, on=join_columns, suffix="_fmlib", validate="1:1")
        if joined.height != reference_scores.height or joined.height != candidate_scores.height:
            msg = "Reference and fmlib score populations do not match by client/month"
            raise AssertionError(msg)
        return {
            learner: _distribution(
                joined[f"score_{learner}"].to_numpy(),
                joined[f"score_{learner}_fmlib"].to_numpy(),
            )
            for learner in _LEARNERS
        }

    def metric_differences_for(reference_metrics, candidate_metrics):
        return {
            learner: {
                name: abs(float(value) - float(candidate_metrics[learner][name]))
                for name, value in reference_metrics[learner].items()
                if name in candidate_metrics[learner]
            }
            for learner in _LEARNERS
        }

    distributions = distributions_for(reference["scores_path"], candidate["scores_path"])
    metric_differences = metric_differences_for(reference["metrics"], candidate["metrics"])
    calibrated_distributions = None
    calibrated_metric_differences = None
    if reference.get("calibrated_scores_path") and candidate.get("calibrated_scores_path"):
        calibrated_distributions = distributions_for(reference["calibrated_scores_path"], candidate["calibrated_scores_path"])
        calibrated_metric_differences = metric_differences_for(reference["calibrated_metrics"], candidate["calibrated_metrics"])
    return {
        "reference_uses_real_trainer_evaluator": True,
        "best_params": {"reference": reference["best_params"], "fmlib": candidate["best_params"]},
        "metric_absolute_differences": metric_differences,
        "calibrated_metric_absolute_differences": calibrated_metric_differences,
        "score_distributions": distributions,
        "calibrated_score_distributions": calibrated_distributions,
        "timing_seconds": {"reference": reference["timing_seconds"], "fmlib": candidate["timing_seconds"]},
        "categorical_note": (
            "Small differences are expected when fmlib uses native CatBoost categorical features while the "
            "reference preprocessing supplies a legacy numeric matrix."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True, type=Path)
    parser.add_argument("--fmlib-config", required=True, type=Path)
    parser.add_argument("--autocampaign-root", required=True, type=Path)
    parser.add_argument("--autocampaign-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    config = UpliftTaskConfig.from_yaml(args.fmlib_config)
    if config.n_trials != 3:
        msg = "Uplift terminal parity requires n_trials=3"
        raise ConfigError(msg)
    reference_payload = (
        json.loads(args.reference_config.read_text(encoding="utf-8")) if args.reference_config.suffix == ".json" else None
    )
    if reference_payload and "data" in reference_payload:
        data = reference_payload["data"]["input_dir"]
    else:
        from omegaconf import OmegaConf

        if not OmegaConf.has_resolver("date"):
            OmegaConf.register_new_resolver("date", date.fromisoformat)
        data = OmegaConf.to_container(OmegaConf.load(args.reference_config), resolve=True)["data"]["input_dir"]
    output = Path(config.output_dir).expanduser().resolve() / "parity"
    output.mkdir(parents=True, exist_ok=True)
    reference = run_reference(
        args.reference_config.resolve(),
        args.fmlib_config.resolve(),
        autocampaign_root=args.autocampaign_root.resolve(),
        autocampaign_python=args.autocampaign_python.resolve(),
        output=output / "reference",
    )
    candidate = run_fmlib(config, Path(data["train"]), Path(data["valid"]), Path(data["test"]), output / "fmlib")
    report = compare(reference, candidate, config)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
