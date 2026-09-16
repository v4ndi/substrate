"""Isolated adapter for the real autocampaignxfm uplift CLI entrypoints."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
from omegaconf import OmegaConf

from avatar.automl import UpliftTaskConfig
from avatar.automl.data import ParquetSource
from tools.automl_parity.reference import _fit_label_encoders, _write_compatible_split
from tools.automl_parity.uplift import _LEARNERS, _metrics


def _entrypoint(script: Path, config: Path, root: Path) -> float:
    started = perf_counter()
    command = [
        sys.executable,
        "-m",
        "tools.automl_parity.autocampaign_entrypoint",
        str(script),
        "--config-path",
        config.parent.as_posix(),
        "--config-name",
        config.stem,
    ]
    subprocess.run(command, cwd=root, check=True)  # noqa: S603
    return perf_counter() - started


def _learner_name(value: str) -> str:
    lowered = value.lower()
    for learner in _LEARNERS:
        if f"base{learner}" in lowered or f"{learner}learner" in lowered:
            return learner
    msg = f"Unknown reference meta-learner label: {value!r}"
    raise ValueError(msg)


def _client_scores(root: Path, config: UpliftTaskConfig, *, calibrated: bool = False) -> pl.DataFrame:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        msg = f"Reference evaluator produced no parquet scores below {root}"
        raise FileNotFoundError(msg)
    frame = pl.concat([pl.read_parquet(path) for path in files], how="diagonal_relaxed")
    if "split_type" in frame.columns:
        frame = frame.filter(pl.col("split_type") == "test")
    learner_column = "metalearner"
    suffix = "_cal" if calibrated else ""
    treatment_column = f"treatment_probs{suffix}"
    control_column = f"control_probs{suffix}"
    required = {
        learner_column,
        treatment_column,
        control_column,
        config.client_id_column,
        config.report_month_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        msg = f"Reference client scores are missing columns: {missing}"
        raise ValueError(msg)
    frame = frame.with_columns(
        pl.col(learner_column).map_elements(_learner_name, return_dtype=pl.String).alias("learner"),
        (pl.col(treatment_column) - pl.col(control_column)).alias("effect"),
    )
    keys = [config.client_id_column, config.report_month_column]
    wide = frame.pivot(on="learner", index=keys, values="effect", aggregate_function="first")
    return wide.rename({learner: f"score_{learner}" for learner in _LEARNERS})


def _best_params(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(root.rglob("*.txt")):
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        try:
            result[path.stem] = ast.literal_eval(first_line)
        except (SyntaxError, ValueError):
            continue
    return result


def _joined_metrics(frame: pl.DataFrame, config: UpliftTaskConfig) -> dict[str, dict[str, float]]:
    """Calculate learner metrics from one key-aligned truth-and-score frame."""
    target = frame[config.target_column].to_numpy()
    treatment = frame[config.treatment_column].to_numpy()
    if config.inverse_treatment:
        treatment = 1 - treatment
    return {
        learner: _metrics(target, treatment, frame[f"score_{learner}"].to_numpy())
        for learner in _LEARNERS
    }


def _composed_config(path: Path):
    """Compose a native Hydra config, including its system/gbdt defaults."""
    import hydra

    if not OmegaConf.has_resolver("date"):
        OmegaConf.register_new_resolver("date", date.fromisoformat)
    with hydra.initialize_config_dir(config_dir=str(path.parent.resolve()), version_base=None):
        return hydra.compose(config_name=path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True, type=Path)
    parser.add_argument("--fmlib-config", required=True, type=Path)
    parser.add_argument("--autocampaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = UpliftTaskConfig.from_yaml(args.fmlib_config)
    runtime = _composed_config(args.reference_config)
    compatible_root = args.output / "compatible_data"
    if compatible_root.exists():
        shutil.rmtree(compatible_root)
    source_paths = {
        split: Path(runtime.data.input_dir[split]).expanduser().resolve()
        for split in ("train", "valid", "test")
    }
    categorical_columns = tuple(
        dict.fromkeys((*config.categorical_columns, *((config.group_column,) if config.group_column else ())))
    )
    label_encoders = _fit_label_encoders(source_paths["train"], categorical_columns)
    compatible_paths: dict[str, str] = {}
    for split, source in source_paths.items():
        destination = compatible_root / split
        _write_compatible_split(
            source,
            destination,
            tuple(config.hidden_state_columns),
            label_encoders,
            {},
        )
        compatible_paths[split] = str(destination.resolve())
    runtime.data.input_dir = compatible_paths
    runtime.train.trials = 3
    runtime.train.random_state = 42
    runtime.train.output_dir = str((args.output / "trained").resolve())
    scope = "product" if bool(runtime.evaluate.is_product) else "channels"
    runtime.evaluate.model_configs_dir = str((args.output / "trained" / f"configs_{scope}").resolve())
    runtime.evaluate.output_dir = str((args.output / "evaluated").resolve())
    runtime_path = args.output / "runtime.yaml"
    OmegaConf.save(runtime, runtime_path)
    os.environ["AUTOCAMPAIGNXFM_DEFAULT_CONF"] = str((args.autocampaign_root / "conf").resolve())
    train_seconds = _entrypoint(args.autocampaign_root / "trainer_booster.py", runtime_path, args.autocampaign_root)
    evaluate_seconds = _entrypoint(args.autocampaign_root / "evaluater_booster.py", runtime_path, args.autocampaign_root)
    metrics_scope = "metrics_product" if bool(runtime.evaluate.is_product) else "metrics_channels"
    scores = _client_scores(args.output / "evaluated" / metrics_scope / "predict", config)
    scores_path = args.output / "scores.parquet"
    scores.write_parquet(scores_path)
    calibrated_scores = (
        _client_scores(
            args.output / "evaluated" / metrics_scope / "predict_calibrated",
            config,
            calibrated=True,
        )
        if config.calibration_windows is not None
        else None
    )
    calibrated_scores_path = args.output / "scores_calibrated.parquet"
    if calibrated_scores is not None:
        calibrated_scores.write_parquet(calibrated_scores_path)
    truth = ParquetSource.resolve(runtime.data.input_dir.test).read()
    truth = truth.join(scores, on=[config.client_id_column, config.report_month_column], validate="1:1")
    calibrated_truth = (
        truth.drop([f"score_{learner}" for learner in _LEARNERS])
        .with_columns(pl.col(config.report_month_column).cast(pl.String))
        .join(
            calibrated_scores.with_columns(pl.col(config.report_month_column).cast(pl.String)),
            on=[config.client_id_column, config.report_month_column],
            validate="1:1",
        )
        if calibrated_scores is not None
        else None
    )
    result = {
        "pipeline": "autocampaignxfm",
        "entrypoints": ["trainer_booster.py", "evaluater_booster.py"],
        "scores_path": str(scores_path),
        "calibrated_scores_path": str(calibrated_scores_path) if calibrated_scores is not None else None,
        "best_params": _best_params(args.output / "trained"),
        "metrics": _joined_metrics(truth, config),
        "calibrated_metrics": _joined_metrics(calibrated_truth, config) if calibrated_truth is not None else None,
        "timing_seconds": {"train": train_seconds, "evaluate": evaluate_seconds},
    }
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
