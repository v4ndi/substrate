"""Run real autocampaignxfm trainer/evaluator entrypoints for regression parity."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from tools.automl_parity.reference import (
    _channel_from_params_path,
    _fit_label_encoders,
    _normalized_scores,
    _prepare_channel_evaluation_configs,
    _read_calibrated_scores,
    _read_scores,
    _run_entrypoint,
    _write_channel_training_configs,
    _write_runtime_config,
)
from tools.automl_parity.regression import _metrics, load_case


def run_reference(case) -> dict:
    """Execute the full reference search and evaluator and normalize their outputs."""
    import uplift_metalearner

    root = Path(uplift_metalearner.__file__).resolve().parent.parent
    config = OmegaConf.to_container(OmegaConf.load(case.autocampaign_config_path), resolve=True)
    train_output = Path(config["train"]["output_dir"]).resolve()
    evaluate_output = Path(config["evaluate"]["output_dir"]).resolve()
    artifact_directory = "configs_product" if case.model_scope == "product" else "configs_channels"
    metrics_output = evaluate_output / ("metrics_product" if case.model_scope == "product" else "metrics_channels")
    for path in (train_output, metrics_output):
        if path.exists():
            shutil.rmtree(path)
    environment = os.environ.copy()
    environment["HYDRA_FULL_ERROR"] = "1"
    output = case.output_dir / "autocampaignxfm"
    output.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(output / ".matplotlib")
    Path(environment["MPLCONFIGDIR"]).mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autocampaign_regression_", dir=case.output_dir) as temporary:
        directory = Path(temporary)
        runtime = _write_runtime_config(case, directory)
        training_configs = (
            _write_channel_training_configs(case, runtime, directory)
            if case.model_scope == "group"
            else [runtime]
        )
        train_seconds = sum(
            _run_entrypoint(root / "trainer_booster.py", item, cwd=root, environment=environment)
            for item in training_configs
        )
        if case.model_scope == "group":
            runtime_config = OmegaConf.load(runtime)
            _prepare_channel_evaluation_configs(
                train_output / "configs_channels",
                Path(runtime_config.evaluate.model_configs_dir),
            )
        evaluate_seconds = _run_entrypoint(root / "evaluater_booster.py", runtime, cwd=root, environment=environment)

    params_paths = sorted((train_output / artifact_directory).glob("*.txt"))
    group_encoders = _fit_label_encoders(case.train_path, (case.group_column,)) if case.group_column else {}
    inverse_groups = {str(value): key for key, value in group_encoders.get(case.group_column or "", {}).items()}
    selected_params = {}
    for path in params_paths:
        channel = _channel_from_params_path(path)
        name = "product" if case.model_scope == "product" else f"group:{inverse_groups.get(channel, channel)}"
        selected_params[name] = ast.literal_eval(path.read_text(encoding="utf-8").splitlines()[0])
    _, test = _read_scores(metrics_output / "predict", case)
    scores = _normalized_scores(test, case)
    scores_path = output / "scores.parquet"
    scores.write_parquet(scores_path)
    calibrated = (
        _read_calibrated_scores(metrics_output / "predict_calibrated", case)
        if case.fmlib_config.calibration_windows is not None
        else None
    )
    calibrated_scores_path = output / "scores_calibrated.parquet"
    if calibrated is not None:
        _normalized_scores(calibrated, case, "prediction_cal").write_parquet(calibrated_scores_path)
    result = {
        "pipeline": "autocampaignxfm",
        "selected_params": selected_params,
        "test_metrics": _metrics(test[case.target_column].to_numpy(), test["prediction"].to_numpy()),
        "calibrated_test_metrics": (
            _metrics(calibrated[case.target_column].to_numpy(), calibrated["prediction_cal"].to_numpy())
            if calibrated is not None
            else None
        ),
        "timing_seconds": {
            "train": train_seconds,
            "evaluate_refit": evaluate_seconds,
            "total": train_seconds + evaluate_seconds,
        },
        "scores_path": str(scores_path),
        "calibrated_scores_path": str(calibrated_scores_path) if calibrated is not None else None,
        "raw_scores_path": str(metrics_output / "predict"),
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fmlib-config", required=True, type=Path)
    args = parser.parse_args()
    run_reference(load_case(args.config, args.fmlib_config))


if __name__ == "__main__":
    main()
