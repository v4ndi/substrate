"""Run the unmodified autocampaignxfm binary training and evaluation entrypoints."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from tools.automl_parity.autocampaign_entrypoint import _decode_name, _encode_path
from tools.automl_parity.binary import BinaryParityCase, _jsonable, load_case


def _run_entrypoint(
    script: Path, config_path: Path, *, cwd: Path, environment: dict[str, str]
) -> float:
    """Run one Hydra CLI with the supplied native autocampaignxfm YAML."""
    started = perf_counter()
    command = [
        sys.executable,
        "-m",
        "tools.automl_parity.autocampaign_entrypoint",
        str(script),
    ]
    subprocess.run(
        [
            *command,
            "--config-path",
            config_path.parent.as_posix(),
            "--config-name",
            config_path.stem,
        ],
        cwd=cwd,
        env=environment,
        check=True,
    )
    return perf_counter() - started


def _single_path(paths: list[Path], description: str) -> Path:
    """Return the only matching artifact and fail on missing or ambiguous output."""
    if len(paths) != 1:
        msg = f"Expected one {description}, found {len(paths)}: {paths}"
        raise RuntimeError(msg)
    return paths[0]


def _parquet_files(source: Path) -> list[Path]:
    """List parquet inputs for either a single file or a partitioned directory."""
    source_files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    if not source_files:
        msg = f"No parquet files found below {source}"
        raise FileNotFoundError(msg)
    return source_files


def _fit_label_encoders(
    source: Path, columns: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    """Build deterministic train-only integer mappings for string categorical columns."""
    values: dict[str, set[str]] = {column: set() for column in columns}
    string_types = (pl.String, pl.Categorical, pl.Enum)
    for source_file in _parquet_files(source):
        frame = pl.read_parquet(source_file, hive_partitioning=False)
        for column in columns:
            if column in frame.columns and isinstance(
                frame.schema[column], string_types
            ):
                values[column].update(
                    frame[column].cast(pl.String).drop_nulls().unique().to_list()
                )
    return {
        column: {value: index for index, value in enumerate(sorted(column_values))}
        for column, column_values in values.items()
        if column_values
    }


def _write_compatible_split(
    source: Path,
    destination: Path,
    hidden_state_columns: tuple[str, ...],
    label_encoders: dict[str, dict[str, int]],
    column_renames: dict[str, str],
) -> None:
    """Copy shards while adapting embeddings and string categories for autocampaignxfm."""
    source_files = _parquet_files(source)
    for source_file in source_files:
        relative_path = (
            Path(source_file.name)
            if source.is_file()
            else source_file.relative_to(source)
        )
        destination_file = destination / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.read_parquet(source_file, hive_partitioning=False)
        array_columns = [
            column
            for column in hidden_state_columns
            if column in frame.columns and isinstance(frame.schema[column], pl.Array)
        ]
        if array_columns:
            frame = frame.with_columns(
                pl.col(column).cast(pl.List(pl.Float32)).alias(column)
                for column in array_columns
            )
        encoded_columns = [
            column for column in label_encoders if column in frame.columns
        ]
        if encoded_columns:
            frame = frame.with_columns(
                pl.col(column)
                .cast(pl.String)
                .replace_strict(
                    label_encoders[column], default=-1, return_dtype=pl.Int32
                )
                .alias(column)
                for column in encoded_columns
            )
        applicable_renames = {
            source: target
            for source, target in column_renames.items()
            if source in frame.columns
        }
        if applicable_renames:
            collisions = (
                set(applicable_renames.values())
                .intersection(frame.columns)
                .difference(applicable_renames)
            )
            if collisions:
                msg = f"Cannot prepare product-only parity data; rename targets already exist: {sorted(collisions)}"
                raise RuntimeError(msg)
            frame = frame.rename(applicable_renames)
        frame.write_parquet(destination_file)


def _write_runtime_config(case: BinaryParityCase, directory: Path) -> Path:
    """Create an autocampaign-only YAML pointing to List-compatible parquet copies."""
    compatible_root = directory / "data"
    categorical_columns = tuple(
        dict.fromkeys((
            *case.categorical_columns,
            *(
                (case.group_column,)
                if case.model_scope == "product" and case.group_column is not None
                else ()
            ),
            *((case.treatment_column,) if case.treatment_column is not None else ()),
        ))
    )
    label_encoders = _fit_label_encoders(case.train_path, categorical_columns)
    # The reference trainer has no product-only flag. For product parity a missing
    # configured group skips channel studies, while the copied group remains a
    # product feature under the evaluator's fallback name. Channel parity keeps
    # the configured group and lets the native trainer create per-channel studies.
    product_group_column = "target_attr_2"
    column_renames = (
        {case.group_column: product_group_column}
        if case.model_scope == "product"
        and case.group_column is not None
        and case.group_column != product_group_column
        else {}
    )
    split_paths: dict[str, str] = {}
    for split_name, source in (
        ("train", case.train_path),
        ("valid", case.valid_path),
        ("test", case.test_path),
    ):
        destination = compatible_root / split_name
        _write_compatible_split(
            source,
            destination,
            case.hidden_state_columns,
            label_encoders,
            column_renames,
        )
        split_paths[split_name] = str(destination.resolve())

    config = OmegaConf.load(case.autocampaign_config_path)
    config.data.input_dir = split_paths
    if case.model_scope == "product":
        config.data.group_column = "__product_only_group_column__"
        config.evaluate.is_product = True
        config.evaluate.model_configs_dir = (
            f"{config.train.output_dir}/configs_product/"
        )
    else:
        config.data.group_column = case.group_column
        config.evaluate.is_product = False
        config.evaluate.model_configs_dir = str(
            (directory / "evaluation_configs_channels").resolve()
        )
    runtime_config = directory / "autocampaignxfm_runtime.yaml"
    OmegaConf.save(config, runtime_config)
    return runtime_config


def _prepare_channel_evaluation_configs(source: Path, destination: Path) -> None:
    """Copy channel params under names accepted by the reference evaluator parser.

    The trainer inserts an underscore between ``'>`` and the channel value,
    while the evaluator treats everything after ``'>`` as the value itself.
    The parity adapter removes only that separator in its temporary copies.
    """
    destination.mkdir(parents=True, exist_ok=True)
    params_paths = sorted(source.glob("*.txt"))
    if not params_paths:
        msg = f"No channel best-parameter files found in {source}"
        raise FileNotFoundError(msg)
    for params_path in params_paths:
        logical_name = _decode_name(params_path.name).replace("'>_", "'>")
        destination_name = _encode_path(logical_name)
        shutil.copy2(params_path, destination / destination_name)


def _channel_from_params_path(path: Path) -> str:
    """Extract the complete channel value from a native trainer filename."""
    stem = Path(_decode_name(path.name)).stem
    if "'>_" in stem:
        return stem.split("'>_", 1)[1]
    msg = f"Cannot extract channel from autocampaign parameter filename: {path.name}"
    raise ValueError(msg)


def _write_channel_training_configs(
    case: BinaryParityCase, runtime_config: Path, directory: Path
) -> list[Path]:
    """Create one native trainer config and filtered parquet set per channel.

    The reference trainer exits cleanly after a single-channel study and therefore
    does not continue into its unconditional product study. Evaluation still
    uses the full compatible data referenced by ``runtime_config``.
    """
    base_config = OmegaConf.load(runtime_config)
    group_column = case.group_column
    if group_column is None:
        msg = "Channel parity requires group_column"
        raise ValueError(msg)
    full_paths = {
        name: Path(base_config.data.input_dir[name])
        for name in ("train", "valid", "test")
    }
    train = pl.concat(
        [
            pl.read_parquet(path, hive_partitioning=False)
            for path in _parquet_files(full_paths["train"])
        ],
        how="diagonal_relaxed",
    )
    group_values = sorted(train[group_column].unique().to_list(), key=str)
    configs: list[Path] = []
    for index, group_value in enumerate(group_values):
        split_paths: dict[str, str] = {}
        for split_name, source in full_paths.items():
            destination = directory / "channel_training_data" / str(index) / split_name
            destination.mkdir(parents=True, exist_ok=True)
            row_count = 0
            for shard_index, source_file in enumerate(_parquet_files(source)):
                frame = pl.read_parquet(source_file, hive_partitioning=False).filter(
                    pl.col(group_column) == group_value
                )
                if frame.is_empty():
                    continue
                row_count += frame.height
                frame.write_parquet(destination / f"part-{shard_index:05d}.parquet")
            if row_count == 0:
                msg = f"Split {split_name!r} has no rows for channel {group_value!r}"
                raise RuntimeError(msg)
            split_paths[split_name] = str(destination.resolve())
        config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
        config.data.input_dir = split_paths
        config_path = directory / f"autocampaignxfm_train_channel_{index}.yaml"
        OmegaConf.save(config, config_path)
        configs.append(config_path)
    return configs


def _read_scores(
    predict_dir: Path, case: BinaryParityCase
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read evaluator parquet shards and return validation and test score tables."""
    shards = sorted(predict_dir.glob("*.parquet"))
    if not shards:
        msg = f"Autocampaignxfm evaluator wrote no score shards to {predict_dir}"
        raise FileNotFoundError(msg)
    scores = pl.concat(
        [pl.read_parquet(path) for path in shards], how="diagonal_relaxed"
    )
    required = {case.client_id_column, case.target_column, "prediction", "split_type"}
    missing = required.difference(scores.columns)
    if missing:
        msg = f"Autocampaignxfm scores are missing required columns: {sorted(missing)}"
        raise RuntimeError(msg)
    validation = scores.filter(pl.col("split_type") == "calib")
    test = scores.filter(pl.col("split_type") == "test")
    if validation.is_empty() or test.is_empty():
        msg = "Autocampaignxfm scores must contain both split_type='calib' and split_type='test'"
        raise RuntimeError(msg)
    return validation, test


def _normalized_scores(
    frame: pl.DataFrame,
    case: BinaryParityCase,
    score_column: str = "prediction",
) -> pl.DataFrame:
    """Convert evaluator score columns to the parity runner's public score schema."""
    columns = [case.client_id_column]
    if case.report_month_column in frame.columns:
        columns.append(case.report_month_column)
    return frame.select(*columns, pl.col(score_column).alias("score"))


def _read_calibrated_scores(root: Path, case: BinaryParityCase) -> pl.DataFrame:
    """Read OOT rows written by the reference rolling calibrator."""
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        msg = f"Autocampaignxfm produced no calibrated score shards in {root}"
        raise FileNotFoundError(msg)
    scores = pl.concat(
        [pl.read_parquet(path) for path in shards], how="diagonal_relaxed"
    )
    required = {
        case.client_id_column,
        case.report_month_column,
        case.target_column,
        "prediction_cal",
    }
    missing = required.difference(scores.columns)
    if missing:
        msg = (
            f"Autocampaignxfm calibrated scores are missing columns: {sorted(missing)}"
        )
        raise RuntimeError(msg)
    return scores


def run_reference(case: BinaryParityCase) -> dict[str, Any]:
    """Run trainer_booster and evaluater_booster and normalize their real artifacts."""
    if case.engine not in {"catboost", "xgboost"}:
        msg = f"Binary parity does not support engine={case.engine!r}"
        raise ValueError(msg)

    import uplift_metalearner
    from uplift_metalearner.metalearners.supervised_pipeline import (
        instantiate_supervised_learner_from_params,
    )

    autocampaign_root = Path(uplift_metalearner.__file__).resolve().parent.parent
    config = OmegaConf.load(case.autocampaign_config_path)
    resolved_config = OmegaConf.to_container(config, resolve=True)
    train_output = Path(resolved_config["train"]["output_dir"]).resolve()
    evaluate_output = Path(resolved_config["evaluate"]["output_dir"]).resolve()
    artifact_directory = (
        "configs_product" if case.model_scope == "product" else "configs_channels"
    )
    metrics_output = evaluate_output / (
        "metrics_product" if case.model_scope == "product" else "metrics_channels"
    )

    # Both entrypoints append artifacts. A parity run must not consume shards or
    # parameter files left by an earlier execution of the same case.
    for path in (train_output, metrics_output):
        if path.exists():
            shutil.rmtree(path)

    environment = os.environ.copy()
    environment["HYDRA_FULL_ERROR"] = "1"
    matplotlib_dir = case.output_dir / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(matplotlib_dir)

    with tempfile.TemporaryDirectory(
        prefix="autocampaign_parity_", dir=case.output_dir
    ) as temporary:
        runtime_config = _write_runtime_config(case, Path(temporary))
        training_configs = (
            _write_channel_training_configs(case, runtime_config, Path(temporary))
            if case.model_scope == "group"
            else [runtime_config]
        )
        train_seconds = sum(
            _run_entrypoint(
                autocampaign_root / "trainer_booster.py",
                training_config,
                cwd=autocampaign_root,
                environment=environment,
            )
            for training_config in training_configs
        )
        if case.model_scope == "group":
            runtime = OmegaConf.load(runtime_config)
            _prepare_channel_evaluation_configs(
                train_output / "configs_channels",
                Path(runtime.evaluate.model_configs_dir),
            )
        evaluate_seconds = _run_entrypoint(
            autocampaign_root / "evaluater_booster.py",
            runtime_config,
            cwd=autocampaign_root,
            environment=environment,
        )

    params_paths = sorted((train_output / artifact_directory).glob("*.txt"))
    if not params_paths:
        msg = f"No best-parameter files found in {train_output / artifact_directory}"
        raise FileNotFoundError(msg)
    group_encoders = (
        _fit_label_encoders(case.train_path, (case.group_column,))
        if case.group_column
        else {}
    )
    inverse_groups = {
        str(value): key
        for key, value in group_encoders.get(case.group_column or "", {}).items()
    }
    selected_params: dict[str, Any] = {}
    resolved_model_params: dict[str, Any] = {}
    params_path_payload: dict[str, str] = {}
    for params_path in params_paths:
        channel = _channel_from_params_path(params_path)
        model_name = (
            "product"
            if case.model_scope == "product"
            else f"group:{inverse_groups.get(channel, channel)}"
        )
        params = ast.literal_eval(
            params_path.read_text(encoding="utf-8").splitlines()[0]
        )
        selected_params[model_name] = params
        resolved_model_params[model_name] = instantiate_supervised_learner_from_params(
            case.engine, "binary_clf", params
        ).get_params()
        params_path_payload[model_name] = str(params_path)

    validation, test = _read_scores(metrics_output / "predict", case)
    calibrated_test = (
        _read_calibrated_scores(metrics_output / "predict_calibrated", case)
        if case.fmlib_config.calibration_windows is not None
        else None
    )
    validation_auc: dict[str, float] = {}
    test_auc: dict[str, float] = {}
    calibrated_test_auc: dict[str, float] = {}
    if case.model_scope == "product":
        validation_auc["product"] = float(
            roc_auc_score(
                validation[case.target_column].to_numpy(),
                validation["prediction"].to_numpy(),
            )
        )
        test_auc["product"] = float(
            roc_auc_score(
                test[case.target_column].to_numpy(), test["prediction"].to_numpy()
            )
        )
        if calibrated_test is not None:
            calibrated_test_auc["product"] = float(
                roc_auc_score(
                    calibrated_test[case.target_column].to_numpy(),
                    calibrated_test["prediction_cal"].to_numpy(),
                )
            )
    else:
        score_group_column = case.group_column
        for encoded_value in validation[score_group_column].unique().to_list():
            group_value = inverse_groups.get(str(encoded_value), str(encoded_value))
            model_name = f"group:{group_value}"
            validation_group = validation.filter(
                pl.col(score_group_column) == encoded_value
            )
            test_group = test.filter(pl.col(score_group_column) == encoded_value)
            validation_auc[model_name] = float(
                roc_auc_score(
                    validation_group[case.target_column].to_numpy(),
                    validation_group["prediction"].to_numpy(),
                )
            )
            test_auc[model_name] = float(
                roc_auc_score(
                    test_group[case.target_column].to_numpy(),
                    test_group["prediction"].to_numpy(),
                )
            )
            if calibrated_test is not None:
                calibrated_group = calibrated_test.filter(
                    pl.col(score_group_column) == encoded_value
                )
                calibrated_test_auc[model_name] = float(
                    roc_auc_score(
                        calibrated_group[case.target_column].to_numpy(),
                        calibrated_group["prediction_cal"].to_numpy(),
                    )
                )
    output_dir = case.output_dir / "autocampaignxfm"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.parquet"
    _normalized_scores(test, case).write_parquet(scores_path)
    calibrated_scores_path = output_dir / "scores_calibrated.parquet"
    if calibrated_test is not None:
        _normalized_scores(calibrated_test, case, "prediction_cal").write_parquet(
            calibrated_scores_path
        )
    result = {
        "pipeline": "autocampaignxfm",
        "selected_params": selected_params,
        "resolved_model_params": resolved_model_params,
        "validation_roc_auc": validation_auc,
        "test_roc_auc": test_auc,
        "calibrated_test_roc_auc": calibrated_test_auc,
        "timing_seconds": {
            "hyperparameter_search_and_train": train_seconds,
            "inference_model_fit": evaluate_seconds,
            "predict": 0.0,
            "total_train_before_predict": train_seconds + evaluate_seconds,
            "full_evaluation": evaluate_seconds,
        },
        "params_txt_path": params_path_payload,
        "raw_scores_path": str(metrics_output / "predict"),
        "scores_path": str(scores_path),
        "calibrated_scores_path": str(calibrated_scores_path)
        if calibrated_test is not None
        else None,
    }
    (output_dir / "result.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Run the real autocampaignxfm reference pipeline from its native YAML."""
    parser = argparse.ArgumentParser(description="Run autocampaignxfm binary reference")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fmlib-config", required=True, type=Path)
    args = parser.parse_args()
    run_reference(load_case(args.config, args.fmlib_config))


if __name__ == "__main__":
    main()
