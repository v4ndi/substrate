import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

from avatar.automl.exceptions import ConfigError
from tools.automl_parity.binary import compare_results, load_case, run_comparison, run_reference, task_config
from tools.automl_parity.reference import (
    _channel_from_params_path,
    _normalized_scores,
    _prepare_channel_evaluation_configs,
    _read_scores,
    _write_channel_training_configs,
    _write_runtime_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
AUTOCAMPAIGN_CONFIG_PATH = REPO_ROOT / "examples" / "automl" / "configs" / "autocampaignxfm_binary.yaml"
FMLIB_CONFIG_PATH = REPO_ROOT / "examples" / "automl" / "configs" / "fmlib_binary.yaml"
XGBOOST_AUTOCAMPAIGN_CONFIG_PATH = (
    REPO_ROOT / "examples" / "automl" / "tests" / "configs" / "parity_xgboost_product_autocampaignxfm.yaml"
)
XGBOOST_FMLIB_CONFIG_PATH = REPO_ROOT / "examples" / "automl" / "tests" / "configs" / "parity_xgboost_product_fmlib.yaml"
XGBOOST_CHANNEL_AUTOCAMPAIGN_CONFIG_PATH = (
    REPO_ROOT / "examples" / "automl" / "tests" / "configs" / "parity_xgboost_channel_autocampaignxfm.yaml"
)
XGBOOST_CHANNEL_FMLIB_CONFIG_PATH = (
    REPO_ROOT / "examples" / "automl" / "tests" / "configs" / "parity_xgboost_channel_fmlib.yaml"
)


@pytest.fixture(autouse=True)
def _server_workspace_root(monkeypatch):
    monkeypatch.setenv("SBER_INFRA_ROOT", str(WORKSPACE_ROOT))


def test_notebook_equivalent_yaml_resolves_expected_binary_settings():
    case = load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH)
    config = task_config(case, case.output_dir / "fmlib")
    autocampaign_config = OmegaConf.load(AUTOCAMPAIGN_CONFIG_PATH)

    assert case.engine == "catboost"
    assert case.n_trials == 3
    assert config.device == "gpu"
    assert config.hyperopt is True
    assert config.verbose is True
    assert config.target_column == "target"
    assert config.group_column == "group"
    assert config.treatment_column == "treatment"
    assert config.inverse_treatment is True
    assert config.hidden_state_columns == ("seq_hidden_state",)
    assert config.categorical_columns == tuple(f"cat_feature_{index}" for index in range(1, 6))
    assert config.numerical_columns == tuple(f"num_feature_{index}" for index in range(1, 6))
    assert autocampaign_config.evaluate.is_product is True
    assert str(autocampaign_config.evaluate.model_configs_dir).endswith("/configs_product/")


def test_xgboost_parity_yaml_uses_shortened_native_search_space():
    case = load_case(XGBOOST_AUTOCAMPAIGN_CONFIG_PATH, XGBOOST_FMLIB_CONFIG_PATH)
    config = task_config(case, case.output_dir / "fmlib")
    autocampaign_config = OmegaConf.load(XGBOOST_AUTOCAMPAIGN_CONFIG_PATH)

    assert case.engine == "xgboost"
    assert case.n_trials == 3
    assert config.device == "gpu"
    assert config.search_space == {
        "n_estimators": {"type": "int", "low": 12, "high": 12},
        "max_depth": {"type": "int", "low": 2, "high": 2},
        "lambda": {"type": "float", "low": 1.0, "high": 1.0},
        "alpha": {"type": "float", "low": 0.0, "high": 0.0},
        "subsample": {"type": "float", "low": 1.0, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 1.0, "high": 1.0},
        "eta": {"type": "float", "low": 0.1, "high": 0.1},
    }
    assert OmegaConf.to_container(autocampaign_config.optuna_ranges.xgboost, resolve=True) == {
        "n_estimators": {"low": 12, "high": 12, "step": 1},
        "max_depth": {"low": 2, "high": 2},
        "lambda": {"low": 1.0, "high": 1.0, "log": False},
        "alpha": {"low": 0.0, "high": 0.0, "log": False},
        "subsample": {"low": 1.0, "high": 1.0},
        "colsample_bytree": {"low": 1.0, "high": 1.0},
        "eta": {"low": 0.1, "high": 0.1, "log": False},
    }


def test_mismatched_native_configs_are_rejected(tmp_path):
    payload = OmegaConf.load(FMLIB_CONFIG_PATH)
    payload.n_trials = 4
    mismatched_path = tmp_path / "fmlib.yaml"
    OmegaConf.save(payload, mismatched_path)

    with pytest.raises(ConfigError, match="configs disagree"):
        load_case(AUTOCAMPAIGN_CONFIG_PATH, mismatched_path)


def test_channel_scope_is_resolved_from_both_native_configs(tmp_path):
    autocampaign = OmegaConf.load(AUTOCAMPAIGN_CONFIG_PATH)
    autocampaign.evaluate.is_product = False
    autocampaign.evaluate.model_configs_dir = "${train.output_dir}/configs_channels/"
    autocampaign_path = tmp_path / "autocampaign_channel.yaml"
    OmegaConf.save(autocampaign, autocampaign_path)
    fmlib = OmegaConf.load(FMLIB_CONFIG_PATH)
    fmlib.model_scope = "group"
    fmlib_path = tmp_path / "fmlib_channel.yaml"
    OmegaConf.save(fmlib, fmlib_path)

    case = load_case(autocampaign_path, fmlib_path)

    assert case.model_scope == "group"
    assert case.fmlib_config.model_scope == "group"


def test_reference_reads_real_evaluator_shards_and_normalizes_test_scores(tmp_path):
    case = load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH)
    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    pl.DataFrame(
        {
            "epk_id": [1],
            "report_month": ["2024-09-01"],
            "target": [0],
            "prediction": [0.25],
            "split_type": ["calib"],
        }
    ).write_parquet(predict_dir / "validation.parquet")
    pl.DataFrame(
        {
            "epk_id": [2],
            "report_month": ["2024-10-01"],
            "target": [1],
            "prediction": [0.75],
            "split_type": ["test"],
        }
    ).write_parquet(predict_dir / "test.parquet")

    validation, test = _read_scores(predict_dir, case)
    normalized = _normalized_scores(test, case)

    assert validation["prediction"].to_list() == [0.25]
    assert normalized.to_dict(as_series=False) == {
        "epk_id": [2],
        "report_month": ["2024-10-01"],
        "score": [0.75],
    }


def test_reference_runtime_config_casts_array_embeddings_without_changing_values(tmp_path):
    case = load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH)
    source_root = tmp_path / "source"
    paths = {}
    values = [[float(index) for index in range(64)]]
    for split_name in ("train", "valid", "test"):
        split_path = source_root / split_name
        split_path.mkdir(parents=True)
        pl.DataFrame(
            {
                "epk_id": [1],
                "cat_feature_1": ["known" if split_name == "train" else "unknown"],
                "group": ["channel_b" if split_name == "train" else "channel_a"],
                "seq_hidden_state": pl.Series(values, dtype=pl.Array(pl.Float32, 64)),
            }
        ).write_parquet(split_path / "part-0.parquet")
        paths[split_name] = split_path
    compatible_case = replace(
        case,
        train_path=paths["train"],
        valid_path=paths["valid"],
        test_path=paths["test"],
    )

    runtime_config = _write_runtime_config(compatible_case, tmp_path / "runtime")
    runtime = OmegaConf.load(runtime_config)
    converted = pl.read_parquet(Path(runtime.data.input_dir.train) / "part-0.parquet")

    assert converted.schema["seq_hidden_state"] == pl.List(pl.Float32)
    assert converted["seq_hidden_state"].to_list() == values
    assert converted["cat_feature_1"].to_list() == [0]
    assert "group" not in converted.columns
    assert converted["target_attr_2"].to_list() == [0]
    converted_test = pl.read_parquet(Path(runtime.data.input_dir.test) / "part-0.parquet")
    assert converted_test["cat_feature_1"].to_list() == [-1]
    assert converted_test["target_attr_2"].to_list() == [-1]
    assert runtime.data.group_column == "__product_only_group_column__"
    assert runtime.train.output_dir == OmegaConf.load(AUTOCAMPAIGN_CONFIG_PATH).train.output_dir


def test_reference_runtime_config_preserves_group_for_channel_training(tmp_path):
    autocampaign = OmegaConf.load(AUTOCAMPAIGN_CONFIG_PATH)
    autocampaign.evaluate.is_product = False
    autocampaign_path = tmp_path / "autocampaign_channel.yaml"
    OmegaConf.save(autocampaign, autocampaign_path)
    fmlib = OmegaConf.load(FMLIB_CONFIG_PATH)
    fmlib.model_scope = "group"
    fmlib_path = tmp_path / "fmlib_channel.yaml"
    OmegaConf.save(fmlib, fmlib_path)
    case = load_case(autocampaign_path, fmlib_path)

    runtime_path = _write_runtime_config(case, tmp_path / "runtime")
    runtime = OmegaConf.load(runtime_path)
    converted = pl.read_parquet(next(Path(runtime.data.input_dir.train).glob("*.parquet")))

    assert runtime.data.group_column == "group"
    assert runtime.evaluate.is_product is False
    assert str(runtime.evaluate.model_configs_dir).endswith("evaluation_configs_channels")
    assert "group" in converted.columns
    assert "target_attr_2" not in converted.columns


@pytest.mark.skipif(os.name == "nt", reason="Native autocampaign filenames contain characters forbidden by Windows")
def test_channel_evaluation_config_copies_remove_only_trainer_separator(tmp_path):
    source = tmp_path / "configs_channels"
    source.mkdir()
    (source / "<'catboost_binary_clf'>_channel_0.txt").write_text("{'depth': 4}", encoding="utf-8")
    destination = tmp_path / "evaluation_configs_channels"

    _prepare_channel_evaluation_configs(source, destination)

    copied = destination / "<'catboost_binary_clf'>channel_0.txt"
    assert copied.read_text(encoding="utf-8") == "{'depth': 4}"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("<'catboost_binary_clf'>_channel_with_underscores.txt", "channel_with_underscores"),
        ("__ac_lt__'catboost_binary_clf'__ac_gt___product.txt", "product"),
    ],
)
def test_channel_parameter_filename_preserves_complete_channel_value(filename, expected):
    assert _channel_from_params_path(Path(filename)) == expected


def test_xgboost_channel_parameter_filename_preserves_complete_channel_value():
    assert _channel_from_params_path(Path("<'xgboost_binary_clf'>_channel_with_underscores.txt")) == (
        "channel_with_underscores"
    )


def test_channel_training_configs_contain_exactly_one_string_group(tmp_path):
    autocampaign = OmegaConf.load(AUTOCAMPAIGN_CONFIG_PATH)
    autocampaign.evaluate.is_product = False
    autocampaign_path = tmp_path / "autocampaign_channel.yaml"
    OmegaConf.save(autocampaign, autocampaign_path)
    fmlib = OmegaConf.load(FMLIB_CONFIG_PATH)
    fmlib.model_scope = "group"
    fmlib_path = tmp_path / "fmlib_channel.yaml"
    OmegaConf.save(fmlib, fmlib_path)
    base_case = load_case(autocampaign_path, fmlib_path)
    source_paths = {}
    for split_name in ("train", "valid", "test"):
        split_path = tmp_path / "source" / split_name
        split_path.mkdir(parents=True)
        pl.DataFrame(
            {
                "epk_id": [1, 2],
                "group": ["channel_a", "channel_b"],
                "treatment": [0, 1],
                "cat_feature_1": ["a", "b"],
            }
        ).write_parquet(split_path / "part.parquet")
        source_paths[split_name] = split_path
    case = replace(
        base_case,
        train_path=source_paths["train"],
        valid_path=source_paths["valid"],
        test_path=source_paths["test"],
        categorical_columns=("cat_feature_1",),
        hidden_state_columns=(),
    )
    runtime_path = _write_runtime_config(case, tmp_path / "runtime")

    configs = _write_channel_training_configs(case, runtime_path, tmp_path / "runtime")

    assert len(configs) == 2
    groups = []
    for config_path in configs:
        config = OmegaConf.load(config_path)
        frame = pl.read_parquet(Path(config.data.input_dir.train) / "part-00000.parquet")
        assert frame.schema["group"] == pl.String
        assert frame["group"].n_unique() == 1
        groups.extend(frame["group"].unique().to_list())
    assert sorted(groups) == ["channel_a", "channel_b"]


def _result(tmp_path, name, params, scores, timing):
    scores_path = tmp_path / f"{name}.parquet"
    pl.DataFrame(
        {
            "epk_id": [1, 2, 3],
            "report_month": ["2026-01-01"] * 3,
            "score": scores,
        }
    ).write_parquet(scores_path)
    timing = {
        "hyperparameter_search_and_train": 0.0,
        "inference_model_fit": 0.0,
        "predict": 0.0,
        "total_train_before_predict": 0.0,
    } | timing
    return {
        "selected_params": {"product": params},
        "resolved_model_params": {"product": params | {"task_type": "GPU"}},
        "feature_names": ["feature"],
        "validation_roc_auc": {"product": 0.8},
        "test_roc_auc": {"product": 0.79},
        "timing_seconds": timing,
        "scores_path": str(scores_path),
    }


def test_comparison_reports_parameters_scores_metrics_and_time(tmp_path):
    case = replace(
        load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH),
        max_roc_auc_difference=0.001,
        max_score_ks_statistic=0.5,
        max_score_quantile_difference=0.02,
    )
    timing = {
        "hyperparameter_search_and_train": 1.0,
        "inference_model_fit": 0.5,
        "predict": 0.1,
        "total_train_before_predict": 1.5,
    }
    reference = _result(tmp_path, "reference", {"depth": 4}, [0.1, 0.5, 0.9], timing)
    candidate = _result(tmp_path, "candidate", {"depth": 4}, [0.1001, 0.5001, 0.9001], timing)
    candidate["resolved_model_params"]["product"]["random_seed"] = 42

    comparison = compare_results(reference, candidate, case)

    assert comparison["passed"] is True
    assert comparison["best_params_equal"] is True
    assert comparison["resolved_model_params_equal"] is False
    assert comparison["resolved_model_param_differences"] == {
        "product": {"random_seed": {"autocampaignxfm": None, "fmlib": 42}}
    }
    assert comparison["feature_names_equal"] is True
    assert comparison["score_distribution"]["passed"] is True
    assert comparison["score_distribution"]["rows"] == 3
    assert comparison["score_distribution"]["ks_statistic"] == pytest.approx(1 / 3)
    assert comparison["metrics"]["passed"] is True
    assert comparison["metrics"]["test_roc_auc"]["product"]["absolute_difference"] == 0.0
    assert comparison["timing_seconds"]["autocampaignxfm"] == timing


def test_comparison_includes_calibrated_scores_and_metrics(tmp_path):
    case = replace(
        load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH),
        max_roc_auc_difference=0.01,
        max_score_ks_statistic=1.0,
        max_score_quantile_difference=1.0,
    )
    reference = _result(tmp_path, "reference_raw", {"depth": 2}, [0.1, 0.5, 0.9], {})
    candidate = _result(tmp_path, "candidate_raw", {"depth": 2}, [0.1, 0.5, 0.9], {})
    for name, result, values in (
        ("reference_cal", reference, [0.2, 0.5, 0.8]),
        ("candidate_cal", candidate, [0.201, 0.501, 0.801]),
    ):
        path = tmp_path / f"{name}.parquet"
        pl.DataFrame({"epk_id": [1, 2, 3], "report_month": ["2026-01-01"] * 3, "score": values}).write_parquet(path)
        result["calibrated_scores_path"] = str(path)
        result["calibrated_test_roc_auc"] = {"product": 0.8}
    candidate_calibrated = pl.read_parquet(candidate["calibrated_scores_path"]).with_columns(
        pl.col("report_month").str.to_date()
    )
    candidate_calibrated.write_parquet(candidate["calibrated_scores_path"])

    comparison = compare_results(reference, candidate, case)

    assert comparison["calibrated_score_distribution"]["rows"] == 3
    assert comparison["metrics"]["calibrated_test_roc_auc"]["product"]["absolute_difference"] == 0.0


def test_comparison_reports_different_best_parameters_without_failing_quality_parity(tmp_path):
    case = replace(
        load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH),
        max_score_ks_statistic=1.0,
        max_score_quantile_difference=1.0,
    )
    timing = {"predict": 0.1}
    reference = _result(tmp_path, "reference", {"depth": 4}, [0.1, 0.5, 0.9], timing)
    candidate = _result(tmp_path, "candidate", {"depth": 6}, [0.1, 0.5, 0.9], timing)

    comparison = compare_results(reference, candidate, case)

    assert comparison["passed"] is True
    assert comparison["best_params_equal"] is False
    expected = {"product": {"depth": {"autocampaignxfm": 4, "fmlib": 6}}}
    assert comparison["best_param_differences"] == expected
    assert comparison["resolved_model_param_differences"] == expected


def test_comparison_supports_per_channel_parameters_and_metrics(tmp_path):
    case = replace(
        load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH),
        model_scope="group",
        max_score_ks_statistic=1.0,
        max_score_quantile_difference=1.0,
    )
    reference = _result(tmp_path, "reference", {"depth": 4}, [0.1, 0.5, 0.9], {})
    candidate = _result(tmp_path, "candidate", {"depth": 4}, [0.1, 0.5, 0.9], {})
    for result in (reference, candidate):
        result["selected_params"] = {
            "group:a": {"depth": 4},
            "group:b": {"depth": 5},
        }
        result["resolved_model_params"] = {
            "group:a": {"depth": 4},
            "group:b": {"depth": 5},
        }
        result["validation_roc_auc"] = {"group:a": 0.8, "group:b": 0.81}
        result["test_roc_auc"] = {"group:a": 0.79, "group:b": 0.80}

    comparison = compare_results(reference, candidate, case)

    assert comparison["passed"] is True
    assert set(comparison["metrics"]["test_roc_auc"]) == {"group:a", "group:b"}


def test_comparison_fails_when_score_distributions_diverge(tmp_path):
    case = replace(
        load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH),
        max_score_ks_statistic=0.1,
        max_score_quantile_difference=0.1,
    )
    reference = _result(tmp_path, "reference", {"depth": 4}, [0.1, 0.2, 0.3], {})
    candidate = _result(tmp_path, "candidate", {"depth": 4}, [0.7, 0.8, 0.9], {})

    comparison = compare_results(reference, candidate, case)

    assert comparison["passed"] is False
    assert comparison["score_distribution"]["passed"] is False


def test_comparison_fails_when_roc_auc_diverges(tmp_path):
    case = replace(
        load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH),
        max_roc_auc_difference=0.001,
        max_score_ks_statistic=1.0,
        max_score_quantile_difference=1.0,
    )
    reference = _result(tmp_path, "reference", {"depth": 4}, [0.1, 0.5, 0.9], {})
    candidate = _result(tmp_path, "candidate", {"depth": 4}, [0.1, 0.5, 0.9], {})
    candidate["test_roc_auc"]["product"] = 0.80

    comparison = compare_results(reference, candidate, case)

    assert comparison["passed"] is False
    assert comparison["metrics"]["passed"] is False


def test_reference_runner_uses_explicit_python_and_repository_paths(tmp_path, monkeypatch):
    case = replace(load_case(AUTOCAMPAIGN_CONFIG_PATH, FMLIB_CONFIG_PATH), output_dir=tmp_path)
    autocampaign_root = WORKSPACE_ROOT / "autocampaignxfm"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = tmp_path / "autocampaignxfm"
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps({"pipeline": "autocampaignxfm"}), encoding="utf-8")

    monkeypatch.setattr("tools.automl_parity.binary.subprocess.run", fake_run)

    result = run_reference(case, autocampaign_root, Path(sys.executable))

    command, kwargs = calls[0]
    assert result == {"pipeline": "autocampaignxfm"}
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "tools.automl_parity.reference"]
    assert command[command.index("--config") + 1] == str(AUTOCAMPAIGN_CONFIG_PATH.resolve())
    assert command[command.index("--fmlib-config") + 1] == str(FMLIB_CONFIG_PATH.resolve())
    assert kwargs["check"] is True
    assert str(autocampaign_root.resolve()) in kwargs["env"]["PYTHONPATH"]
    assert kwargs["env"]["MPLBACKEND"] == "Agg"
    assert kwargs["env"]["MPLCONFIGDIR"] == str((tmp_path / "autocampaignxfm" / ".matplotlib").resolve())


@pytest.mark.skipif(
    os.environ.get("FMLIB_RUN_AUTOCAMPAIGN_PARITY") != "1",
    reason="Set FMLIB_RUN_AUTOCAMPAIGN_PARITY=1 to run the GPU comparison",
)
def test_live_binary_parity():
    comparison = run_comparison(
        AUTOCAMPAIGN_CONFIG_PATH,
        FMLIB_CONFIG_PATH,
        autocampaign_root=WORKSPACE_ROOT / "autocampaignxfm",
        autocampaign_python=sys.executable,
    )

    assert comparison["passed"] is True


@pytest.mark.skipif(
    os.environ.get("FMLIB_RUN_AUTOCAMPAIGN_PARITY") != "1",
    reason="Set FMLIB_RUN_AUTOCAMPAIGN_PARITY=1 to run the GPU comparison",
)
@pytest.mark.parametrize(
    ("autocampaign_config", "fmlib_config"),
    [
        (XGBOOST_AUTOCAMPAIGN_CONFIG_PATH, XGBOOST_FMLIB_CONFIG_PATH),
        (XGBOOST_CHANNEL_AUTOCAMPAIGN_CONFIG_PATH, XGBOOST_CHANNEL_FMLIB_CONFIG_PATH),
    ],
    ids=["xgboost-product", "xgboost-channel"],
)
def test_live_xgboost_binary_parity(autocampaign_config, fmlib_config):
    comparison = run_comparison(
        autocampaign_config,
        fmlib_config,
        autocampaign_root=WORKSPACE_ROOT / "autocampaignxfm",
        autocampaign_python=Path(os.environ.get("AUTOCAMPAIGN_PYTHON", sys.executable)),
    )

    assert comparison["passed"] is True
