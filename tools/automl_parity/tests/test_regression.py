import os
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from tools.automl_parity.regression import compare_results, load_case


def _case(monkeypatch):
    root = Path(__file__).resolve().parents[3].parent
    monkeypatch.setenv("SBER_INFRA_ROOT", str(root))
    configs = root / "fmlib-main" / "examples" / "automl" / "tests" / "configs"
    return load_case(
        configs / "parity_regression_product_autocampaignxfm.yaml",
        configs / "parity_regression_product_fmlib.yaml",
    )


def test_regression_parity_config_resolves_three_trial_mse_product(monkeypatch):
    case = _case(monkeypatch)
    assert case.model_scope == "product"
    assert case.metric == "mse"
    assert case.n_trials == 3
    assert case.random_state == 42


def test_regression_comparison_covers_metrics_distribution_params_and_time(
    tmp_path, monkeypatch
):
    case = replace(
        _case(monkeypatch),
        max_relative_metric_difference=0.05,
        max_score_ks_statistic=0.5,
        max_score_quantile_difference=0.5,
    )
    reference_scores = tmp_path / "reference.parquet"
    candidate_scores = tmp_path / "candidate.parquet"
    base = pl.DataFrame({
        "epk_id": [1, 2, 3, 4],
        "report_month": ["2026-01-01"] * 4,
        "score": [1.0, 2.0, 3.0, 4.0],
    })
    base.write_parquet(reference_scores)
    base.with_columns(pl.col("score") + 0.01).write_parquet(candidate_scores)
    reference = {
        "selected_params": {"product": {"depth": 4}},
        "test_metrics": {"mse": 1.0, "mae": 0.8, "mape": 0.2},
        "timing_seconds": {"total": 4.0},
        "scores_path": str(reference_scores),
    }
    candidate = {
        "selected_params": {"product": {"depth": 5}},
        "test_metrics": {"mse": 1.01, "mae": 0.81, "mape": 0.201},
        "timing_seconds": {"total": 2.0},
        "scores_path": str(candidate_scores),
    }

    comparison = compare_results(reference, candidate, case)

    assert comparison["passed"]
    assert set(comparison["metrics"]) == {"mse", "mae", "mape"}
    assert comparison["best_params_equal"] is False
    assert "native categorical" in comparison["best_param_note"].lower()
    assert comparison["timing_seconds"]["autocampaignxfm"]["total"] == 4.0


def test_regression_comparison_includes_calibrated_outputs(tmp_path, monkeypatch):
    case = replace(
        _case(monkeypatch),
        max_relative_metric_difference=0.05,
        max_score_ks_statistic=1.0,
        max_score_quantile_difference=1.0,
    )
    paths = []
    for name, offset in (("reference", 0.0), ("candidate", 0.01)):
        path = tmp_path / f"{name}.parquet"
        pl.DataFrame({
            "epk_id": [1, 2, 3],
            "report_month": ["2026-01-01"] * 3,
            "score": [1.0 + offset, 2.0 + offset, 3.0 + offset],
        }).write_parquet(path)
        paths.append(path)
    common = {
        "selected_params": {"product": {"depth": 2}},
        "test_metrics": {"mse": 1.0, "mae": 0.8, "mape": 0.2},
        "calibrated_test_metrics": {"mse": 0.9, "mae": 0.7, "mape": 0.19},
        "timing_seconds": {"total": 1.0},
    }
    reference = common | {
        "scores_path": str(paths[0]),
        "calibrated_scores_path": str(paths[0]),
    }
    candidate = common | {
        "scores_path": str(paths[1]),
        "calibrated_scores_path": str(paths[1]),
    }
    pl.read_parquet(paths[1]).with_columns(
        pl.col("report_month").str.to_date()
    ).write_parquet(paths[1])

    comparison = compare_results(reference, candidate, case)

    assert comparison["calibrated_metrics"]["mse"]["relative_difference"] == 0.0
    assert comparison["calibrated_score_distribution"]["rows"] == 3


@pytest.mark.skipif(
    os.environ.get("FMLIB_RUN_LIVE_REGRESSION_PARITY") != "1",
    reason="requires GPU reference environment",
)
def test_live_regression_product_and_channel_parity(monkeypatch):
    from tools.automl_parity.regression import run_comparison

    root = os.environ["SBER_INFRA_ROOT"]
    configs = os.path.join(root, "fmlib-main", "examples", "automl", "tests", "configs")
    python = os.path.join(root, "autocampaignxfm", "env", "bin", "python")
    for scope in ("product", "group"):
        result = run_comparison(
            os.path.join(configs, f"parity_regression_{scope}_autocampaignxfm.yaml"),
            os.path.join(configs, f"parity_regression_{scope}_fmlib.yaml"),
            autocampaign_root=os.path.join(root, "autocampaignxfm"),
            autocampaign_python=python,
        )
        assert result["passed"]
