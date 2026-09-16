"""Isolated uplift parity normalization without importing reference dependencies."""

from __future__ import annotations

import polars as pl
import pytest

from avatar.automl import UpliftTaskConfig
from tools.automl_parity.uplift_reference import _client_scores, _joined_metrics


def _config(**overrides) -> UpliftTaskConfig:
    values = {
        "env_type": "local",
        "backend": "boosting",
        "engine": "xgboost",
        "device": "cpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "treatment_column": "treatment",
        "inverse_treatment": False,
        "report_month_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ("x",),
        "hidden_state_columns": (),
        "model_scope": "product",
        "hyperopt": False,
        "output_dir": "fmlib_automl_outputs",
        "environment": {},
        "estimate_propensity": False,
    }
    values.update(overrides)
    return UpliftTaskConfig(**values)


def test_reference_client_artifacts_are_normalized_from_real_metalearner_labels(tmp_path) -> None:
    labels = ["<class 'causalml.inference.meta.slearner.BaseSClassifier'>", "BaseTLearner", "BaseXLearner"]
    for index, label in enumerate(labels):
        pl.DataFrame(
            {
                "epk_id": [1, 2],
                "report_month": ["2026-01-01", "2026-01-01"],
                "metalearner": [label, label],
                "treatment_probs": [0.6 + index * 0.01, 0.7 + index * 0.01],
                "control_probs": [0.2, 0.3],
                "split_type": ["test", "test"],
            }
        ).write_parquet(tmp_path / f"{index}.parquet")
    config = _config()
    scores = _client_scores(tmp_path, config)
    assert scores.columns == ["epk_id", "report_month", "score_s", "score_t", "score_x"]
    assert scores.shape == (2, 5)


def test_reference_calibrated_client_artifacts_use_calibrated_probability_columns(tmp_path) -> None:
    labels = ["BaseSLearner", "BaseTLearner", "BaseXLearner"]
    for index, label in enumerate(labels):
        pl.DataFrame(
            {
                "epk_id": [1],
                "report_month": ["2026-01-01"],
                "metalearner": [label],
                "treatment_probs": [0.9],
                "control_probs": [0.1],
                "treatment_probs_cal": [0.6 + index * 0.01],
                "control_probs_cal": [0.2],
            }
        ).write_parquet(tmp_path / f"{index}.parquet")
    config = _config()

    scores = _client_scores(tmp_path, config, calibrated=True)

    assert scores.select("score_s", "score_t", "score_x").row(0) == pytest.approx((0.4, 0.41, 0.42))


def test_reference_metrics_use_truth_from_the_same_joined_score_rows() -> None:
    rows = 20
    frame = pl.DataFrame(
        {
            "epk_id": list(reversed(range(rows))),
            "report_month": ["2026-01-01"] * rows,
            "target": [1, 0, 0, 1] * 5,
            "treatment": [1, 0] * 10,
            "score_s": list(reversed(range(rows))),
            "score_t": [value + 0.1 for value in reversed(range(rows))],
            "score_x": [value + 0.2 for value in reversed(range(rows))],
        }
    )
    config = _config()

    metrics = _joined_metrics(frame, config)

    assert set(metrics) == {"s", "t", "x"}
    expected_names = {"qini_auc", "uplift_auc", "uplift_at_10", "uplift_at_20", "uplift_at_50"}
    assert all(set(values) == expected_names for values in metrics.values())


def test_production_package_has_no_causalml_import() -> None:
    from pathlib import Path

    automl_root = Path(__file__).resolve().parents[3] / "fmlib/automl"
    production_files = [
        path for path in automl_root.rglob("*.py") if "_parity" not in path.parts and "tests" not in path.parts
    ]
    assert all("import causalml" not in path.read_text(encoding="utf-8") for path in production_files)


def test_native_date_resolver_can_resolve_reference_config(tmp_path, monkeypatch) -> None:
    """Native autocampaign dates and environment paths resolve in the uplift CLI."""
    from datetime import date

    from omegaconf import OmegaConf

    monkeypatch.setenv("PARITY_DATA_ROOT", str(tmp_path))
    if not OmegaConf.has_resolver("date"):
        OmegaConf.register_new_resolver("date", date.fromisoformat)
    config = OmegaConf.create(
        {
            "data": {
                "input_dir": {"train": "${oc.env:PARITY_DATA_ROOT}/train"},
                "train_date_range": {"min": "${date: 2024-01-01}"},
            }
        }
    )

    resolved = OmegaConf.to_container(config, resolve=True)

    assert resolved["data"]["input_dir"]["train"] == f"{tmp_path}/train"
    assert resolved["data"]["train_date_range"]["min"] == date(2024, 1, 1)
