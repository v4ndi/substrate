"""Exact learner mathematics and native-engine uplift integration."""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from fmlib.automl import PredictionResult, UpliftTask, UpliftTaskConfig
from fmlib.automl.backends.boosting.uplift import (
    UPLIFT_SCORE_COLUMNS,
    UpliftBoostingBackend,
)
from fmlib.automl.data import FeatureSchema
from fmlib.automl.exceptions import ConfigError, SchemaError
from fmlib.automl.metrics import resolve_metric


@dataclass
class _ConstantComponent:
    values: np.ndarray

    def predict_score(self, frame, schema):
        return self.values[: frame.height]


def _schema() -> FeatureSchema:
    return FeatureSchema(
        categorical=(),
        numerical=("x",),
        feature_order=("x",),
        dtypes={"x": "Float64"},
        target_column="target",
        client_id_column="epk_id",
        treatment_column="treatment",
        group_column="group",
    )


def test_x_learner_effect_formula_and_constant_propensity() -> None:
    frame = pl.DataFrame({"x": [0.0, 1.0]})
    backend = UpliftBoostingBackend(
        engine="xgboost", params={}, random_state=42, device="cpu"
    )
    components = {
        "x_control_outcome": _ConstantComponent(np.array([0.2, 0.4])),
        "x_treatment_outcome": _ConstantComponent(np.array([0.6, 0.8])),
        "x_control_effect": _ConstantComponent(np.array([0.1, 0.2])),
        "x_treatment_effect": _ConstantComponent(np.array([0.3, 0.4])),
    }
    effect, control, treated, propensity = backend._predict_learner(
        "x", components, frame, _schema()
    )
    np.testing.assert_allclose(propensity, 0.5)
    np.testing.assert_allclose(effect, [0.2, 0.3])
    np.testing.assert_allclose(control, [0.2, 0.4])
    np.testing.assert_allclose(treated, [0.6, 0.8])
    assert not np.allclose(treated - control, effect)


def test_x_formula_uses_bounded_effect_regressors_before_propensity_combination() -> (
    None
):
    frame = pl.DataFrame({"x": [0.0]})
    backend = UpliftBoostingBackend(
        engine="xgboost", params={}, random_state=42, device="cpu"
    )
    components = {
        "x_control_outcome": _ConstantComponent(np.array([0.2])),
        "x_treatment_outcome": _ConstantComponent(np.array([0.6])),
        "x_control_effect": _ConstantComponent(np.array([2.0])),
        "x_treatment_effect": _ConstantComponent(np.array([-0.5])),
        "x_propensity": _ConstantComponent(np.array([0.8])),
    }
    effect, control, treated, propensity = backend._predict_learner(
        "x", components, frame, _schema()
    )
    np.testing.assert_allclose(propensity, [0.8])
    np.testing.assert_allclose(effect, [0.8 * 1.0 + 0.2 * -0.5])
    np.testing.assert_allclose(control, [0.2])
    np.testing.assert_allclose(treated, [0.6])


def test_x_learner_pseudo_outcomes_follow_the_standard_definition(monkeypatch) -> None:
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    target = np.array([0.0, 1.0, 1.0, 0.0])
    treatment = np.array([0, 0, 1, 1], dtype=np.int8)
    backend = UpliftBoostingBackend(
        engine="xgboost", params={}, random_state=42, device="cpu"
    )
    outcome_components = {
        "x_control_outcome": _ConstantComponent(np.full(4, 0.2)),
        "x_treatment_outcome": _ConstantComponent(np.full(4, 0.8)),
    }
    fitted_targets: dict[str, np.ndarray] = {}

    monkeypatch.setattr(
        backend, "_fit_t", lambda *args, **kwargs: dict(outcome_components)
    )

    def capture_effect_fit(
        kind, params, train, train_target, valid, valid_target, schema
    ):
        label = "control" if not fitted_targets else "treatment"
        fitted_targets[label] = np.asarray(train_target)
        return _ConstantComponent(np.zeros(train.height))

    monkeypatch.setattr(backend, "_fit_component", capture_effect_fit)
    backend._fit_x({}, frame, target, treatment, frame, target, treatment, _schema())

    np.testing.assert_allclose(fitted_targets["control"], [0.8 - 0.0, 0.8 - 1.0])
    np.testing.assert_allclose(fitted_targets["treatment"], [1.0 - 0.2, 0.0 - 0.2])


def test_hyperopt_keeps_each_already_fitted_best_composite(monkeypatch) -> None:
    frame = pl.DataFrame({"x": np.arange(8.0), "treatment": [0, 1] * 4})
    target = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    treatment = np.array([0, 1] * 4, dtype=np.int8)
    backend = UpliftBoostingBackend(
        engine="xgboost", params={}, random_state=42, device="cpu"
    )
    fitted: list[dict[str, object]] = []

    def fake_fit(params):
        composite = {"marker": object(), "value": params["max_depth"]}
        fitted.append(composite)
        return composite

    monkeypatch.setattr(backend, "_fit_s", lambda params, *args: fake_fit(params))
    monkeypatch.setattr(
        backend, "_fit_t", lambda params, *args, **kwargs: fake_fit(params)
    )
    monkeypatch.setattr(backend, "_fit_x", lambda params, *args: fake_fit(params))
    monkeypatch.setattr(
        backend,
        "_objective",
        lambda learner, components, *args: float(components["value"]),
    )
    backend.fit_composite(
        frame,
        target,
        treatment,
        frame,
        target,
        treatment,
        _schema(),
        hyperopt=True,
        n_trials=3,
        model_params={},
        search_space={"max_depth": {"type": "int", "low": 1, "high": 3}},
        metric=resolve_metric("qini_auc", "uplift", "optimization"),
        part_name="global",
    )
    assert len(fitted) == 9
    assert set(backend.learner_params) == {"s", "t", "x"}
    assert backend.components["marker"] in {item["marker"] for item in fitted}


def test_compatible_constituents_reuse_prepared_catboost_data(monkeypatch) -> None:
    calls = {"prepare": 0, "fit_prepared": 0}

    class FakeComponent:
        def prepare_fit_data(self, *args, **kwargs):
            calls["prepare"] += 1
            return object()

        def fit_prepared(self, prepared):
            calls["fit_prepared"] += 1

    backend = UpliftBoostingBackend(
        engine="catboost", params={}, random_state=42, device="cpu"
    )
    backend._reuse_prepared = True
    monkeypatch.setattr(backend, "_component", lambda kind, params: FakeComponent())
    frame = pl.DataFrame({"x": [0.0, 1.0]})
    target = np.array([0, 1])
    for _ in range(2):
        backend._fit_component(
            "classifier",
            {},
            frame,
            target,
            frame,
            target,
            _schema(),
            cache_key="shared",
        )
    assert calls == {"prepare": 1, "fit_prepared": 2}


def _split(rows: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = np.arange(rows) % 2
    x = rng.normal(size=rows)
    probability = 1 / (
        1 + np.exp(-(-0.3 + 0.7 * x + treatment * (0.8 + 0.4 * (x > 0))))
    )
    target = (rng.random(rows) < probability).astype(np.int8)
    for arm in (0, 1):
        indices = np.flatnonzero(treatment == arm)[:2]
        target[indices] = [0, 1]
    return pl.DataFrame({
        "epk_id": np.arange(seed * 1000, seed * 1000 + rows),
        "report_month": ["2026-01-01"] * rows,
        "target": target,
        "treatment": treatment.astype(np.int8),
        "x": x,
    })


@pytest.mark.parametrize(
    ("engine", "model_params"),
    [
        ("xgboost", {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.2}),
        ("catboost", {"iterations": 8, "depth": 2, "learning_rate": 0.2}),
    ],
)
def test_real_engine_composite_round_trip(
    tmp_path, engine: str, model_params: dict
) -> None:
    train, valid, test = (_split(80, seed) for seed in (1, 2, 3))
    paths = [tmp_path / f"{name}.parquet" for name in ("train", "valid", "test")]
    for frame, path in zip((train, valid, test), paths, strict=True):
        frame.write_parquet(path)
    config = UpliftTaskConfig(
        env_type="local",
        backend="boosting",
        engine=engine,
        device="cpu",
        target_column="target",
        client_id_column="epk_id",
        numerical_columns=["x"],
        categorical_columns=(),
        hidden_state_columns=(),
        group_column=None,
        treatment_column="treatment",
        inverse_treatment=False,
        date_column="report_month",
        hyperopt=False,
        environment={},
        estimate_propensity=False,
        model_params=model_params,
        verbose=False,
        output_dir=tmp_path / "output",
    )
    task = UpliftTask(config)
    training = task.train(paths[0], paths[1])
    assert set(training.validation_metrics) == {"global:s", "global:t", "global:x"}
    prediction = task.predict(paths[2])
    assert prediction.scores.columns == [
        "epk_id",
        "report_month",
        "treatment",
        *UPLIFT_SCORE_COLUMNS,
    ]
    values = prediction.scores.select(UPLIFT_SCORE_COLUMNS).to_numpy()
    task._validate_score_matrix(values)
    open_figures = set(plt.get_fignums())
    evaluation = task.evaluate(paths[2], prediction)
    arm_evaluation = task.evaluate(
        paths[2], prediction, metrics=["treatment_roc_auc", "control_roc_auc"]
    )
    assert {"s_qini_auc", "t_uplift_at_20", "x_control_roc_auc"} <= set(
        evaluation.metrics_raw
    )
    assert set(arm_evaluation.metrics_raw) == {
        f"{learner}_{name}"
        for learner in ("s", "t", "x")
        for name in (
            "n_samples",
            "n_treatment",
            "n_control",
            "n_positive_treatment",
            "n_positive_control",
            "treatment_roc_auc",
            "control_roc_auc",
        )
    }
    assert evaluation.metrics_by_group_raw["learner"].unique().sort().to_list() == [
        "s",
        "t",
        "x",
    ]
    assert {"metrics_raw", "feature_importance_raw"} <= set(evaluation.excel_paths_raw)
    assert set(plt.get_fignums()) == open_figures
    shuffled_evaluation = task.evaluate(
        paths[2], PredictionResult(prediction.scores.reverse())
    )
    assert shuffled_evaluation.metrics_raw == pytest.approx(evaluation.metrics_raw)

    with pytest.raises(SchemaError, match="missing required columns.*report_month"):
        task.evaluate(
            paths[2], PredictionResult(prediction.scores.drop("report_month"))
        )

    mismatched_scores = prediction.scores.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("epk_id") + 10_000_000)
        .otherwise(pl.col("epk_id"))
        .alias("epk_id")
    )
    with pytest.raises(SchemaError, match="must match evaluation rows one-to-one"):
        task.evaluate(paths[2], PredictionResult(mismatched_scores))
    artifact = task.save()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    restored = UpliftTask.load(artifact)
    assert not hasattr(restored, "_calibrators")
    assert "uplift_calibrators" not in manifest
    restored_values = (
        restored.predict(paths[2]).scores.select(UPLIFT_SCORE_COLUMNS).to_numpy()
    )
    np.testing.assert_allclose(restored_values, values, atol=2e-7)
    production_path = tmp_path / "production.parquet"
    test.drop("target", "treatment").write_parquet(production_path)
    production = restored.predict(production_path)
    assert production.scores.height == test.height
    assert (
        "target" not in production.scores.columns
        and "treatment" not in production.scores.columns
    )


def test_combined_layout_returns_independent_branches_and_rejects_unknown_group(
    tmp_path,
) -> None:
    frames = []
    for seed in (11, 12, 13):
        frame = _split(96, seed).with_columns(
            pl.when(pl.int_range(pl.len()) % 4 < 2)
            .then(pl.lit("a"))
            .otherwise(pl.lit("b"))
            .alias("group")
        )
        target = frame["target"].to_numpy().copy()
        treatment = frame["treatment"].to_numpy()
        groups = frame["group"].to_numpy()
        for group in ("a", "b"):
            for arm in (0, 1):
                indices = np.flatnonzero((groups == group) & (treatment == arm))[:2]
                target[indices] = [0, 1]
        frames.append(frame.with_columns(pl.Series("target", target)))
    paths = [tmp_path / f"{name}.parquet" for name in ("train", "valid", "test")]
    for frame, path in zip(frames, paths, strict=True):
        frame.write_parquet(path)
    task = UpliftTask(
        UpliftTaskConfig(
            env_type="local",
            backend="boosting",
            engine="xgboost",
            device="cpu",
            target_column="target",
            client_id_column="epk_id",
            numerical_columns=["x"],
            categorical_columns=(),
            hidden_state_columns=(),
            group_column="group",
            treatment_column="treatment",
            model_layout="global_and_per_group",
            inverse_treatment=False,
            date_column="report_month",
            hyperopt=False,
            environment={},
            estimate_propensity=False,
            model_params={"n_estimators": 4, "max_depth": 2},
            verbose=False,
            output_dir=tmp_path / "output",
        )
    )
    task.train(paths[0], paths[1])
    prediction = task.predict(paths[2])
    evaluation = task.evaluate(paths[2], prediction)
    assert prediction.scores.height == frames[2].height * 2
    assert set(prediction.scores["model_layout"].unique()) == {"global", "per_group"}
    assert {"global_s_qini_auc", "per_group_s_qini_auc"} <= set(evaluation.metrics_raw)
    assert set(evaluation.metrics_by_group_raw["model_layout"].unique()) == {
        "global",
        "per_group",
    }
    unknown = frames[2].with_columns(pl.lit("unknown").alias("group"))
    unknown_path = tmp_path / "unknown.parquet"
    unknown.write_parquet(unknown_path)
    with pytest.raises(SchemaError, match=r"Unknown group values.*'unknown'.*96 rows"):
        task.predict(unknown_path)
    global_model = next(item for item in task._models if item.layout == "global")
    expected = task._predict_entry(
        task._read(unknown_path).with_row_index("__row_id"), global_model
    )
    global_only = (
        task.predict(unknown_path, model_layout="global")
        .scores.select(UPLIFT_SCORE_COLUMNS)
        .to_numpy()
    )
    np.testing.assert_allclose(global_only, expected)
    with pytest.raises(SchemaError, match=r"Unknown group values.*'unknown'.*96 rows"):
        task.predict(unknown_path, model_layout="per_group")
    task._models = [item for item in task._models if item.layout == "per_group"]
    with pytest.raises(ConfigError, match="unavailable model branches"):
        task.predict(unknown_path)


def test_group_scope_is_forbidden_without_group_column(tmp_path) -> None:
    with pytest.raises(
        ConfigError, match="model_layout='per_group'.*group_column=None"
    ):
        UpliftTaskConfig(
            env_type="local",
            backend="boosting",
            engine="xgboost",
            device="cpu",
            target_column="target",
            client_id_column="epk_id",
            numerical_columns=["x"],
            categorical_columns=(),
            hidden_state_columns=(),
            group_column=None,
            treatment_column="treatment",
            model_layout="per_group",
            inverse_treatment=False,
            date_column="report_month",
            hyperopt=False,
            environment={},
            estimate_propensity=False,
            model_params={"n_estimators": 4, "max_depth": 2},
            verbose=False,
            output_dir=tmp_path / "logical-output",
        )


def test_estimated_propensity_is_fitted_saved_and_bounded(tmp_path) -> None:
    train, valid, test = (_split(80, seed) for seed in (41, 42, 43))
    paths = [
        tmp_path / f"propensity-{name}.parquet" for name in ("train", "valid", "test")
    ]
    for frame, path in zip((train, valid, test), paths, strict=True):
        frame.write_parquet(path)
    task = UpliftTask(
        UpliftTaskConfig(
            env_type="local",
            backend="boosting",
            engine="xgboost",
            device="cpu",
            target_column="target",
            client_id_column="epk_id",
            numerical_columns=["x"],
            categorical_columns=(),
            hidden_state_columns=(),
            group_column=None,
            treatment_column="treatment",
            estimate_propensity=True,
            inverse_treatment=False,
            date_column="report_month",
            hyperopt=False,
            environment={},
            model_params={"n_estimators": 6, "max_depth": 2},
            verbose=False,
            output_dir=tmp_path / "propensity-output",
        )
    )
    task.train(paths[0], paths[1])
    propensity = task.predict(paths[2]).scores["score_x_propensity"].to_numpy()
    assert np.all((propensity >= 0) & (propensity <= 1))
    assert not np.allclose(propensity, 0.5)
    restored = UpliftTask.load(task.save())
    np.testing.assert_allclose(
        restored.predict(paths[2]).scores["score_x_propensity"].to_numpy(), propensity
    )
