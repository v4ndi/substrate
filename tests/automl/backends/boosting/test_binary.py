import json

import numpy as np
import polars as pl
import pytest

from avatar.automl.backends.boosting import BinaryBoostingBackend
from avatar.automl.data import FeatureSchema
from avatar.automl.exceptions import ArtifactIntegrityError, UnsupportedBackendError


def _schema():
    return FeatureSchema(
        categorical=("segment",),
        numerical=("balance",),
        feature_order=("segment", "balance"),
        dtypes={"segment": "String", "balance": "Float64"},
        target_column="target",
        client_id_column="epk_id",
        treatment_column=None,
        group_column=None,
    )


def _frame():
    return pl.DataFrame(
        {
            "segment": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "balance": [-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0],
        }
    )


def test_supported_catboost_accepts_native_polars_pool_input():
    catboost = pytest.importorskip("catboost")
    frame = _frame()

    pool = catboost.Pool(
        frame,
        label=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        cat_features=["segment"],
    )

    assert pool.num_row() == frame.height
    assert pool.num_col() == frame.width


@pytest.mark.parametrize(
    ("engine", "params"),
    [
        ("catboost", {"iterations": 8, "depth": 2}),
        ("xgboost", {"n_estimators": 8, "max_depth": 2}),
    ],
)
def test_backend_fit_predict_is_deterministic_and_ignores_column_order(tmp_path, engine, params):
    pytest.importorskip(engine)
    frame = _frame()
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    backend = BinaryBoostingBackend(engine, params, random_state=42, device="cpu")

    backend.fit(frame, target, _schema(), valid_frame=frame, valid_target=target)
    direct = backend.predict_score(frame, _schema())
    reordered = backend.predict_score(
        frame.with_columns(pl.lit("unused").alias("extra")).select("extra", "balance", "segment"), _schema()
    )
    artifact = tmp_path / engine
    backend.save(artifact)
    restored = BinaryBoostingBackend.load(artifact)
    gpu_restored = BinaryBoostingBackend.load(artifact, device="gpu")
    metadata = json.loads((artifact / "backend.json").read_text(encoding="utf-8"))

    assert direct.shape == (frame.height,)
    assert np.isfinite(direct).all()
    assert np.all((direct >= 0) & (direct <= 1))
    assert np.allclose(direct, reordered)
    assert np.allclose(direct, restored.predict_score(frame, _schema()))
    assert restored.device == "cpu"
    assert gpu_restored.device == "gpu"
    assert "device" not in metadata
    if engine == "catboost":
        assert backend.train_rows == frame.height
        assert backend.model.get_params()["learning_rate"] == 0.02
    else:
        assert backend.model.get_booster().feature_types == ["c", "float"]


@pytest.mark.parametrize("engine", ["catboost", "xgboost"])
def test_unknown_category_is_handled_at_inference(engine):
    pytest.importorskip(engine)
    frame = _frame()
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    params = {"iterations": 6, "depth": 2} if engine == "catboost" else {"n_estimators": 6, "max_depth": 2}
    backend = BinaryBoostingBackend(engine, params, random_state=42, device="cpu")
    backend.fit(frame, target, _schema(), valid_frame=frame, valid_target=target)

    scores = backend.predict_score(pl.DataFrame({"segment": ["new"], "balance": [0.0]}), _schema())

    assert scores.shape == (1,)
    assert 0 <= scores[0] <= 1


@pytest.mark.parametrize("train_has_null", [False, True])
def test_xgboost_polars_categories_match_pandas_on_synthetic_data(tmp_path, train_has_null):
    pytest.importorskip("xgboost")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(42)
    segments = ["a", "b", "c", None if train_has_null else "d"] * 64
    frame = pl.DataFrame({"segment": segments, "balance": rng.normal(size=256)})
    target = np.asarray([segment in ("b", "c") for segment in segments], dtype=int)
    train, valid = frame[:192], frame[192:]
    backend = BinaryBoostingBackend("xgboost", {"n_estimators": 12, "max_depth": 2}, 42, "cpu", verbose=False)
    prepared = backend.prepare_fit_data(train, target[:192], _schema(), valid_frame=valid, valid_target=target[192:])
    assert isinstance(prepared.train_features, pl.DataFrame)
    assert isinstance(prepared.valid_features, pl.DataFrame)
    assert prepared.train_features["segment"].dtype == prepared.valid_features["segment"].dtype
    assert isinstance(prepared.train_features["segment"].dtype, pl.Enum)
    backend.fit_prepared(prepared)

    categories = [*backend.category_values["segment"], "__FMLIB_UNKNOWN__"]

    def pandas_features(source):
        result = source.to_pandas()
        values = result["segment"].fillna("__FMLIB_NULL__")
        values = values.where(values.isin(backend.category_values["segment"]), "__FMLIB_UNKNOWN__")
        result["segment"] = pd.Categorical(values, categories=categories)
        return result

    reference = backend._make_model()
    reference.fit(
        pandas_features(train),
        target[:192],
        eval_set=[(pandas_features(valid), target[192:])],
        verbose=False,
    )
    test = pl.DataFrame({"segment": ["c", "new", None, "a", "b"], "balance": [0.0] * 5})
    scores = backend.predict_score(test.select("balance", "segment"), _schema())
    np.testing.assert_allclose(scores, reference.predict_proba(pandas_features(test))[:, 1])
    assert scores[0] > scores[3]
    artifact = tmp_path / "polars"
    backend.save(artifact)
    restored = BinaryBoostingBackend.load(artifact)
    np.testing.assert_allclose(scores, restored.predict_score(test, _schema()))


def test_device_and_default_runtime_parameters_are_forwarded_to_native_boosters():
    catboost_gpu = BinaryBoostingBackend("catboost", {}, 42, "gpu")._make_model()
    catboost_cpu = BinaryBoostingBackend("catboost", {}, 42, "cpu")._make_model()
    xgboost_gpu = BinaryBoostingBackend("xgboost", {}, 42, "gpu")._make_model()

    assert catboost_gpu.get_params()["task_type"] == "GPU"
    assert catboost_gpu.get_params()["allow_writing_files"] is False
    assert catboost_gpu.get_params()["num_trees"] == 3000
    assert catboost_gpu.get_params()["max_depth"] == 5
    assert catboost_gpu.get_params()["learning_rate"] == 0.03
    assert catboost_gpu.get_params()["od_wait"] == 100
    assert catboost_cpu.get_params()["task_type"] == "CPU"
    assert xgboost_gpu.get_params()["device"] == "cuda"
    assert xgboost_gpu.get_params()["n_estimators"] == 3000
    assert xgboost_gpu.get_params()["tree_method"] == "hist"
    assert xgboost_gpu.get_params()["enable_categorical"] is True
    assert xgboost_gpu.get_params()["verbosity"] == 0
    assert xgboost_gpu.get_params()["early_stopping_rounds"] == 100


def test_catboost_dataset_defaults_preserve_explicit_native_overrides():
    backend = BinaryBoostingBackend(
        engine="catboost",
        params={"learning_rate": 0.2, "iterations": 12},
        random_state=42,
        device="cpu",
        train_rows=6_000,
    )

    params = backend._make_model().get_params()

    assert params["learning_rate"] == 0.2
    assert params["iterations"] == 12
    assert "num_trees" not in params


def test_catboost_prediction_always_uses_cpu():
    class FakeEstimator:
        def __init__(self):
            self.task_types = []

        def predict_proba(self, features, *, task_type):
            self.task_types.append(task_type)
            return np.column_stack((np.zeros(len(features)), np.ones(len(features))))

    backend = BinaryBoostingBackend("catboost", {}, 42, "gpu")
    backend.model = FakeEstimator()
    features = pl.DataFrame({"balance": [1.0]})

    backend.predict_prepared_score(features)
    backend.set_runtime_device("cpu")
    backend.predict_prepared_score(features)

    assert backend.model.task_types == ["CPU", "CPU"]


def test_xgboost_prediction_keeps_gpu_runtime_device():
    class FakeEstimator:
        def __init__(self):
            self.params = []

        def set_params(self, **params):
            self.params.append(params)

        def predict_proba(self, features):
            return np.column_stack((np.zeros(len(features)), np.ones(len(features))))

    backend = BinaryBoostingBackend("xgboost", {}, 42, "cpu")
    backend.model = FakeEstimator()

    backend.set_runtime_device("gpu")
    backend.predict_prepared_score(pl.DataFrame({"balance": [1.0]}))

    assert backend.model.params == [{"device": "cuda"}]


def test_catboost_feature_importance_uses_native_method_instead_of_scalar_property():
    class FakeEstimator:
        feature_importances_ = np.asarray(50.0)

        def get_feature_importance(self, **kwargs):
            assert kwargs == {"type": "FeatureImportance"}
            return np.array([25.0, 75.0])

    backend = BinaryBoostingBackend("catboost", {}, 42, "cpu")
    backend.model = FakeEstimator()

    result = backend.feature_importance(_schema())

    assert result is not None
    assert result.to_dict(as_series=False) == {
        "feature": ["balance", "segment"],
        "importance": [75.0, 25.0],
    }


@pytest.mark.parametrize(
    ("search_space", "expected"),
    [
        (None, True),
        ({"depth": {"type": "int", "low": 2, "high": 4}}, True),
        ({"border_count": {"type": "int", "low": 32, "high": 64}}, False),
    ],
)
def test_catboost_prequantization_is_disabled_only_for_tuned_quantization(search_space, expected):
    assert BinaryBoostingBackend("catboost", {}, 42, "cpu").can_prequantize(search_space) is expected
    assert BinaryBoostingBackend("xgboost", {}, 42, "cpu").can_prequantize(search_space) is False


def test_backend_rejects_unknown_engine_without_implicit_fallback():
    with pytest.raises(UnsupportedBackendError, match="Unsupported boosting engine"):
        BinaryBoostingBackend("unknown", {}, 42, "cpu")._make_model()


def test_unfitted_backend_cannot_be_saved(tmp_path):
    with pytest.raises(ArtifactIntegrityError, match="unfitted"):
        BinaryBoostingBackend("catboost", {}, 42, "cpu").save(tmp_path / "model")
