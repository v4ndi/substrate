import json

import numpy as np
import polars as pl
import pytest

from fmlib.automl.backends.boosting import MulticlassBoostingBackend
from fmlib.automl.backends.search import suggest_params
from fmlib.automl.data import FeatureSchema
from fmlib.automl.exceptions import ConfigError, UnsupportedBackendError


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


def _data():
    balance = np.linspace(-3, 3, 36)
    frame = pl.DataFrame({"segment": np.tile(["a", "b", "c"], 12), "balance": balance})
    target = np.tile(np.arange(3), 12)
    return frame, target


@pytest.mark.parametrize(
    ("engine", "params"),
    [
        ("catboost", {"iterations": 10, "depth": 2}),
        ("xgboost", {"n_estimators": 10, "max_depth": 2}),
    ],
)
def test_multiclass_backend_fit_predict_and_native_round_trip(tmp_path, engine, params):
    pytest.importorskip(engine)
    frame, target = _data()
    backend = MulticlassBoostingBackend(
        engine, params, 42, "cpu", verbose=False, num_classes=3
    )
    backend.fit(frame, target, _schema(), valid_frame=frame, valid_target=target)
    direct = backend.predict_score(frame, _schema())
    artifact = tmp_path / engine
    backend.save(artifact)
    restored = MulticlassBoostingBackend.load(artifact)
    metadata = json.loads((artifact / "backend.json").read_text(encoding="utf-8"))

    assert direct.shape == (frame.height, 3)
    assert np.isfinite(direct).all()
    assert np.all((direct >= 0) & (direct <= 1))
    assert np.allclose(direct.sum(axis=1), 1.0)
    assert np.allclose(direct, restored.predict_score(frame, _schema()))
    assert metadata["task_state"]["num_classes"] == 3
    assert (
        tuple(metadata["task_state"]["estimator_class_order"])
        == backend.estimator_class_order
    )


def test_estimator_probability_columns_are_aligned_to_global_encoded_order():
    class FakeEstimator:
        @staticmethod
        def predict_proba(_features, **_kwargs):
            return np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])

    backend = MulticlassBoostingBackend("catboost", {}, 42, "cpu", num_classes=3)
    backend.model = FakeEstimator()
    backend.estimator_class_order = (2, 0, 1)

    aligned = backend.predict_prepared_score(object())

    assert np.allclose(aligned, [[0.3, 0.1, 0.6], [0.5, 0.3, 0.2]])


def test_multiclass_backend_device_mapping_and_explicit_engine_errors():
    cat = MulticlassBoostingBackend(
        "catboost", {}, 42, "gpu", num_classes=3
    )._make_model()
    xgb = MulticlassBoostingBackend(
        "xgboost", {}, 42, "cpu", num_classes=4
    )._make_model()
    assert cat.get_params()["task_type"] == "GPU"
    assert cat.get_params()["loss_function"] == "MultiClass"
    assert "boost_from_average" not in cat.get_params()
    assert cat.get_params()["num_trees"] == 3000
    assert xgb.get_params()["device"] == "cpu"
    assert xgb.get_params()["n_estimators"] == 3000
    assert xgb.get_params()["objective"] == "multi:softprob"
    assert xgb.get_params()["num_class"] == 4
    assert xgb.get_params()["early_stopping_rounds"] == 100
    with pytest.raises(UnsupportedBackendError, match="Unsupported boosting engine"):
        MulticlassBoostingBackend("unknown", {}, 42, "cpu", num_classes=3)._make_model()


class _Trial:
    def suggest_categorical(self, _name, choices):
        return choices[0]

    def suggest_int(self, _name, low, _high, **kwargs):
        self.int_kwargs = kwargs
        return low

    def suggest_float(self, _name, low, _high, **kwargs):
        self.float_kwargs = kwargs
        return low


def test_multiclass_custom_search_space_supports_all_range_types_and_float_step():
    trial = _Trial()
    params = suggest_params(
        trial,
        engine="catboost",
        model_params={"iterations": 5},
        search_space={
            "depth": {"type": "int", "low": 3, "high": 7, "step": 2},
            "learning_rate": {"type": "float", "low": 0.1, "high": 0.3, "step": 0.1},
            "grow_policy": {
                "type": "categorical",
                "choices": ["SymmetricTree", "Depthwise"],
            },
        },
    )
    assert params == {
        "iterations": 5,
        "depth": 3,
        "learning_rate": 0.1,
        "grow_policy": "SymmetricTree",
    }
    assert trial.int_kwargs == {"step": 2, "log": False}
    assert trial.float_kwargs == {"step": 0.1, "log": False}


@pytest.mark.parametrize(
    ("engine", "n_trials", "integer_parameters"),
    [
        ("catboost", 51, ("max_depth", "min_data_in_leaf")),
        ("xgboost", 31, ("min_child_weight",)),
    ],
)
def test_default_search_space_suggests_integer_parameters_as_integers(
    engine, n_trials, integer_parameters
):
    params = suggest_params(
        _Trial(), engine=engine, model_params={}, search_space=None, n_trials=n_trials
    )

    assert all(isinstance(params[name], int) for name in integer_parameters)


def test_multiclass_log_and_step_validation_is_diagnostic():
    with pytest.raises(ConfigError, match="cannot combine"):
        suggest_params(
            _Trial(),
            engine="catboost",
            model_params={},
            search_space={
                "learning_rate": {
                    "type": "float",
                    "low": 0.1,
                    "high": 0.3,
                    "step": 0.1,
                    "log": True,
                }
            },
        )
