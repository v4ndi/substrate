import numpy as np
import polars as pl
import pytest

from fmlib.automl.backends.boosting import RegressionBoostingBackend
from fmlib.automl.data import FeatureSchema
from fmlib.automl.exceptions import UnsupportedBackendError


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


@pytest.mark.parametrize(
    ("engine", "params"),
    [
        ("catboost", {"iterations": 12, "depth": 2}),
        ("xgboost", {"n_estimators": 12, "max_depth": 2}),
    ],
)
def test_regression_backend_fit_predict_and_native_round_trip(tmp_path, engine, params):
    pytest.importorskip(engine)
    frame = pl.DataFrame({"segment": ["a", "b"] * 8, "balance": np.linspace(-2, 2, 16)})
    target = 2.5 * frame["balance"].to_numpy() + np.tile([0.2, -0.1], 8)
    backend = RegressionBoostingBackend(engine, params, 42, "cpu", verbose=False)
    backend.fit(frame, target, _schema(), valid_frame=frame, valid_target=target)

    direct = backend.predict_score(frame, _schema())
    path = tmp_path / engine
    backend.save(path)
    restored = RegressionBoostingBackend.load(path)

    assert direct.shape == (16,)
    assert np.isfinite(direct).all()
    assert np.allclose(direct, restored.predict_score(frame, _schema()))


def test_regression_backend_maps_devices_and_rejects_unknown_engine():
    cat = RegressionBoostingBackend("catboost", {}, 42, "gpu")._make_model()
    assert cat.get_params()["task_type"] == "GPU"
    assert cat.get_params()["boost_from_average"] is True
    xgb = RegressionBoostingBackend("xgboost", {}, 42, "cpu")._make_model()
    assert xgb.get_params()["device"] == "cpu"
    assert xgb.get_params()["n_estimators"] == 3000
    assert xgb.get_params()["early_stopping_rounds"] == 100
    with pytest.raises(UnsupportedBackendError, match="Unsupported boosting engine"):
        RegressionBoostingBackend("unknown", {}, 42, "cpu")._make_model()
