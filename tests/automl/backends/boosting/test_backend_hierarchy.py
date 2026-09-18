import inspect

import pytest

from fmlib.automl.backends.boosting import (
    BinaryBoostingBackend,
    MulticlassBoostingBackend,
    RegressionBoostingBackend,
    UpliftBoostingBackend,
)
from fmlib.automl.backends.boosting.base import BaseBoostingBackend
from fmlib.automl.backends.boosting.interface import BoostingBackend
from fmlib.automl.exceptions import ConfigError


def test_boosting_backends_are_independent_siblings():
    """Estimator adapters for distinct formulations must not inherit each other."""
    assert BinaryBoostingBackend.__bases__ == (BaseBoostingBackend,)
    assert RegressionBoostingBackend.__bases__ == (BaseBoostingBackend,)
    assert MulticlassBoostingBackend.__bases__ == (BaseBoostingBackend,)
    assert issubclass(BaseBoostingBackend, BoostingBackend)
    assert issubclass(UpliftBoostingBackend, BoostingBackend)
    assert not issubclass(UpliftBoostingBackend, BaseBoostingBackend)
    assert inspect.isabstract(BaseBoostingBackend)
    assert not issubclass(RegressionBoostingBackend, BinaryBoostingBackend)
    assert not issubclass(BinaryBoostingBackend, RegressionBoostingBackend)
    assert not issubclass(
        MulticlassBoostingBackend, BinaryBoostingBackend | RegressionBoostingBackend
    )
    assert not issubclass(
        UpliftBoostingBackend, BinaryBoostingBackend | RegressionBoostingBackend
    )


def test_composite_exposes_only_supported_operations():
    backend = UpliftBoostingBackend(
        engine="xgboost", params={}, random_state=42, device="cpu"
    )
    for name in (
        "fit",
        "fit_prepared",
        "prepare_fit_data",
        "predict_prepared_score",
        "_make_model",
    ):
        assert not hasattr(backend, name)
    assert callable(backend.fit_composite)
    assert not inspect.isabstract(UpliftBoostingBackend)


@pytest.mark.parametrize(
    "backend_class",
    [BinaryBoostingBackend, RegressionBoostingBackend, UpliftBoostingBackend],
)
def test_common_runtime_device_validation_preserves_state(backend_class):
    backend = backend_class(engine="xgboost", params={}, random_state=42, device="cpu")
    with pytest.raises(ConfigError, match="Unsupported boosting runtime device"):
        backend.set_runtime_device("invalid")
    assert backend.device == "cpu"


def test_composite_runtime_device_reaches_native_components():
    class NativeModel:
        def set_params(self, **params):
            self.params = params

    component = BinaryBoostingBackend(
        engine="xgboost", params={}, random_state=42, device="cpu", model=NativeModel()
    )
    backend = UpliftBoostingBackend(
        engine="xgboost",
        params={},
        random_state=42,
        device="cpu",
        components={"s_outcome": component},
    )
    backend.set_runtime_device("gpu")
    assert backend.device == component.device == "gpu"
    assert component.model.params == {"device": "cuda"}
    backend.set_runtime_device("cpu")
    assert backend.device == component.device == "cpu"
    assert component.model.params == {"device": "cpu"}
