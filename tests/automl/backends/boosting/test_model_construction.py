"""How each backend builds its native estimator, including the paths we cannot fit.

Three things were untested here and all three are reachable from a config the
user writes.

The `device="gpu"` branch is one of them. Building an estimator is not the same
as training on one, so what the flag turns into can be asserted on a machine
with no card: CatBoost gets `task_type="GPU"`, XGBoost gets `device="cuda"`. A
run whose device silently fell back to CPU would still produce numbers, just
slowly, and nobody would find out from the metrics.

The refusals are the other two: a device name that is neither, and a multiclass
backend handed fewer than three classes.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from fmlib.automl.backends.boosting import (
    BinaryBoostingBackend,
    MulticlassBoostingBackend,
    RegressionBoostingBackend,
)
from fmlib.automl.exceptions import (
    ArtifactIntegrityError,
    MissingDependencyError,
    UnsupportedBackendError,
)

ENGINES = ("catboost", "xgboost")


def _backend(backend_class, engine: str, device: str = "cpu", **extra):
    kwargs = {
        "engine": engine,
        "params": {},
        "random_state": 42,
        "device": device,
        "verbose": False,
    }
    if backend_class is MulticlassBoostingBackend:
        kwargs.setdefault("num_classes", 3)
    kwargs.update(extra)
    return backend_class(**kwargs)


BACKENDS = (
    BinaryBoostingBackend,
    RegressionBoostingBackend,
    MulticlassBoostingBackend,
)


@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("device", ["tpu", "GPU", "", None])
def test_an_unknown_device_is_refused_before_anything_is_built(
    backend_class, engine, device
):
    """Including 'GPU': the accepted spelling is lowercase, and near-misses are
    the ones that would otherwise run on the wrong hardware."""
    with pytest.raises(
        UnsupportedBackendError, match="Unsupported local boosting device"
    ):
        _backend(backend_class, engine, device=device)._make_model()


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_catboost_is_asked_for_the_gpu_when_the_device_says_so(backend_class):
    """Construction only -- no card needed to check what the flag became."""
    model = _backend(backend_class, "catboost", device="gpu")._make_model()
    assert model.get_params()["task_type"] == "GPU"

    model = _backend(backend_class, "catboost", device="cpu")._make_model()
    assert model.get_params()["task_type"] == "CPU"


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_xgboost_is_asked_for_the_gpu_when_the_device_says_so(backend_class):
    model = _backend(backend_class, "xgboost", device="gpu")._make_model()
    assert model.get_params()["device"] == "cuda"

    model = _backend(backend_class, "xgboost", device="cpu")._make_model()
    assert model.get_params()["device"] == "cpu"


@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("engine", ENGINES)
def test_the_seed_reaches_the_estimator(backend_class, engine):
    """A seed that does not arrive makes every reproducibility claim false."""
    model = _backend(backend_class, engine, random_state=1234)._make_model()
    parameters = model.get_params()
    seed = parameters.get("random_seed", parameters.get("random_state"))
    assert seed == 1234


@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize(
    ("engine", "package"), [("catboost", "catboost"), ("xgboost", "xgboost")]
)
def test_a_missing_engine_names_the_package_to_install(
    backend_class, engine, package, monkeypatch
):
    """The engines are optional extras, so this is a message a user will read."""
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == package:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(MissingDependencyError, match=package):
        _backend(backend_class, engine)._make_model()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("num_classes", [0, 1, 2])
def test_a_multiclass_backend_refuses_fewer_than_three_classes(engine, num_classes):
    """Two classes is the binary backend; one is not a classification problem."""
    backend = _backend(MulticlassBoostingBackend, engine, num_classes=num_classes)
    with pytest.raises(ArtifactIntegrityError, match="num_classes >= 3"):
        backend._make_model()


# --------------------------------------------------------------------------- #
# Multiclass probability alignment                                             #
# --------------------------------------------------------------------------- #
class _Estimator:
    """Stands in for a fitted native model returning a chosen probability block."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features, **kwargs):
        return self.probabilities


def _fitted(probabilities, class_order, num_classes=3):
    backend = _backend(MulticlassBoostingBackend, "catboost", num_classes=num_classes)
    backend.model = _Estimator(probabilities)
    backend.estimator_class_order = tuple(class_order)
    return backend


def test_probabilities_are_reordered_into_the_task_class_order():
    """The estimator's own class order is not the task's, and the gap is silent.

    A native model trained on labels it saw in a different order returns columns
    in *its* order. Mapping them back is the whole job of this method, and an
    off-by-one here relabels every prediction while every metric stays plausible.
    """
    backend = _fitted([[0.1, 0.6, 0.3]], class_order=(2, 0, 1))
    aligned = backend.predict_prepared_score(np.zeros((1, 1)))

    # column 0 of the estimator is task class 2, and so on.
    np.testing.assert_allclose(aligned, [[0.6, 0.3, 0.1]])


def test_rows_are_renormalised_to_sum_to_one():
    backend = _fitted([[0.2, 0.2, 0.2]], class_order=(0, 1, 2))
    aligned = backend.predict_prepared_score(np.zeros((1, 1)))
    assert aligned.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("probabilities", "class_order", "expected"),
    [
        pytest.param(
            [[0.5, 0.5]],
            (0, 1, 2),
            "inconsistent with its class mapping",
            id="too-few-columns",
        ),
        pytest.param(
            [0.3, 0.3, 0.4],
            (0, 1, 2),
            "inconsistent with its class mapping",
            id="one-dimensional",
        ),
        pytest.param(
            [[0.3, 0.3, 0.4]],
            (0, 1, 7),
            "unknown encoded class",
            id="class-out-of-range",
        ),
        pytest.param(
            [[0.3, np.nan, 0.4]],
            (0, 1, 2),
            "non-finite or out-of-range",
            id="nan-probability",
        ),
        pytest.param(
            [[-0.1, 0.7, 0.4]],
            (0, 1, 2),
            "non-finite or out-of-range",
            id="negative-probability",
        ),
        pytest.param([[0.0, 0.0, 0.0]], (0, 1, 2), "non-positive sum", id="empty-row"),
    ],
)
def test_an_estimator_that_contradicts_its_class_mapping_is_refused(
    probabilities, class_order, expected
):
    """Every one of these would otherwise become a score somebody acts on."""
    backend = _fitted(probabilities, class_order)
    with pytest.raises(ArtifactIntegrityError, match=expected):
        backend.predict_prepared_score(np.zeros((1, 1)))
