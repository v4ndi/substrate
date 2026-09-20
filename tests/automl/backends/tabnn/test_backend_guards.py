"""What the TabNN backend refuses, and the empty-frame shortcut.

Four guards, all of them on the loading side, all of them protecting the same
thing: a backend that scores when it should not. An unfitted backend has no
weights; an artifact written by the boosting family is not a TabNN artifact;
an artifact missing one of its four files is not an artifact; and a model
trained for one task must not be asked to score another.

The empty frame is here for a different reason: it is the one case where the
backend returns without touching torch, so the *shape* of what it returns has
to be right on its own. A wrong shape there surfaces as a confusing error much
later, in whatever tries to concatenate the result.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from fmlib.automl.backends.tabnn.base import TabNNBackend
from fmlib.automl.data import FeatureSchema
from fmlib.automl.exceptions import ArtifactIntegrityError, NotFittedError


def _bare(**overrides) -> TabNNBackend:
    values = {
        "engine": "tabular_transformer",
        "task_name": "binary",
        "num_classes": 1,
        "score_transform": "sigmoid",
        "class_order": None,
        "column_names": {},
        "hidden_states": {},
        "batch_size": 128,
        "validation_metric": 0.5,
        "params": {},
        "random_state": 42,
        "train_config": {},
        "preprocessor_state": None,
        "state_dict": None,
    }
    values.update(overrides)
    return TabNNBackend(**values)


def test_a_backend_without_weights_says_so_rather_than_failing_in_torch():
    backend = _bare()
    with pytest.raises(NotFittedError, match="never fitted or loaded"):
        backend._model()


def test_a_backend_without_a_preprocessor_says_so():
    backend = _bare()
    with pytest.raises(NotFittedError, match="no fitted preprocessor"):
        backend._preprocessor()


def test_a_backend_without_weights_refuses_to_save():
    backend = _bare()
    with pytest.raises(NotFittedError, match="no weights to save"):
        backend.save(pytest.importorskip("pathlib").Path("/tmp/never-written"))


@pytest.mark.parametrize(
    ("task_name", "asked"),
    [("binary", "regression"), ("multiclass", "binary"), ("uplift", "response")],
)
def test_an_artifact_refuses_to_be_used_for_another_task(task_name, asked):
    """Scoring a regression artifact as a classifier would return numbers."""
    backend = _bare(task_name=task_name)
    with pytest.raises(ArtifactIntegrityError, match=f"not {asked!r}"):
        backend.expect_task(asked)


def test_expect_task_accepts_its_own_task():
    _bare(task_name="binary").expect_task("binary")


def test_loading_a_directory_without_metadata_names_the_missing_file(tmp_path):
    with pytest.raises(ArtifactIntegrityError, match="missing backend.json"):
        TabNNBackend.load(tmp_path)


def test_loading_another_family_s_artifact_is_refused(tmp_path):
    """The boosting artifact has a backend.json too, and it is not this one."""
    (tmp_path / "backend.json").write_text(
        json.dumps({"backend": "boosting", "engine": "catboost"}), encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="written by backend 'boosting'"):
        TabNNBackend.load(tmp_path)


@pytest.mark.parametrize(
    "missing", ["train_config.yaml", "preprocessor.yaml", "model.safetensors"]
)
def test_loading_an_incomplete_artifact_names_the_file_that_is_absent(
    tmp_path, missing
):
    """Which file is missing is the whole content of this error message."""
    (tmp_path / "backend.json").write_text(
        json.dumps({
            "backend": "tabnn",
            "engine": "tabular_transformer",
            "task_name": "binary",
            "num_classes": 1,
            "score_transform": "sigmoid",
            "class_order": None,
            "column_names": {},
            "hidden_states": {},
            "batch_size": 128,
            "validation_metric": 0.5,
            "params": {},
            "random_state": 42,
        }),
        encoding="utf-8",
    )
    for name in ("train_config.yaml", "preprocessor.yaml", "model.safetensors"):
        if name != missing:
            (tmp_path / name).write_bytes(b"placeholder")

    with pytest.raises(ArtifactIntegrityError, match=f"missing {missing}"):
        TabNNBackend.load(tmp_path)


@pytest.mark.parametrize(
    ("transform", "num_classes", "expected_shape"),
    [
        pytest.param("sigmoid", 1, (0,), id="binary"),
        pytest.param("identity", 1, (0,), id="regression"),
        pytest.param("softmax", 4, (0, 4), id="multiclass"),
    ],
)
def test_an_empty_frame_returns_the_right_shape_without_touching_the_model(
    transform, num_classes, expected_shape
):
    """No rows still has a width, and the width is what the caller stacks on."""
    backend = _bare(score_transform=transform, num_classes=num_classes)
    schema = FeatureSchema(
        feature_order=("feature",),
        categorical=(),
        numerical=("feature",),
        dtypes={"feature": "Float64"},
        target_column="target",
        client_id_column="epk_id",
        group_column=None,
        treatment_column=None,
        hidden_states={},
    )
    scores = backend.predict_score(
        pl.DataFrame({"feature": pl.Series("feature", [], pl.Float64)}), schema
    )

    assert scores.shape == expected_shape
    assert scores.dtype == np.float64
