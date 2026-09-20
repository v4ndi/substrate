"""Every way an artifact can be wrong, and the refusal it earns.

`_validate_manifest` is the longest chain of guards in the package and was the
least exercised: 80% covered, with the uncovered part being all of the guards
and none of the happy path. That is the wrong way round. An artifact is read
long after it was written, usually by a different process and often by a
different person, and the failure mode these guards prevent is not a crash —
it is a model that loads and scores the wrong thing.

The table below tampers with one field at a time and asserts the message names
what is wrong. The artifact itself is produced by a real train and save, so the
starting point is what production writes rather than a hand-built manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.exceptions import ArtifactIntegrityError

MANIFEST = "manifest.json"


def _config(tmp_path: Path, **overrides) -> BinaryTaskConfig:
    values = {
        "output_dir": tmp_path / "out",
        "env_type": "local",
        "device": "cpu",
        "backend": "boosting",
        "engine": "catboost",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": None,
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ["feature"],
        "hidden_state_columns": (),
        "hyperopt": False,
        "verbose": False,
        "random_state": 42,
        "model_params": {"iterations": 10, "depth": 2, "thread_count": 1},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


@pytest.fixture
def artifact(tmp_path) -> Path:
    """A real artifact: trained and saved through the public API."""
    rng = np.random.default_rng(0)
    rows = 300
    feature = rng.normal(size=rows)
    data = tmp_path / "data.parquet"
    pl.DataFrame({
        "epk_id": np.arange(rows),
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "feature": feature,
        "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-2 * feature))).astype(
            np.int8
        ),
    }).write_parquet(data)

    task = BinaryTask(_config(tmp_path))
    task.train(data, data)
    return Path(task.save(tmp_path / "artifact"))


def _retouch(artifact: Path, mutate) -> None:
    path = artifact / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_an_untouched_artifact_loads(artifact):
    """The control: without it, every refusal below could be refusing anything."""
    task = BinaryTask.load(artifact)
    assert task.is_fitted


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(lambda m: m.update(task="regression"), "task mismatch", id="task"),
        pytest.param(
            lambda m: m.update(backend_family="tabnn"),
            "backend_family mismatch",
            id="backend-family",
        ),
        pytest.param(
            lambda m: m.update(engine="xgboost"), "engine mismatch", id="engine"
        ),
        pytest.param(lambda m: m.update(models=[]), "non-empty list", id="no-models"),
        pytest.param(
            lambda m: m.update(models={}), "non-empty list", id="models-not-a-list"
        ),
        pytest.param(
            lambda m: m.update(models=["not a mapping"]),
            "must be a mapping",
            id="model-not-a-mapping",
        ),
        pytest.param(
            lambda m: m["models"][0].update(scope="global"),
            "legacy model scope",
            id="legacy-scope",
        ),
        pytest.param(
            lambda m: m["models"][0].update(model_layout="per_client"),
            "model_layout is unsupported",
            id="unknown-layout",
        ),
        pytest.param(
            lambda m: m["models"][0].update(model_layout="per_group", group_value=None),
            "inconsistent model_layout",
            id="per-group-without-a-group",
        ),
        pytest.param(
            lambda m: m["models"][0].update(path=""),
            "must be a non-empty relative path",
            id="empty-path",
        ),
        pytest.param(
            lambda m: m["models"][0].update(path=123),
            "must be a non-empty relative path",
            id="path-not-a-string",
        ),
        pytest.param(
            lambda m: m["models"][0].update(path="../../elsewhere"),
            "escapes the artifact root",
            id="path-traversal",
        ),
        pytest.param(
            lambda m: m["models"][0].update(path="model/99"),
            "directory is missing or invalid",
            id="missing-directory",
        ),
        pytest.param(
            lambda m: m["models"][0].pop("validation_metric"),
            "metadata or schema is invalid",
            id="no-validation-metric",
        ),
        pytest.param(
            lambda m: m["models"][0].update(validation_metric="high"),
            "metadata or schema is invalid",
            id="validation-metric-not-a-number",
        ),
        pytest.param(
            lambda m: m["models"][0].update(hidden_dimensions=[]),
            "must be mappings",
            id="hidden-dimensions-not-a-mapping",
        ),
        pytest.param(
            lambda m: m["models"][0].update(best_params=None),
            "must be mappings",
            id="best-params-not-a-mapping",
        ),
    ],
)
def test_a_tampered_manifest_is_refused_by_name(artifact, mutate, expected):
    _retouch(artifact, mutate)
    with pytest.raises(ArtifactIntegrityError, match=expected):
        BinaryTask.load(artifact)


def test_a_duplicate_model_identity_is_refused(artifact):
    """Two models claiming the same slot: whichever wins, one is being ignored."""
    _retouch(artifact, lambda m: m["models"].append(dict(m["models"][0])))
    with pytest.raises(ArtifactIntegrityError, match="duplicate model identity"):
        BinaryTask.load(artifact)


def test_a_feature_schema_that_contradicts_itself_is_refused(artifact):
    """A column in both the categorical and the numerical list is not a schema."""

    def mutate(manifest):
        schema = manifest["models"][0]["schema"]
        schema["categorical"] = list(schema["numerical"])

    _retouch(artifact, mutate)
    with pytest.raises(ArtifactIntegrityError, match="feature schema is inconsistent"):
        BinaryTask.load(artifact)


def test_a_backend_file_that_disagrees_with_the_manifest_is_refused(artifact):
    """The manifest says catboost and the weights say otherwise: refuse, do not guess."""
    backend_path = next(artifact.rglob("backend.json"))
    metadata = json.loads(backend_path.read_text(encoding="utf-8"))
    metadata["engine"] = "xgboost"
    backend_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="backend engine mismatch"):
        BinaryTask.load(artifact)


def test_a_missing_backend_file_names_the_path(artifact):
    next(artifact.rglob("backend.json")).unlink()
    with pytest.raises(ArtifactIntegrityError, match="backend metadata is missing"):
        BinaryTask.load(artifact)


def test_an_unreadable_manifest_is_refused(artifact):
    (artifact / MANIFEST).write_text("{ not json", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        BinaryTask.load(artifact)
