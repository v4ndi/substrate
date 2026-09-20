"""Reusing encoded data is only safe if the marker is checked, so check the check.

Processed data is addressed by a content key and certified by a completion
marker written last. The point of the marker is that an interrupted encoding
leaves a directory that is *not* mistaken for a finished one, and that data
written by a different version of the contract is *not* silently read by this
one. Both failures are silent by nature: the run would train on whatever is
there and report a number.

The refusals therefore matter more than the happy path, and the plan named the
contract-version one specifically. Every message here has to say what to do,
because the user's next move is to delete a directory they may have paid for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fmlib.automl.backends.tabnn.data import (
    CONTRACT_VERSION,
    MANIFEST_NAME,
    _read_manifest,
    _validate,
)
from fmlib.automl.exceptions import ArtifactIntegrityError, SchemaError

KEY = "a" * 32


@pytest.fixture
def processed(tmp_path) -> Path:
    """A directory shaped like finished processed data."""
    root = tmp_path / "processed" / KEY
    for split in ("train", "valid"):
        (root / split).mkdir(parents=True)
        pl.DataFrame({"a": [1, 2]}).write_parquet(root / split / "part-00000.parquet")
    (root / MANIFEST_NAME).write_text(
        json.dumps({
            "complete": True,
            "contract_version": CONTRACT_VERSION,
            "key": KEY,
            "train": {"parts": ["part-00000.parquet"], "rows": 2},
            "valid": {"parts": ["part-00000.parquet"], "rows": 2},
        }),
        encoding="utf-8",
    )
    return root


def test_finished_data_is_accepted(processed):
    """The control, without which every refusal below could be refusing anything."""
    _validate(processed, _read_manifest(processed), KEY)


def test_an_interrupted_encoding_is_not_mistaken_for_a_finished_one(processed):
    """`complete` is written last; absent, the directory is a half-written run."""

    def mutate(manifest):
        manifest["complete"] = False

    _retouch(processed, mutate)
    with pytest.raises(
        ArtifactIntegrityError, match="does not say the encoding finished"
    ):
        _validate(processed, _read_manifest(processed), KEY)


@pytest.mark.parametrize("version", ["0", "2", "", None])
def test_data_written_under_another_contract_is_refused(processed, version):
    """The layout may have changed; reading it as if it had not is the failure."""

    def mutate(manifest):
        manifest["contract_version"] = version

    _retouch(processed, mutate)
    with pytest.raises(ArtifactIntegrityError, match="contract version"):
        _validate(processed, _read_manifest(processed), KEY)


def test_data_encoded_for_other_inputs_is_refused(processed):
    """The key is the identity of the inputs: a mismatch means different data."""
    with pytest.raises(ArtifactIntegrityError, match="marker names key"):
        _validate(processed, _read_manifest(processed), "b" * 32)


def test_a_part_the_marker_lists_but_does_not_exist_is_refused(processed):
    (processed / "train" / "part-00000.parquet").unlink()
    with pytest.raises(ArtifactIntegrityError, match="train/part-00000.parquet"):
        _validate(processed, _read_manifest(processed), KEY)


def test_every_problem_is_reported_at_once(processed):
    """One message, not a discover-fix-rerun loop over an expensive encoding."""

    def mutate(manifest):
        manifest["complete"] = False
        manifest["contract_version"] = "0"

    _retouch(processed, mutate)
    (processed / "valid" / "part-00000.parquet").unlink()

    with pytest.raises(ArtifactIntegrityError) as error:
        _validate(processed, _read_manifest(processed), "b" * 32)

    message = str(error.value)
    for expected in (
        "does not say the encoding finished",
        "contract version",
        "marker names key",
        "valid/part-00000.parquet",
    ):
        assert expected in message
    # And it says what to do about it.
    assert "Nothing was overwritten" in message
    assert "processed_data_path" in message


def test_a_directory_with_no_marker_reads_as_absent(tmp_path):
    """No marker is not an error: it is an interrupted run, to be re-encoded."""
    empty = tmp_path / "processed" / KEY
    empty.mkdir(parents=True)
    assert _read_manifest(empty) is None


def test_an_unreadable_marker_says_what_to_do(tmp_path):
    """Corrupt is different from absent, and the user has to be told which."""
    root = tmp_path / "processed" / KEY
    root.mkdir(parents=True)
    (root / MANIFEST_NAME).write_text("{ truncated", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError) as error:
        _read_manifest(root)
    assert "unreadable completion marker" in str(error.value)
    assert "processed_data_path" in str(error.value)


def _retouch(root: Path, mutate) -> None:
    path = root / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Schema refusals reached from a config                                        #
# --------------------------------------------------------------------------- #
def test_a_configuration_with_no_features_at_all_is_refused(tmp_path):
    """Three empty lists is a config mistake, not an empty model."""
    from fmlib.automl import BinaryTask, BinaryTaskConfig

    rows = 100
    rng = np.random.default_rng(0)
    data = tmp_path / "data.parquet"
    pl.DataFrame({
        "epk_id": np.arange(rows),
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "target": rng.integers(0, 2, rows).astype(np.int8),
    }).write_parquet(data)

    task = BinaryTask(
        BinaryTaskConfig(
            output_dir=tmp_path / "out",
            env_type="local",
            device="cpu",
            backend="tabnn",
            engine="tabular_transformer",
            target_column="target",
            client_id_column="epk_id",
            group_column=None,
            date_column="report_month",
            categorical_columns=(),
            numerical_columns=(),
            hidden_state_columns=(),
            hyperopt=False,
            verbose=False,
            model_params={"max_epochs": 1, "batch_size": 32, "num_workers": 0},
        )
    )
    with pytest.raises(SchemaError, match="No features configured"):
        task.train(data, data)
