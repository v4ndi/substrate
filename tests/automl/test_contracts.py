"""T1: every document that crosses a boundary, frozen as a reviewable snapshot.

A document written by one process and read by another -- or by another machine,
or by this machine tomorrow -- is an interface. Changing it is allowed; changing
it *by accident* is the failure this module exists to prevent, and it is exactly
what happened when ``TrialSpec.write`` started storing a config's ``repr``: the
producer kept working, every in-process test stayed green, and only the readers
broke.

So the rule these tests follow is the one that bug taught:

    the input is produced by production, never constructed by the test.

Each test drives the real producer, normalises what it wrote and compares it to
a committed snapshot in ``contracts/``. Re-record with::

    AUTOML_CONTRACT_RECORD=1 pytest -m "" tests/automl/test_contracts.py

Re-recording is a reviewable diff. If a change was meant to be
contract-preserving, that diff must be empty; if it was not, the diff is the
description of what downstream readers now have to handle.

**Numbers are normalised away on purpose.** Floats become ``<float>``: what a
model scored is the parity gate's question, and freezing it here would make
these files churn on every retrain and teach everyone to re-record without
reading. Integers stay, because in these documents they are configuration --
``num_gpus``, ``batch_size``, ``random_state`` -- not measurements.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from automl.scheduler_stub import LocalScheduler
from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.backends.tabnn.data import MANIFEST_NAME
from fmlib.automl.environment import EnvironmentRunner

CONTRACTS = Path(__file__).parent / "contracts"
RECORD = os.environ.get("AUTOML_CONTRACT_RECORD") == "1"
REPO_ROOT = Path(__file__).resolve().parents[2]

_HEX32 = re.compile(r"\b[0-9a-f]{32}\b")
_HEX8 = re.compile(r"\b[0-9a-f]{8}\b")


# --------------------------------------------------------------------------- #
# Normalisation                                                                #
# --------------------------------------------------------------------------- #
def _scrub(text: str, roots: dict[str, Path]) -> str:
    for label, root in roots.items():
        text = text.replace(str(root), f"<{label}>")
    text = _HEX32.sub("<hex32>", text)
    return _HEX8.sub("<hex8>", text)


#: Keys whose value is a fact about *this* run rather than about the contract.
#: Listed one by one rather than guessed from the value, so adding a volatile
#: field is a deliberate edit and not something a heuristic quietly swallows.
VOLATILE_KEYS = frozenset({
    "modified_ns",  # the source file's mtime: different on every checkout
})


def _normalize(value, roots: dict[str, Path], key: str | None = None):
    """Strip what changes between runs, keep what a reader depends on."""
    if key in VOLATILE_KEYS:
        return "<volatile>"
    if isinstance(value, dict):
        return {str(k): _normalize(v, roots, str(k)) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_normalize(v, roots, key) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return "<float>"
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return _scrub(value, roots)
    # Anything else reaching a document means a `default=str` fired somewhere,
    # which is the bug class this module is about. Say so loudly.
    return f"<non-json:{type(value).__name__}>"


def _assert_contract(name: str, document, roots: dict[str, Path]) -> None:
    normalized = _normalize(document, roots)
    path = CONTRACTS / f"{name}.json"
    if RECORD:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return
    if not path.exists():
        pytest.fail(
            f"no snapshot for {name!r}; record it with "
            "AUTOML_CONTRACT_RECORD=1 pytest -m '' tests/automl/test_contracts.py"
        )
    assert normalized == json.loads(path.read_text(encoding="utf-8"))


def _tree(root: Path) -> list[str]:
    """Sorted relative paths of everything under ``root`` -- the layout itself."""
    return sorted(
        str(item.relative_to(root)) for item in root.rglob("*") if item.is_file()
    )


# --------------------------------------------------------------------------- #
# Producers                                                                    #
# --------------------------------------------------------------------------- #
def _data(path: Path, rows: int = 200, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    pl.DataFrame({
        "epk_id": np.arange(rows),
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "group": rng.choice(["a", "b"], rows),
        "feature": rng.normal(size=rows),
        "target": rng.integers(0, 2, rows),
    }).write_parquet(path)
    return path


def _osiris_config(tmp_path: Path, **overrides) -> BinaryTaskConfig:
    venv = tmp_path / "venv"
    venv.mkdir(exist_ok=True)
    values = {
        "env_type": "osiris",
        "backend": "boosting",
        "engine": "catboost",
        "device": "gpu",
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": "group",
        "date_column": "report_month",
        "categorical_columns": (),
        "numerical_columns": ["feature"],
        "hidden_state_columns": (),
        "model_layout": "global",
        "hyperopt": False,
        "model_params": {"iterations": 5, "depth": 2, "thread_count": 1},
        "verbose": False,
        "output_dir": tmp_path / "outputs",
        "environment": {"venv_path": str(venv), "pool": "common"},
    }
    values.update(overrides)
    return BinaryTaskConfig(**values)


# --------------------------------------------------------------------------- #
# driver -> cluster                                                            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def submitted(tmp_path):
    """One real submit, nothing executed: the documents, not the outcome."""
    train = _data(tmp_path / "train.parquet")
    scheduler = LocalScheduler(execute=False)
    task = BinaryTask(_osiris_config(tmp_path))
    task._environment_runner = EnvironmentRunner(scheduler)
    task.train(train, train)
    return task, scheduler, {"tmp": tmp_path, "repo": REPO_ROOT}


def test_the_run_spec_handed_to_the_cluster(submitted):
    """What `python -m fmlib.automl.run` is given to reconstruct the work."""
    task, _, roots = submitted
    operation = task._store.latest("train")
    spec_path = Path(operation["jobs"][0]["spec_path"])
    _assert_contract(
        "osiris_run_spec", json.loads(spec_path.read_text(encoding="utf-8")), roots
    )


def test_the_request_the_scheduler_receives(submitted):
    """Image, command, resources, environment -- the job as Osiris sees it."""
    _, scheduler, roots = submitted
    assert len(scheduler.create_calls) == 1
    _assert_contract("osiris_create_request", scheduler.create_calls[0], roots)


def test_the_submit_log_left_behind(submitted):
    """The only record of a submit that a later process can read."""
    task, _, roots = submitted
    operation = task._store.latest("train")
    log_path = Path(operation["jobs"][0]["submit_log_path"])
    _assert_contract(
        "osiris_submit_log", json.loads(log_path.read_text(encoding="utf-8")), roots
    )


def test_the_job_handle_kept_on_the_operation_record(submitted):
    """What survives a driver restart and makes an in-flight job findable."""
    task, _, roots = submitted
    operation = task._store.latest("train")
    _assert_contract("osiris_job_handle", operation["jobs"][0], roots)


# --------------------------------------------------------------------------- #
# crash -> recovery                                                            #
# --------------------------------------------------------------------------- #
def test_the_operation_record(submitted):
    """The document that decides whether an interrupted run can be resumed."""
    task, _, roots = submitted
    record = dict(task._store.latest("train"))
    # Jobs have their own snapshot; keeping them here would duplicate the diff.
    record["jobs"] = f"<{len(record['jobs'])} job(s), see osiris_job_handle>"
    _assert_contract("operation_record", record, roots)


# --------------------------------------------------------------------------- #
# training -> prediction                                                       #
# --------------------------------------------------------------------------- #
def test_the_saved_boosting_artifact(tmp_path):
    """Four files and a metadata document are what `load` is promised."""
    train = _data(tmp_path / "train.parquet")
    config = _osiris_config(tmp_path, env_type="local", device="cpu")
    task = BinaryTask(config)
    task.train(train, train)
    artifact = task.save(tmp_path / "artifact")

    roots = {"tmp": tmp_path, "repo": REPO_ROOT}
    _assert_contract("boosting_artifact_layout", _tree(Path(artifact)), roots)
    metadata = sorted(Path(artifact).rglob("backend.json"))
    assert metadata, "the artifact carries no backend metadata"
    _assert_contract(
        "boosting_backend_metadata",
        json.loads(metadata[0].read_text(encoding="utf-8")),
        roots,
    )


# --------------------------------------------------------------------------- #
# driver -> rank, and encoding -> reading                                      #
# --------------------------------------------------------------------------- #
TABNN_PARAMS = {
    "max_epochs": 1,
    "batch_size": 128,
    "hidden_size": 16,
    "num_layers": 1,
    "num_heads": 2,
    "num_workers": 0,
    "evaluations_per_epoch": 1,
    "patience": 1,
}


@pytest.fixture
def tabnn_trained(tmp_path):
    """One real tabnn run: the spec, the result, the artifact, the encoding."""
    train = _data(tmp_path / "train.parquet", rows=400, seed=1)
    valid = _data(tmp_path / "valid.parquet", rows=200, seed=2)
    config = _osiris_config(
        tmp_path,
        env_type="local",
        device="cpu",
        backend="tabnn",
        engine="tabular_transformer",
        group_column=None,
        model_layout=None,
        model_params=dict(TABNN_PARAMS),
    )
    task = BinaryTask(config)
    task.train(train, valid)
    artifact = task.save(tmp_path / "artifact")
    return task, Path(artifact), {"tmp": tmp_path, "repo": REPO_ROOT}


@pytest.mark.slow
def test_the_spec_a_rank_is_handed(tabnn_trained):
    """Not the Osiris document of the same name -- the one a trial runs from.

    This is the file that held a config's `repr` for the whole of S11-S15. Its
    shape is frozen here so the next time it changes, the change is a diff.
    """
    _, _, roots = tabnn_trained
    specs = sorted(Path(roots["tmp"]).rglob("tabnn/**/run_spec.json"))
    assert len(specs) == 1, f"expected one trial spec, found {specs}"
    _assert_contract(
        "tabnn_trial_spec", json.loads(specs[0].read_text(encoding="utf-8")), roots
    )


@pytest.mark.slow
def test_what_a_rank_reports_back(tabnn_trained):
    """The only thing the driver learns from a trial it did not run."""
    _, _, roots = tabnn_trained
    results = sorted(Path(roots["tmp"]).rglob("tabnn/**/result.json"))
    assert len(results) == 1, f"expected one trial result, found {results}"
    _assert_contract(
        "tabnn_trial_result", json.loads(results[0].read_text(encoding="utf-8")), roots
    )


@pytest.mark.slow
def test_the_completion_marker_of_processed_data(tabnn_trained):
    """Written last, on purpose: its presence is what makes a directory usable."""
    _, _, roots = tabnn_trained
    markers = sorted(Path(roots["tmp"]).rglob(f"processed/*/{MANIFEST_NAME}"))
    assert len(markers) == 1, f"expected one completion marker, found {markers}"
    _assert_contract(
        "tabnn_processed_manifest",
        json.loads(markers[0].read_text(encoding="utf-8")),
        roots,
    )
    # The layout the marker certifies: a reader resolves splits by these paths.
    _assert_contract("tabnn_processed_layout", _tree(markers[0].parent), roots)


@pytest.mark.slow
def test_the_saved_tabnn_artifact(tabnn_trained):
    """Metadata, train config, preprocessor, weights -- and no pickle."""
    _, artifact, roots = tabnn_trained
    layout = _tree(artifact)
    _assert_contract("tabnn_artifact_layout", layout, roots)
    assert not [name for name in layout if name.endswith((".pt", ".bin", ".pkl"))], (
        "artifact weights must be safetensors, not a pickled state dict"
    )
    metadata = sorted(artifact.rglob("backend.json"))
    assert metadata, "the artifact carries no backend metadata"
    _assert_contract(
        "tabnn_backend_metadata",
        json.loads(metadata[0].read_text(encoding="utf-8")),
        roots,
    )
