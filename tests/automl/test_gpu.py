"""What only a real card can answer.

Everything else in this suite runs on CPU, which is what makes it runnable
anywhere -- and what leaves the GPU path asserted only in construction. The
flag becoming `task_type="GPU"` is checked without a card in
`tests/automl/backends/boosting/test_model_construction.py`; that a model
*trained* that way comes back and scores is checked here.

Two things are worth the card specifically:

* mixed precision is chosen from the hardware, never searched, so bf16 on an
  Ampere-or-newer card is a claim about this machine and not about a mock;
* `CUDA_VISIBLE_DEVICES` is narrowed for the duration of a local boosting run
  and must be put back, including when the run raises. A notebook kernel that
  silently keeps one visible card after a failed cell is a bug nobody reports
  and everybody works around.

Deselected by default and skipped on a machine with no device, so this file is
never the reason a run is red somewhere else.

**What this does not cover.** This machine has one card, so nothing here says
anything about several. The multi-rank tests run on gloo with CPU ranks, which
proves the launch, the spec round trip, the agreement between ranks and the
sharding -- but not NCCL collectives, not how ranks are laid out over devices,
and not that two cards produce what one produces. That needs a second card, and
until there is one the claim stays unmade rather than assumed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.backends.tabnn.assembly import resolve_amp
from fmlib.automl.environment import _cuda_visible_devices, _local_cuda_visibility

pytestmark = pytest.mark.gpu


def _write(path: Path, rows: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=rows)
    other = rng.normal(size=rows)
    logit = 1.5 * feature - 0.6 * other
    pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "segment": pl.Series([str(v) for v in rng.choice(["a", "b", "c"], rows)]),
        "feature": feature,
        "other": other,
        "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
    }).write_parquet(path)
    return path


def _config(tmp_path: Path, backend: str, **overrides) -> BinaryTaskConfig:
    values: dict = {
        "output_dir": tmp_path / f"out_{backend}",
        "env_type": "local",
        "device": "gpu",
        "backend": backend,
        "target_column": "target",
        "client_id_column": "epk_id",
        "group_column": None,
        "date_column": "report_month",
        "categorical_columns": ["segment"],
        "numerical_columns": ["feature", "other"],
        "hidden_state_columns": (),
        "hyperopt": False,
        "verbose": False,
        "random_state": 42,
    }
    if backend == "boosting":
        values["engine"] = "catboost"
        values["model_params"] = {"iterations": 20, "depth": 3}
    else:
        values["engine"] = "tabular_transformer"
        values["model_params"] = {
            "max_epochs": 1,
            "batch_size": 128,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "num_workers": 0,
            "evaluations_per_epoch": 1,
            "patience": 1,
        }
    values.update(overrides)
    return BinaryTaskConfig(**values)


# --------------------------------------------------------------------------- #
# Mixed precision comes from the card                                          #
# --------------------------------------------------------------------------- #
def test_amp_matches_what_this_card_supports():
    """Not a mocked answer: what `resolve_amp` says here, and why."""
    expected = "bf16" if torch.cuda.is_bf16_supported() else "no"
    assert resolve_amp("gpu") == expected
    assert resolve_amp("cpu") == "no"


def test_the_amp_choice_is_recorded_in_the_run_config(tmp_path):
    """Trials at different precision are not comparable, so it is written down."""
    from fmlib.automl.backends.tabnn.assembly import build_train_config
    from fmlib.automl.backends.tabnn.data import build_schema, prepare_processed_data
    from fmlib.automl.data import CanonicalColumnMapper, ParquetSource
    from fmlib.automl.tasks.preparation import DataPreparation

    train = _write(tmp_path / "train.parquet", 400, 1)
    valid = _write(tmp_path / "valid.parquet", 200, 2)
    config = _config(tmp_path, "tabnn")
    mapper = CanonicalColumnMapper.from_config(config)
    internal = mapper.normalize_config(config)
    source = ParquetSource.resolve(train)
    valid_source = ParquetSource.resolve(valid)
    schema = build_schema(source, internal, mapper.to_external)
    processed = prepare_processed_data(
        config=internal,
        schema=schema,
        train_source=source,
        valid_source=valid_source,
        train_manifest=DataPreparation.source_manifest(source),
        valid_manifest=DataPreparation.source_manifest(valid_source),
        to_external=mapper.to_external,
    )

    built = build_train_config(
        config=config,
        task_name="binary",
        processed=processed,
        params=dict(config.model_params),
        trial_dir=tmp_path / "trial",
        device="gpu",
    )
    assert built["amp"] == resolve_amp("gpu")


# --------------------------------------------------------------------------- #
# Training on the card                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["boosting", "tabnn"])
def test_a_model_trains_on_the_card_and_scores(tmp_path, backend):
    """The whole public cycle with device='gpu', which no other test runs."""
    train = _write(tmp_path / "train.parquet", 800, 1)
    valid = _write(tmp_path / "valid.parquet", 400, 2)
    test = _write(tmp_path / "test.parquet", 400, 3)

    task = BinaryTask(_config(tmp_path, backend))
    result = task.train(train, valid)
    assert np.isfinite(result.validation_metrics["global"])

    artifact = task.save(tmp_path / f"artifact_{backend}")
    restored = BinaryTask.load(artifact)
    prediction = restored.predict(test)

    assert prediction.scores.height == 400
    assert np.isfinite(prediction.scores["score"].to_numpy()).all()


@pytest.mark.slow
def test_a_gpu_artifact_scores_the_same_on_the_cpu(tmp_path):
    """Inference must not depend on where training happened.

    The trees are the trees. If this ever stops holding, an artifact trained on
    the cluster and scored on a laptop silently gives different answers.
    """
    train = _write(tmp_path / "train.parquet", 800, 1)
    test = _write(tmp_path / "test.parquet", 400, 3)

    task = BinaryTask(_config(tmp_path, "boosting"))
    task.train(train, train)
    artifact = task.save(tmp_path / "artifact")

    restored = BinaryTask.load(artifact)
    on_card = restored.predict(test, device="gpu")
    on_cpu = restored.predict(test, device="cpu")

    np.testing.assert_allclose(
        on_card.scores["score"].to_numpy(),
        on_cpu.scores["score"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------------------------------- #
# The environment is put back                                                  #
# --------------------------------------------------------------------------- #
def test_boosting_is_pinned_to_one_card_and_tabnn_is_not(tmp_path):
    """CatBoost with task_type='GPU' spreads over every visible card unless
    pinned; TabNN takes its device count from the config, so pinning would both
    hide cards and make that field a lie."""
    assert _local_cuda_visibility(_config(tmp_path, "boosting")) == "0"
    assert _local_cuda_visibility(_config(tmp_path, "tabnn")) is None


@pytest.mark.parametrize("previous", ["3", "", None])
def test_the_variable_is_restored_including_its_absence(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", previous)

    with _cuda_visible_devices("0"):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous


@pytest.mark.parametrize("previous", ["3", None])
def test_the_variable_is_restored_after_a_failure(monkeypatch, previous):
    """The case that matters: a cell that raised must not narrow the kernel."""
    if previous is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", previous)

    with pytest.raises(RuntimeError, match="training blew up"):
        with _cuda_visible_devices("0"):
            raise RuntimeError("training blew up")

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous


def test_passing_none_changes_nothing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    with _cuda_visible_devices(None):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"


@pytest.mark.slow
def test_a_local_boosting_run_leaves_the_variable_as_it_found_it(tmp_path, monkeypatch):
    """End to end, not just the context manager: the run is what narrows it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    train = _write(tmp_path / "train.parquet", 400, 1)

    task = BinaryTask(_config(tmp_path, "boosting"))
    task.train(train, train)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
