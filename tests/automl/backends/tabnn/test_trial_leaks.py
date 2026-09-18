"""Fifty trials in one process must leave it where they found it.

The in-process runner is the only one that executes anything today. If it
leaks, the subprocess runner stops being an optimisation and becomes a
requirement, so the question is worth answering rather than assuming.

What is measured: resident memory, open file descriptors, and that every trial
actually produced an objective — a runner that silently stopped training would
otherwise look beautifully leak-free.
"""

from __future__ import annotations

import os
import resource
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from fmlib.automl import BinaryTaskConfig
from fmlib.automl.backends.tabnn.assembly import build_train_config
from fmlib.automl.backends.tabnn.data import build_schema, prepare_processed_data
from fmlib.automl.backends.tabnn.runner import InProcessRunner, TrialSpec
from fmlib.automl.data import CanonicalColumnMapper, ParquetSource
from fmlib.automl.execution import ExecutionContext
from fmlib.automl.tasks.preparation import DataPreparation

pytestmark = pytest.mark.slow

TRIALS = 50


def _write(directory: Path, rows: int, seed: int) -> ParquetSource:
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    balance = rng.normal(size=rows)
    pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 1000,
        "segment": rng.choice(["a", "b"], rows),
        "balance": balance,
        "y": (rng.uniform(size=rows) < 1 / (1 + np.exp(-2 * balance))).astype(np.int8),
    }).write_parquet(directory / "part-0.parquet")
    return ParquetSource.resolve(directory)


def _open_descriptors() -> int:
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def test_fifty_sequential_trials_do_not_grow_the_process(tmp_path):
    config = BinaryTaskConfig(
        env_type="local",
        backend="tabnn",
        engine="tabular_transformer",
        device="cpu",
        target_column="y",
        client_id_column="epk_id",
        group_column=None,
        date_column=None,
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hidden_state_columns=[],
        hyperopt=False,
        verbose=False,
        output_dir=tmp_path / "outputs",
        environment={},
    )
    context = ExecutionContext.from_config(config)
    internal = context.internal_config
    mapper = CanonicalColumnMapper.from_config(config)
    train = _write(tmp_path / "train", 400, 1)
    valid = _write(tmp_path / "valid", 200, 2)
    schema = build_schema(train, internal, mapper.to_external)
    processed = prepare_processed_data(
        config=internal,
        schema=schema,
        train_source=train,
        valid_source=valid,
        train_manifest=DataPreparation.source_manifest(train),
        valid_manifest=DataPreparation.source_manifest(valid),
        to_external=mapper.to_external,
    )

    runner = InProcessRunner()
    settled_rss: float | None = None
    settled_fds: int | None = None
    objectives: list[float] = []

    for index in range(TRIALS):
        params = {
            "max_epochs": 1,
            "batch_size": 128,
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "num_workers": 0,
            "evaluations_per_epoch": 1,
            "patience": 1,
        }
        train_config = build_train_config(
            config=config,
            task_name="binary",
            processed=processed,
            params=params,
            trial_dir=tmp_path / "trials" / f"trial-{index:04d}",
        )
        spec = TrialSpec(
            trial_id=f"trial-{index:04d}",
            config=OmegaConf.to_container(train_config, resolve=True),
            trial_dir=str(tmp_path / "trials" / f"trial-{index:04d}"),
            metric_name=train_config["automl"]["metric"],
            direction=train_config["automl"]["direction"],
        )
        result = runner.collect(runner.submit(spec))
        assert result.completed, result.error
        objectives.append(result.objective)
        # The first few trials still pay one-off costs — lazy imports, arena
        # growth — so the baseline is taken once those are behind us.
        if index == 9:
            settled_rss, settled_fds = _rss_mib(), _open_descriptors()

    assert len(objectives) == TRIALS
    assert all(np.isfinite(objectives))

    assert _open_descriptors() <= settled_fds + 5, (
        f"descriptors grew from {settled_fds} to {_open_descriptors()}"
    )
    assert _rss_mib() <= settled_rss * 1.25, (
        f"peak RSS grew from {settled_rss:.0f} MiB to {_rss_mib():.0f} MiB "
        f"over {TRIALS} trials"
    )
