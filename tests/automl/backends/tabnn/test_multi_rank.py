"""Two ranks, under a real torchrun, on CPU.

What this proves and what it does not, stated plainly: ranks launch, the spec
travels, the result comes back, the ranks agree, and every record of a split is
read exactly once across them. What it does **not** prove is NCCL collectives,
rank-to-card placement or GPU memory races -- there is one card on this machine
and the ranks here run on gloo with it hidden. "Verified on CPU ranks", not
"multi-GPU covered".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from automl.backends.tabnn.handwritten import (
    cpu_ranks_env,
    digest,
    handwritten_config,
    only_checkpoint,
    run_handwritten,
)
from fmlib.automl import BinaryTaskConfig
from fmlib.automl.backends.tabnn.assembly import build_train_config
from fmlib.automl.backends.tabnn.data import build_schema, prepare_processed_data
from fmlib.automl.backends.tabnn.runner import TorchrunRunner, TrialSpec
from fmlib.automl.data import CanonicalColumnMapper, ParquetSource
from fmlib.automl.execution import ExecutionContext
from fmlib.automl.tasks.preparation import DataPreparation

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[4]
ROWS = 800

PARAMS = {
    "max_epochs": 1,
    "batch_size": 64,
    "hidden_size": 16,
    "num_layers": 1,
    "num_heads": 2,
    "num_workers": 0,
    "evaluations_per_epoch": 1,
    "patience": 1,
}


def _write(directory: Path, rows: int, seed: int) -> ParquetSource:
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    balance = rng.normal(size=rows)
    frame = pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "segment": rng.choice(["a", "b"], rows),
        "balance": balance,
        "y": (rng.uniform(size=rows) < 1 / (1 + np.exp(-2 * balance))).astype(np.int8),
    })
    for index in range(4):
        part = frame.slice(index * rows // 4, rows // 4)
        part.write_parquet(directory / f"part-{index}.parquet")
    return ParquetSource.resolve(directory)


@pytest.fixture
def prepared(tmp_path):
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
        model_params=dict(PARAMS),
        verbose=False,
        output_dir=tmp_path / "outputs",
        environment={},
    )
    context = ExecutionContext.from_config(config)
    mapper = CanonicalColumnMapper.from_config(config)
    train = _write(tmp_path / "train", ROWS, 1)
    valid = _write(tmp_path / "valid", ROWS // 2, 2)
    schema = build_schema(train, context.internal_config, mapper.to_external)
    processed = prepare_processed_data(
        config=context.internal_config,
        schema=schema,
        train_source=train,
        valid_source=valid,
        train_manifest=DataPreparation.source_manifest(train),
        valid_manifest=DataPreparation.source_manifest(valid),
        to_external=mapper.to_external,
    )
    return config, processed


def _spec(config, processed, trial_dir: Path, num_gpus: int) -> TrialSpec:
    train_config = build_train_config(
        config=config,
        task_name="binary",
        processed=processed,
        params=dict(PARAMS),
        trial_dir=trial_dir,
        world_size=num_gpus,
    )
    return TrialSpec(
        # The DictConfig the assembly returns, exactly as `fit_model_part`
        # hands it over. Converting it here instead would test a shape
        # production never produces -- which is how a stringified spec reached
        # the ranks unnoticed.
        trial_id=trial_dir.name,
        config=train_config,
        trial_dir=str(trial_dir),
        metric_name=train_config["automl"]["metric"],
        direction=train_config["automl"]["direction"],
        num_gpus=num_gpus,
    )


def test_spec_survives_the_trip_to_disk(tmp_path, prepared):
    """What a rank reads back has to be the run, not a picture of it.

    ``build_train_config`` returns a ``DictConfig``, and a ``DictConfig`` is not
    a ``dict``: written naively it lands in ``run_spec.json`` as its own repr,
    and every runner that reads the spec instead of holding it -- torchrun,
    Osiris -- gets a string. The in-process runner cannot catch this, so the
    round trip is asserted here.
    """
    config, processed = prepared
    spec = _spec(config, processed, tmp_path / "trial-0", num_gpus=1)

    assert isinstance(spec.config, dict), (
        "the spec must normalise the assembly's config"
    )

    path = spec.write()
    restored = TrialSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert isinstance(restored.config, dict)
    assert restored.config == spec.config
    assert (
        restored.config["model"]["_target_"]
        == "fmlib.pipeline.tabular.SupervisedLearner"
    )


@pytest.mark.parametrize("ranks", [1, 2])
def test_a_trial_runs_under_torchrun_and_comes_back_through_result_json(
    tmp_path, prepared, ranks
):
    config, processed = prepared
    runner = TorchrunRunner(timeout=240.0, env=cpu_ranks_env())
    spec = _spec(config, processed, tmp_path / f"trial-{ranks}", ranks)

    result = runner.collect(runner.submit(spec))

    assert result.state == "COMPLETE", result.error
    assert np.isfinite(result.objective)
    # One result, written by rank 0 only: the collection path is the same one
    # an Osiris job uses.
    assert len(list(Path(spec.trial_dir).glob("result.json"))) == 1
    payload = json.loads((Path(spec.trial_dir) / "result.json").read_text())
    assert payload["objective"] == pytest.approx(result.objective)


def test_a_trial_that_cannot_start_is_a_failed_trial_not_an_exception(tmp_path):
    runner = TorchrunRunner(timeout=120.0, env=cpu_ranks_env())
    spec = TrialSpec(
        trial_id="broken",
        config={"model": {"_target_": "nope.NotAThing"}},
        trial_dir=str(tmp_path / "broken"),
        metric_name="roc_auc",
        direction="max",
    )
    result = runner.collect(runner.submit(spec))
    assert result.state == "FAIL"
    assert result.error


RANK_PROBE = """
import json, os, sys
import torch
from torch.utils.data import DataLoader
from fmlib.data import TabularDataset, TabularCollateFn

path, out = sys.argv[1], sys.argv[2]
torch.distributed.init_process_group(backend="gloo")
rank = torch.distributed.get_rank()
dataset = TabularDataset(
    path=path, shuffle_files=False, shuffle_pq=False, drop_tail=False
)
loader = DataLoader(dataset, batch_size=16, num_workers=0,
                    collate_fn=TabularCollateFn(target_column="y"))
seen = []
for batch in loader:
    seen.extend(int(value) for value in batch["epk_id"])
gathered = [None] * torch.distributed.get_world_size()
torch.distributed.all_gather_object(gathered, seen)
if rank == 0:
    with open(out, "w") as handle:
        json.dump(gathered, handle)
torch.distributed.destroy_process_group()
"""


def test_every_record_is_read_exactly_once_across_ranks(tmp_path, prepared):
    """What a distributed predict rests on: a sharded split covers itself."""
    _, processed = prepared
    probe = tmp_path / "probe.py"
    probe.write_text(RANK_PROBE, encoding="utf-8")
    out = tmp_path / "seen.json"

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "--standalone",
            str(probe),
            str(processed.valid.path),
            str(out),
        ],
        env={**os.environ, **cpu_ranks_env()},
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=240,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    per_rank = json.loads(out.read_text())
    assert len(per_rank) == 2
    everything = [value for rows in per_rank for value in rows]
    expected = pl.read_parquet(processed.valid.path)["epk_id"].to_list()

    # Exactly once: no duplicates across ranks, and nothing dropped.
    assert len(everything) == len(set(everything))
    assert sorted(everything) == sorted(expected)
    # And the order is recoverable, which is what the identity column is for.
    assert sorted(everything) == sorted(set(expected))


# --------------------------------------------------------------------------- #
# T4: the number two ranks produce, not the fact that they finished            #
# --------------------------------------------------------------------------- #
def _run_ranks(config, processed, tmp_path: Path, ranks: int) -> tuple[Path, float]:
    """Run one trial through AutoML on ``ranks`` ranks; return checkpoint and score."""
    runner = TorchrunRunner(timeout=240.0, env=cpu_ranks_env())
    spec = _spec(config, processed, tmp_path / f"automl-{ranks}", ranks)
    result = runner.collect(runner.submit(spec))
    assert result.state == "COMPLETE", result.error
    return Path(spec.trial_dir), float(result.objective)


def test_two_ranks_produce_the_same_run_as_a_handwritten_torchrun(tmp_path, prepared):
    """Same claim as the single-rank parity, on the path that was broken.

    Until the spec fix, a rank read a string where the run should have been, so
    everything here would have failed at launch. What it asserts now is the
    thing the earlier multi-rank test did not: that two ranks under AutoML take
    the same steps as two ranks a person would have launched, rather than
    merely that they finish and return a number.

    Both sides are pinned to one thread by `cpu_ranks_env`, because bit
    identity holds at a matched thread count and not otherwise.
    """
    config, processed = prepared
    ours, objective = _run_ranks(config, processed, tmp_path, ranks=2)

    handwritten = tmp_path / "handwritten-2"
    run_handwritten(handwritten_config(ours / "run_spec.json", handwritten), ranks=2)

    mine = only_checkpoint(ours)
    theirs = only_checkpoint(handwritten)
    assert mine.name == theirs.name, "the two runs stopped at different steps"
    for name in ("model.bin", "checkpoint.pt"):
        assert digest(mine / name) == digest(theirs / name), (
            f"{name} differs between AutoML and a hand-written torchrun"
        )

    state = torch.load(mine / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert state["state"]["best_metric"] == pytest.approx(objective)


def test_one_rank_and_two_ranks_agree_on_the_metric_without_agreeing_on_the_bits(
    tmp_path, prepared
):
    """Two ranks is a different computation, not a faster identical one.

    Each rank sees half the shards, so the batches differ, the gradient
    reductions differ and the weights differ -- expecting bit identity here
    would be wrong, and a test that demanded it would be deleted the first time
    it failed. What must hold is that the two agree on the answer, so the
    assertion is the metric within a stated tolerance, and the inequality of
    the weights is asserted too so that a silent collapse to one rank cannot
    pass as success.
    """
    config, processed = prepared
    one, objective_one = _run_ranks(config, processed, tmp_path, ranks=1)
    two, objective_two = _run_ranks(config, processed, tmp_path, ranks=2)

    # Wide on purpose: this is agreement between two computations, not noise
    # around one. A model this small on 800 rows has real run-to-run spread.
    assert objective_two == pytest.approx(objective_one, abs=0.15)

    left = torch.load(
        only_checkpoint(one) / "model.bin", map_location="cpu", weights_only=True
    )
    right = torch.load(
        only_checkpoint(two) / "model.bin", map_location="cpu", weights_only=True
    )
    assert list(left) == list(right)
    assert any(not torch.equal(left[key], right[key]) for key in left), (
        "two ranks produced bit-identical weights, which means the second rank "
        "saw the same data as the first"
    )
