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

    assert isinstance(spec.config, dict), "the spec must normalise the assembly's config"

    path = spec.write()
    restored = TrialSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert isinstance(restored.config, dict)
    assert restored.config == spec.config
    assert restored.config["model"]["_target_"] == "fmlib.pipeline.tabular.SupervisedLearner"


def _cpu_ranks_env() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join([
            str(REPO_ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep),
    }


@pytest.mark.parametrize("ranks", [1, 2])
def test_a_trial_runs_under_torchrun_and_comes_back_through_result_json(
    tmp_path, prepared, ranks
):
    config, processed = prepared
    runner = TorchrunRunner(timeout=900.0, env=_cpu_ranks_env())
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
    runner = TorchrunRunner(timeout=300.0, env=_cpu_ranks_env())
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
        env={**os.environ, **_cpu_ranks_env()},
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=900,
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
