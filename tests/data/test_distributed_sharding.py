"""End-to-end sharding tests over real ``torchrun`` processes.

These spawn actual worker processes with a real ``torch.distributed`` process
group on the gloo backend, so ranks, collectives and DataLoader workers all
behave as they do in a training job. The gloo backend runs on CPU, which is what
lets a multi-rank — and a simulated multi-node — job run on a single-GPU host;
NCCL cannot be used that way because two ranks cannot share one device.

Every test here is marked ``slow``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.slow

WORKER = Path(__file__).parent / "distributed_sharding_worker.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def run_workers(
    path: str,
    out: Path,
    nproc_per_node: int,
    nnodes: int = 1,
    port: int = 29711,
    **worker_args,
) -> dict:
    """Launch ``nnodes`` torchrun groups and return rank 0's gathered report.

    ``nnodes > 1`` starts several torchrun launchers against one rendezvous
    endpoint. Each launcher owns a distinct ``node_rank``, which is exactly the
    shape of a multi-node job; only the network between them is local.
    """
    common = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
    ]
    tail = [str(WORKER), "--path", str(path), "--out", str(out)]
    for key, value in worker_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                tail.append(flag)
        else:
            tail.extend([flag, str(value)])

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    env["OMP_NUM_THREADS"] = "1"

    processes = [
        subprocess.Popen(
            [*common, f"--node_rank={node_rank}", *tail],
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for node_rank in range(nnodes)
    ]
    outputs = []
    for process in processes:
        stdout, _ = process.communicate(timeout=900)
        outputs.append(stdout)
    for process, stdout in zip(processes, outputs, strict=False):
        if process.returncode != 0:
            pytest.fail(f"torchrun exited with {process.returncode}:\n{stdout}")

    return json.loads(out.read_text())


@pytest.fixture
def corpus(synth_sequence_dataset):
    """Uneven synthetic corpus — equal-sized files hide sharding bugs."""
    return synth_sequence_dataset(
        num_records=613, num_output_partitions=7, num_events_range=(0, 60)
    )


@pytest.mark.parametrize("nproc_per_node", [2, 3])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_ranks_partition_the_corpus(
    corpus, tmp_path, nproc_per_node, num_workers, worker_port
):
    report = run_workers(
        corpus,
        tmp_path / "report.json",
        nproc_per_node=nproc_per_node,
        port=worker_port,
        num_workers=num_workers,
        min_length=10,
    )

    lengths = pd.read_parquet(corpus)["mcc"].apply(len)
    total_valid = int((lengths >= 10).sum())
    world_size = report["world_size"]
    assert world_size == nproc_per_node

    per_rank = [rank["epochs"][0] for rank in report["ranks"]]
    all_ids = [record_id for epoch in per_rank for record_id in epoch["ids"]]

    # Equal cardinality on every rank is what keeps DDP from deadlocking.
    assert len({epoch["num_batches"] for epoch in per_rank}) == 1
    assert len({len(epoch["ids"]) for epoch in per_rank}) == 1
    # __len__ must agree with what iteration actually produced.
    for epoch in per_rank:
        assert epoch["len"] == len(epoch["ids"])

    assert len(set(all_ids)) == len(all_ids), "a record reached two ranks"
    assert len(all_ids) == total_valid - total_valid % world_size


def test_simulated_multi_node(corpus, tmp_path, worker_port):
    """Two launcher groups of two ranks each: a 4-rank, 2-node job."""
    report = run_workers(
        corpus,
        tmp_path / "report.json",
        nproc_per_node=2,
        nnodes=2,
        port=worker_port,
        min_length=10,
    )
    assert report["world_size"] == 4

    per_rank = [rank["epochs"][0] for rank in report["ranks"]]
    all_ids = [record_id for epoch in per_rank for record_id in epoch["ids"]]

    lengths = pd.read_parquet(corpus)["mcc"].apply(len)
    total_valid = int((lengths >= 10).sum())

    assert len({len(epoch["ids"]) for epoch in per_rank}) == 1
    assert len(set(all_ids)) == len(all_ids)
    assert len(all_ids) == total_valid - total_valid % 4


def test_event_type_filtering_across_ranks(corpus, tmp_path, worker_port):
    report = run_workers(
        corpus,
        tmp_path / "report.json",
        nproc_per_node=2,
        port=worker_port,
        min_length=3,
        selected_event_ids="1",
    )
    per_rank = [rank["epochs"][0] for rank in report["ranks"]]
    all_ids = [record_id for epoch in per_rank for record_id in epoch["ids"]]

    assert len({len(epoch["ids"]) for epoch in per_rank}) == 1
    assert len(set(all_ids)) == len(all_ids)
    assert all_ids, "event-type filtering must not empty the dataset"


def test_epoch_rotation_changes_the_dropped_tail(
    synth_sequence_dataset, tmp_path, worker_port
):
    # 100 records over 3 ranks leaves a tail of 1, which is what rotates.
    path = synth_sequence_dataset(
        num_records=100, num_output_partitions=6, num_events_range=(5, 60)
    )
    report = run_workers(
        path,
        tmp_path / "report.json",
        nproc_per_node=3,
        port=worker_port,
        min_length=1,
        epochs=2,
        rotate_tail=True,
    )

    tail = report["ranks"][0]["tail"]
    assert tail > 0, (
        "the corpus divides evenly across the ranks, so there is no tail to "
        "rotate — pick a record count that leaves a remainder"
    )

    first = {
        record_id for rank in report["ranks"] for record_id in rank["epochs"][0]["ids"]
    }
    second = {
        record_id for rank in report["ranks"] for record_id in rank["epochs"][1]["ids"]
    }
    assert len(first) == len(second)
    assert first != second, "rotate_tail must change which records are dropped"


def test_tabular_dataset_partitions_the_corpus(
    synth_sequence_dataset, tmp_path, worker_port
):
    path = synth_sequence_dataset(
        num_records=257, num_output_partitions=5, tabular_features=True
    )
    report = run_workers(
        path,
        tmp_path / "report.json",
        nproc_per_node=3,
        port=worker_port,
        modality="tabular",
    )

    per_rank = [rank["epochs"][0] for rank in report["ranks"]]
    all_ids = [record_id for epoch in per_rank for record_id in epoch["ids"]]

    assert len({len(epoch["ids"]) for epoch in per_rank}) == 1
    assert len(set(all_ids)) == len(all_ids)
    assert len(all_ids) == 257 - 257 % 3
