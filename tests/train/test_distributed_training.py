"""End-to-end training over real ``torchrun`` processes.

The design doc calls a multi-rank smoke test a prerequisite for the rewrite,
not a follow-up: the old loop had no tests at all, and the failure modes being
removed here (double-sharding, mismatched batch counts, scheduler step budgets)
all present as a hang or as silently worse training rather than an exception.

Ranks run on gloo with the GPUs hidden, which is what lets a 2- and 4-rank job
— and a simulated 2-node job — run on a single-GPU host.

Every test here is marked ``slow``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

WORKER = Path(__file__).parent / "trainer_worker.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def run_workers(
    out: Path,
    nproc_per_node: int,
    nnodes: int = 1,
    port: int = 29811,
    **worker_args,
) -> dict:
    """Launch ``nnodes`` torchrun groups and return every rank's report."""
    common = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
    ]
    tail = [str(WORKER), "--out", str(out)]
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
    # Hide the GPUs: gloo collectives belong on CPU, and several ranks cannot
    # meaningfully share one device anyway.
    env["CUDA_VISIBLE_DEVICES"] = ""

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


@pytest.mark.parametrize("nproc_per_node", [1, 2])
def test_training_converges_and_ranks_agree(tmp_path, nproc_per_node, worker_port):
    report = run_workers(
        tmp_path / "report.json",
        nproc_per_node=nproc_per_node,
        port=worker_port,
        epochs=4,
    )
    assert report["world_size"] == nproc_per_node
    ranks = report["ranks"]

    for rank in ranks:
        losses = rank["train_losses"]
        assert len(losses) == 4
        assert losses[-1] < losses[0], f"loss did not fall: {losses}"

    # DDP averages gradients, so every rank must end on the same weights. A
    # mismatch means a rank skipped or duplicated an all-reduce.
    fingerprints = {tuple(rank["params"]) for rank in ranks}
    assert len(fingerprints) == 1, "ranks diverged"

    # Equal step counts are the other half of that contract.
    assert len({rank["global_step"] for rank in ranks}) == 1
    # Per-rank losses are a global reduction, so they agree too.
    assert len({tuple(rank["train_losses"]) for rank in ranks}) == 1


def test_checkpoint_round_trips(tmp_path, worker_port):
    report = run_workers(
        tmp_path / "report.json",
        nproc_per_node=2,
        port=worker_port,
        epochs=2,
    )
    main_rank = report["ranks"][0]
    assert main_rank["checkpoint_roundtrip"] is True
    assert main_rank["checkpoints"], "no checkpoint was written"


def test_simulated_multi_node(tmp_path, worker_port):
    """Two launcher groups of two ranks each: a 4-rank, 2-node job."""
    report = run_workers(
        tmp_path / "report.json",
        nproc_per_node=2,
        nnodes=2,
        port=worker_port,
        epochs=3,
    )
    assert report["world_size"] == 4
    ranks = report["ranks"]
    assert len({tuple(rank["params"]) for rank in ranks}) == 1
    assert ranks[0]["train_losses"][-1] < ranks[0]["train_losses"][0]


def test_gradient_accumulation_halves_the_optimizer_steps(tmp_path, worker_port):
    """Accumulating over 2 micro-batches must take half as many steps."""
    plain = run_workers(
        tmp_path / "plain.json",
        nproc_per_node=2,
        port=worker_port,
        epochs=2,
    )
    accumulated = run_workers(
        tmp_path / "accum.json",
        nproc_per_node=2,
        port=worker_port + 1,
        epochs=2,
        grad_accum=2,
    )
    assert (
        accumulated["ranks"][0]["global_step"] * 2 == (plain["ranks"][0]["global_step"])
    )
    assert len({tuple(rank["params"]) for rank in accumulated["ranks"]}) == 1


def test_gradient_clipping_keeps_ranks_in_sync(tmp_path, worker_port):
    report = run_workers(
        tmp_path / "report.json",
        nproc_per_node=2,
        port=worker_port,
        epochs=2,
        clip_grad_norm=0.5,
    )
    assert len({tuple(rank["params"]) for rank in report["ranks"]}) == 1
    assert (
        report["ranks"][0]["train_losses"][-1] < (report["ranks"][0]["train_losses"][0])
    )


def test_dataloader_workers_do_not_duplicate_records(tmp_path, worker_port):
    report = run_workers(
        tmp_path / "report.json",
        nproc_per_node=2,
        port=worker_port,
        epochs=2,
        num_workers=2,
    )
    ranks = report["ranks"]
    assert len({rank["samples_seen"] for rank in ranks}) == 1
    assert len({tuple(rank["params"]) for rank in ranks}) == 1


def test_bf16_autocast_trains(tmp_path, worker_port):
    """bf16 needs no GradScaler; the loop must still converge and stay in sync."""
    report = run_workers(
        tmp_path / "report.json",
        nproc_per_node=2,
        port=worker_port,
        epochs=3,
        amp="bf16",
    )
    losses = report["ranks"][0]["train_losses"]
    assert losses[-1] < losses[0]
    assert len({tuple(rank["params"]) for rank in report["ranks"]}) == 1
