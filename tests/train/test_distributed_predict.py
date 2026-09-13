"""Multi-rank inference over real ``torchrun`` processes.

Both failure modes these cover present as a hang or as missing rows, never as
an exception, and neither is reachable from a single process:

* ranks holding different numbers of batches used to deadlock at
  ``gather_object`` — one rank reached it, the others had already left the loop;
* two ranks writing predictions into one directory used to name their parts
  identically, so the second overwrote the first.

Ranks run on gloo with the GPUs hidden, which is what lets several of them
share a single-GPU host. Every test here is marked ``slow``.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.slow

WORKER = Path(__file__).parent / "predict_worker.py"
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Deliberately indivisible by the rank counts under test.
RECORDS = 101
ROWS_PER_FILE = 20


@pytest.fixture
def corpus(tmp_path) -> Path:
    """A parquet corpus whose record count does not divide by two."""
    path = tmp_path / "corpus"
    path.mkdir()
    frame = pd.DataFrame({
        "id": list(range(RECORDS)),
        "target": [index % 2 for index in range(RECORDS)],
        "cat_features": [[index % 7, index % 5] for index in range(RECORDS)],
        "num_features": [[float(index), index / 2] for index in range(RECORDS)],
    })
    for part, start in enumerate(range(0, RECORDS, ROWS_PER_FILE)):
        chunk = frame.iloc[start : start + ROWS_PER_FILE]
        pq.write_table(
            pa.Table.from_pandas(chunk, preserve_index=False),
            path / f"part-{part:05d}.parquet",
        )
    return path


def run_ranks(tmp_path, corpus, port, metric, nproc=2, drop_tail=False) -> list[dict]:
    """Launch ``nproc`` ranks and return every rank's report."""
    out = tmp_path / "reports"
    dump = tmp_path / "dump"
    out.mkdir(exist_ok=True)
    dump.mkdir(exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        str(WORKER),
        "--path",
        str(corpus),
        "--out",
        str(out),
        "--dump-dir",
        str(dump),
        "--metric",
        metric,
        "--batch-size",
        "50",
    ]
    if drop_tail:
        command.append("--drop-tail")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    env["OMP_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""

    process = subprocess.run(
        command,
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    if process.returncode != 0:
        pytest.fail(f"torchrun exited with {process.returncode}:\n{process.stdout}")

    return [
        json.loads(Path(name).read_text())
        for name in sorted(glob.glob(str(out / "rank*.json")))
    ]


def test_uneven_shards_do_not_deadlock_and_lose_nothing(tmp_path, corpus, worker_port):
    """The case that used to hang: 101 records over two ranks, tail kept."""
    reports = run_ranks(tmp_path, corpus, worker_port, metric="population")

    owned = sorted(report["records_owned"] for report in reports)
    assert owned == [50, 51], (
        f"the shards were not uneven, so nothing was tested: {owned}"
    )

    main = next(report for report in reports if report["rank"] == 0)
    assert main["scored_ids"] == list(range(RECORDS))


def test_dropping_the_tail_is_what_loses_records(tmp_path, corpus, worker_port):
    """The other half of the claim: with drop_tail the remainder is gone."""
    reports = run_ranks(
        tmp_path, corpus, worker_port, metric="population", drop_tail=True
    )

    main = next(report for report in reports if report["rank"] == 0)
    assert main["scored_ids"] == list(range(RECORDS - 1))


def test_collectors_write_one_directory_without_overwriting(
    tmp_path, corpus, worker_port
):
    """No rank needs another's rows, so every rank writes its own parts."""
    run_ranks(tmp_path, corpus, worker_port, metric="artifact")

    parts = sorted(glob.glob(str(tmp_path / "dump" / "*.parquet")))
    assert len(parts) == 2, f"expected one part per rank, got {parts}"
    assert len({Path(part).name for part in parts}) == 2, "parts collided on a name"

    written = sorted(
        pd.concat([pd.read_parquet(part) for part in parts])["id"].tolist()
    )
    assert written == list(range(RECORDS))
