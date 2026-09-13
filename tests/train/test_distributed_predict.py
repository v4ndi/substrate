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


def run_ranks(
    tmp_path, corpus, port, metric, nproc=2, nnodes=1, drop_tail=False
) -> list[dict]:
    """Launch ``nnodes`` launcher groups of ``nproc`` ranks; return the reports.

    ``nnodes > 1`` is a real multi-node job in every respect the code can see —
    separate launcher groups, separate node ranks, one rendezvous — except that
    the nodes happen to share this host's filesystem and clock.
    """
    out = tmp_path / "reports"
    dump = tmp_path / "dump"
    out.mkdir(exist_ok=True)
    dump.mkdir(exist_ok=True)

    common = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--nnodes={nnodes}",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
    ]
    tail = [
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
        tail.append("--drop-tail")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)
    env["OMP_NUM_THREADS"] = "1"
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
    outputs = [process.communicate(timeout=900)[0] for process in processes]
    for process, stdout in zip(processes, outputs, strict=False):
        if process.returncode != 0:
            pytest.fail(f"torchrun exited with {process.returncode}:\n{stdout}")

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


def test_two_nodes_keep_every_record_and_every_file(tmp_path, corpus, worker_port):
    """Four ranks in two launcher groups, a collector, and no coordination.

    Multi-node is where the part naming has to hold up without anyone checking:
    nothing is gathered, so the only thing keeping two writers apart is that the
    rank in the filename is the *global* rank, not the local one. Four ranks
    across two nodes must produce four distinct files.
    """
    reports = run_ranks(
        tmp_path, corpus, worker_port, metric="artifact", nproc=2, nnodes=2
    )
    assert [report["world_size"] for report in reports] == [4, 4, 4, 4]
    assert sorted(report["rank"] for report in reports) == [0, 1, 2, 3]

    parts = sorted(glob.glob(str(tmp_path / "dump" / "*.parquet")))
    assert len({Path(part).name for part in parts}) == len(parts) == 4, (
        f"one part file per rank expected, got {[Path(p).name for p in parts]}"
    )
    written = sorted(
        pd.concat([pd.read_parquet(part) for part in parts])["id"].tolist()
    )
    assert written == list(range(RECORDS))


def test_two_nodes_with_a_population_metric_still_agree(tmp_path, corpus, worker_port):
    """The gather path across node boundaries, with shards that differ in size."""
    reports = run_ranks(
        tmp_path, corpus, worker_port, metric="population", nproc=2, nnodes=2
    )
    owned = sorted(report["records_owned"] for report in reports)
    assert owned == [25, 25, 25, 26], f"shards were even, nothing was tested: {owned}"

    main = next(report for report in reports if report["rank"] == 0)
    assert main["scored_ids"] == list(range(RECORDS))
