"""Training must stream: ten times the rows must not mean ten times the memory.

Out-of-core is the property the whole data path is built for, so it is checked
rather than asserted in a docstring. Each size runs in its own process and
reports its own peak RSS, because ``ru_maxrss`` is a high-water mark that never
comes back down within one process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.slow

CHILD = '''
import json, resource, sys
import numpy as np
from fmlib.automl import BinaryTask, BinaryTaskConfig

rows, train_dir, valid_dir, output_dir = sys.argv[1:]
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
    model_params={
        "max_epochs": 1,
        "batch_size": 256,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 2,
        "num_workers": 0,
        "evaluations_per_epoch": 1,
    },
    verbose=False,
    output_dir=output_dir,
    environment={},
)
task = BinaryTask(config)
task.train(train_dir, valid_dir)
peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"rows": int(rows), "peak_mib": peak_kib / 1024}))
'''


def _write(directory, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({
        "epk_id": np.arange(rows),
        "segment": rng.choice(["a", "b", "c"], rows),
        "balance": rng.normal(size=rows),
        "y": rng.integers(0, 2, rows).astype(np.int8),
    })
    for index in range(4):
        part = frame.slice(index * rows // 4, rows // 4)
        part.write_parquet(directory / f"part-{index}.parquet")
    return str(directory)


def _peak_mib(tmp_path, rows: int, tag: str) -> float:
    train = _write(tmp_path / f"{tag}-train", rows, 1)
    valid = _write(tmp_path / f"{tag}-valid", rows // 4, 2)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(CHILD),
            str(rows),
            train,
            valid,
            str(tmp_path / f"{tag}-out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])["peak_mib"]


def test_peak_memory_does_not_follow_the_row_count(tmp_path):
    small = _peak_mib(tmp_path, 4_000, "small")
    large = _peak_mib(tmp_path, 40_000, "large")
    # Ten times the rows. A pipeline that materialized a split would show it;
    # a streaming one pays only for one batch, plus whatever the interpreter
    # and torch already cost, which is most of both numbers.
    assert large < small * 1.5, f"small={small:.0f} MiB large={large:.0f} MiB"
