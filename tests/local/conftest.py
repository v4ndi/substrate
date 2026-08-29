"""Fixtures for the local preprocessing backend tests.

The pure-local tests need only pyarrow + numpy. The Spark-parity tests
(``test_spark_parity.py``) reuse the ``spark_session`` fixture from the root
``tests/conftest.py`` and are skipped automatically when no Spark-compatible
JDK is available.
"""

from __future__ import annotations

import datetime as dt
import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def write_parquet(tmp_path):
    """Return ``write(table, n_files=1) -> dir_path`` splitting ``table`` into files."""

    def _write(table: pa.Table, n_files: int = 3, name: str = "ds") -> str:
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        n = table.num_rows
        step = max(1, -(-n // n_files))
        for i, start in enumerate(range(0, max(n, 1), step)):
            pq.write_table(table.slice(start, step), d / f"part-{i:03d}.parquet")
        if not glob.glob(str(d / "*.parquet")):
            pq.write_table(table, d / "part-000.parquet")
        return str(d)

    return _write


@pytest.fixture
def tabular_table():
    rng = np.random.default_rng(7)
    n = 12000
    cat_a = rng.choice(["r", "s", "t", "u"], n, p=[0.5, 0.25, 0.15, 0.10])
    cat_a = np.array(cat_a, dtype=object)
    cat_a[rng.random(n) < 0.02] = None
    cat_b = rng.choice([-1, 0, 1, 2, 3], n, p=[0.05, 0.1, 0.15, 0.3, 0.4])
    num_1 = rng.normal(500, 120, n)
    num_2 = rng.exponential(4, n)
    num_3 = rng.normal(0, 1, n)
    return pa.table({
        "epk_id": pa.array(np.arange(n)),
        "cat_a": pa.array(cat_a),
        "cat_b": pa.array(cat_b),
        "num_1": pa.array(num_1, mask=rng.random(n) < 0.03),
        "num_2": pa.array(num_2),
        "num_3": pa.array(num_3),
        "target": pa.array(rng.integers(0, 2, n)),
    })


@pytest.fixture
def sequence_table():
    rng = np.random.default_rng(11)
    base = dt.datetime(2024, 1, 1)
    rows = []
    for uid in range(1, 201):
        k = int(rng.integers(3, 20))
        offs = np.sort(rng.choice(np.arange(0, 20000), size=k, replace=False))
        for off in offs:
            rows.append({
                "epk_id": int(uid),
                "timestamps": base + dt.timedelta(minutes=int(off)),
                "mcc": rng.choice(["5411", "5812", "5999", "4814", "6011"]),
                "direction": None if rng.random() < 0.05 else rng.choice(["in", "out"]),
                "price": float(round(rng.uniform(1, 5000), 2)),
                "event_ids": int(rng.integers(0, 3)),
            })
    rng.shuffle(rows)
    schema = pa.schema([
        ("epk_id", pa.int32()),
        ("timestamps", pa.timestamp("us")),
        ("mcc", pa.string()),
        ("direction", pa.string()),
        ("price", pa.float64()),
        ("event_ids", pa.int32()),
    ])
    return pa.Table.from_pylist(rows, schema=schema)
