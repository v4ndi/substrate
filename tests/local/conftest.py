"""Fixtures for the local preprocessing backend tests.

The pure-local tests need only pyarrow + numpy. The Spark-parity tests
(``test_spark_parity.py``) are skipped automatically when a working Spark /
JDK 17 environment is not available.
"""

from __future__ import annotations

import datetime as dt
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import glob as _glob
import subprocess as _sp

_JDK_CANDIDATES = [
    os.environ.get("SPARK_JDK", ""),
    "/home/jovyan/.sdkman/candidates/java/17.0.13-tem",
    "/usr/lib/jvm/java-17-openjdk-amd64",
    *_glob.glob("/home/jovyan/.sdkman/candidates/java/1[17]*"),
    *_glob.glob("/usr/lib/jvm/*-1[17]-*"),
    os.environ.get("JAVA_HOME", ""),
]


def _java_major(home: str) -> int | None:
    java = os.path.join(home, "bin", "java")
    if not os.path.exists(java):
        return None
    try:
        out = _sp.run([java, "-version"], capture_output=True, text=True).stderr
    except OSError:
        return None
    for tok in out.replace('"', " ").split():
        if tok.startswith(("1.8", "8.", "11.", "17.")):
            return 8 if tok.startswith(("1.8", "8.")) else int(tok.split(".")[0])
    return None


def _ensure_java() -> bool:
    """Point JAVA_HOME at a Spark-compatible JDK (8/11/17). Spark 3.5 breaks on 21+."""
    for home in _JDK_CANDIDATES:
        if home and _java_major(home) in (8, 11, 17):
            os.environ["JAVA_HOME"] = home
            os.environ["PATH"] = f"{home}/bin:{os.environ['PATH']}"
            return True
    return False


@pytest.fixture(scope="session")
def spark_session():
    if not _ensure_java():
        pytest.skip("no JDK found for Spark parity tests")
    try:
        from pyspark.sql import SparkSession
    except ImportError:  # pragma: no cover
        pytest.skip("pyspark not installed")
    try:
        spark = (
            SparkSession.builder.appName("local-parity")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not start Spark: {exc}")
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


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
