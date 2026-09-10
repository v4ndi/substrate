"""Shared test fixtures.

Two groups:

* ``_ensure_java`` / ``spark_session`` / ``spark`` — point ``JAVA_HOME`` at a
  Spark-compatible JDK (8/11/17; Spark 3.5 breaks on 21+) and hand out a single
  session-scoped :class:`~pyspark.sql.SparkSession`. Tests that need Spark are
  **skipped** (not errored) when no such JDK / ``pyspark`` is available.
* ``synth_sequence_dataset`` — a pure pandas/pyarrow generator for synthetic
  event-sequence parquet (no Spark), replacing the removed ``avatar.synth``.
"""

from __future__ import annotations

import glob as _glob
import os
import subprocess as _sp
import tempfile

import numpy as np
import pandas as pd
import pytest

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
    """Point ``JAVA_HOME`` at a Spark-compatible JDK (8/11/17)."""
    for home in _JDK_CANDIDATES:
        if home and _java_major(home) in (8, 11, 17):
            os.environ["JAVA_HOME"] = home
            os.environ["PATH"] = f"{home}/bin:{os.environ['PATH']}"
            return True
    return False


@pytest.fixture(scope="session")
def spark_session():
    if not _ensure_java():
        pytest.skip("no Spark-compatible JDK (8/11/17) found")
    try:
        from pyspark.sql import SparkSession
    except ImportError:  # pragma: no cover
        pytest.skip("pyspark not installed")
    try:
        spark = (
            SparkSession.builder
            .appName("avatar-tests")
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


@pytest.fixture(scope="session")
def spark(spark_session):
    """Alias — some suites request the fixture as ``spark``."""
    return spark_session


# --------------------------------------------------------------------------- #
# Synthetic event-sequence data (no Spark)                                     #
# --------------------------------------------------------------------------- #
def generate_sequence_dataset(
    num_records: int = 1000,
    num_output_partitions: int = 8,
    num_events_range: tuple[int, int] = (0, 100),
    target_column: str | None = None,
    tabular_features: bool = False,
    seed: int = 0,
) -> str:
    """Write partitioned parquet with per-row event-sequence lists.

    Columns: ``epk_id`` (int64), ``evt_dttm`` (list<timestamp>), ``mcc``
    (list<int64>, 6 classes), ``price`` (list<float>), ``event_ids``
    (list<int64>, values 0/1); ``target`` (int64) when
    ``target_column == "classification"``; ``cat_features`` /``num_features``
    (10 each) when ``tabular_features``. Files are partitioned by
    ``epk_id % num_output_partitions``. Returns the dataset directory.
    """
    if target_column not in (None, "classification"):
        raise ValueError(f"Invalid target_column: {target_column}")

    rng = np.random.default_rng(seed)
    lo, hi = num_events_range
    base = np.datetime64("2024-01-01T00:00:00")

    rows = []
    for i in range(num_records):
        epk_id = i + 1
        n = int(rng.integers(lo, hi + 1))
        start = base - np.timedelta64(int(rng.integers(1, 366)), "D")
        evt_dttm = [
            pd.Timestamp(start + np.timedelta64(j, "h")).to_pydatetime()
            for j in range(n)
        ]
        row = {
            "epk_id": epk_id,
            "evt_dttm": evt_dttm,
            "mcc": rng.integers(0, 6, n).tolist(),
            "price": np.round(rng.uniform(1.0, 1000.0, n), 2).tolist(),
            "event_ids": rng.integers(0, 2, n).tolist(),
        }
        if target_column == "classification":
            row["target"] = int(rng.integers(0, 2))
        if tabular_features:
            row["cat_features"] = rng.integers(
                epk_id * 5 - 4, epk_id * 5 + 1, 10
            ).tolist()
            row["num_features"] = (
                np.round(rng.uniform(1.0, 1000.0, 10), 2).astype("float32").tolist()
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df["partition"] = df["epk_id"] % num_output_partitions

    out = os.path.join(tempfile.mkdtemp(), "synthetic_sequence_data")
    df.to_parquet(out, partition_cols=["partition"], engine="pyarrow")
    return out


@pytest.fixture
def synth_sequence_dataset():
    """Return the :func:`generate_sequence_dataset` factory (no Spark)."""
    return generate_sequence_dataset
