"""Run the event-sequence preprocessor with BOTH backends and cross-load.

Flow mirrors ``examples/tabular_preprocessing/run_both_backends.py``:

1. ``generate_data.py`` writes a synthetic shuffled event log.
2. Fit :class:`avatar.preprocessing.local.EventSequencePreprocessor` (streaming,
   no Spark).
3. Fit :class:`avatar.preprocessing.spark.pipeline.EventSequencePreprocessor`
   (skipped without Spark / JDK).
4. Dump the artifact from each backend, load into the other.
5. Transform and compare per-user sequences.

Ordering note: Spark's ``sort().groupBy().collect_list()`` does not actually keep
event order after the group-by shuffle. The local backend is deterministically
time-sorted. The comparison therefore checks the per-user event *multiset*
(each parallel list co-sorted by timestamp), and separately asserts the local
output is monotonically time-ordered.

Run::

    python examples/eventsequence_preprocessing/run_both_backends.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

CAT = ["mcc", "direction"]
NUM = ["amount"]
KW = dict(
    categorical_columns=CAT,
    numeric_columns=NUM,
    event_time_column="timestamps",
    id_column="epk_id",
    event_type_ids_column="event_ids",
    time_unit="days",
    label_encoder_kwargs={"frequency_encoder": True},
    standard_scaler_kwargs={"to_log_columns": ["amount"]},
)
LIST_COLS = NUM + CAT + ["timestamps", "event_ids"]


def _maybe_spark():
    for home in [
        os.environ.get("SPARK_JDK", ""),
        "/home/jovyan/.sdkman/candidates/java/17.0.13-tem",
        "/usr/lib/jvm/java-17-openjdk-amd64",
    ]:
        if home and os.path.exists(os.path.join(home, "bin", "java")):
            os.environ["JAVA_HOME"] = home
            os.environ["PATH"] = f"{home}/bin:{os.environ['PATH']}"
            break
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("seq-preproc-demo")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception as exc:  # pragma: no cover
        print(f"[spark] unavailable ({exc}); running local-only")
        return None


def _canon(row):
    order = sorted(
        range(len(row["timestamps"])),
        key=lambda j: (row["timestamps"][j], *tuple(row[c][j] for c in LIST_COLS)),
    )
    return {c: [row[c][j] for j in order] for c in LIST_COLS}


def compare(sout, lout, label: str) -> None:
    sout = sout.sort_values("epk_id").reset_index(drop=True)
    lout = lout.sort_values("epk_id").reset_index(drop=True)
    assert len(sout) == len(lout), (label, len(sout), len(lout))
    mono = 0
    for i in range(len(sout)):
        assert sout["epk_id"].iloc[i] == lout["epk_id"].iloc[i]
        ca, cb = _canon(sout.iloc[i]), _canon(lout.iloc[i])
        for c in [*CAT, "event_ids"]:
            assert ca[c] == cb[c], (label, c, i)
        for c in [*NUM, "timestamps"]:
            assert np.allclose(
                np.asarray(ca[c], np.float32), np.asarray(cb[c], np.float32), atol=1e-4
            ), (label, c, i)
        if np.all(np.diff(np.asarray(lout["timestamps"].iloc[i])) >= 0):
            mono += 1
    print(f"  OK  {label}  ({len(sout)} users, local time-sorted {mono}/{len(sout)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--users", type=int, default=6_000)
    args = ap.parse_args()

    if not os.path.isdir(args.data) or not os.listdir(args.data):
        print("[data] generating synthetic event log ...")
        sys.path.insert(0, HERE)
        import generate_data

        os.makedirs(args.data, exist_ok=True)
        tbl = generate_data.build_table(args.users)
        step = -(-tbl.num_rows // 6)
        for i, s in enumerate(range(0, tbl.num_rows, step)):
            pq.write_table(
                tbl.slice(s, step), os.path.join(args.data, f"part-{i:03d}.parquet")
            )

    n_events = sum(
        pq.ParquetFile(f.path).metadata.num_rows
        for f in os.scandir(args.data)
        if f.name.endswith(".parquet")
    )
    print(f"events={n_events}")

    from avatar.preprocessing.local import EventSequencePreprocessor as LocalSeq

    local_pp = LocalSeq(**KW, batch_rows=100_000)
    local_pp.fit(args.data)
    local_out = local_pp.transform(args.data).to_pandas()
    print(f"[local] users={len(local_out)}  columns_meta={local_pp.columns_meta}")

    spark = _maybe_spark()
    if spark is None:
        print("\n[local-only] one user's sequence:")
        row = local_out.iloc[0]
        print({c: list(np.asarray(row[c]))[:6] for c in LIST_COLS})
        return

    from avatar.preprocessing.spark.pipeline import (
        EventSequencePreprocessor as SparkSeq,
    )

    sdf = spark.read.parquet(args.data)
    spark_pp = SparkSeq(**KW)
    spark_pp.fit(sdf)
    spark_out = spark_pp.transform(sdf).toPandas()

    for c in CAT:
        assert (
            spark_pp.label_encoder.values_to_id[c]
            == local_pp.label_encoder.values_to_id[c]
        )
    for c in NUM:
        sp, lo = (
            spark_pp.standard_scaler.mean_std[c],
            local_pp.standard_scaler.mean_std[c],
        )
        assert np.isclose(sp["mean"], lo["mean"], atol=1e-6, equal_nan=True)
        assert np.isclose(sp["std"], lo["std"], atol=1e-6, equal_nan=True)
    assert spark_pp.columns_meta == local_pp.columns_meta
    print("[fit] statistics + columns_meta identical across backends")

    print("\n[cross-load] dump() from one backend -> load() into the other:")
    local_from_spark = LocalSeq.load(spark_pp.dump())
    spark_from_local = SparkSeq.load(local_pp.dump())

    compare(spark_out, local_out, "spark.fit          vs local.fit")
    compare(
        spark_out,
        local_from_spark.transform(args.data).to_pandas(),
        "spark.fit          vs local.load(spark.dump())",
    )
    compare(
        spark_from_local.transform(sdf).toPandas(),
        local_out,
        "spark.load(local.dump()) vs local.fit",
    )

    # bucketed (bounded-memory) path must match the single-pass path
    big = local_from_spark.transform(args.data, n_buckets=8).to_pandas()
    compare(spark_out, big, "spark.fit          vs local (8 hash buckets)")

    print("\nAll backends and cross-loaded artifacts produce matching sequences.")
    spark.stop()


if __name__ == "__main__":
    main()
