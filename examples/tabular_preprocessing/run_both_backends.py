"""Run the tabular preprocessor with BOTH backends and cross-load the artifact.

Flow:

1. ``generate_data.py`` writes a synthetic wide table (run it first, or this
   script will do it automatically).
2. Fit :class:`avatar.preprocessing.local.TabularPreprocessor` -- a single
   streaming pass, no Spark.
3. Fit :class:`avatar.preprocessing.spark.pipeline.TabularPreprocessor` (skipped
   if no Spark / JDK is available).
4. Dump the artifact from each backend and load it into the *other* one.
5. Transform with every combination and assert the outputs are identical
   (categorical ids exactly, standardized numerics within 1e-4).

Run::

    python examples/tabular_preprocessing/run_both_backends.py
    # or point at your own data / columns:
    python examples/tabular_preprocessing/run_both_backends.py --data /path/to/parquet
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")


def _load_columns(path: str):
    names = pq.read_schema(
        sorted(f.path for f in os.scandir(path) if f.name.endswith(".parquet"))[0]
    ).names
    cat = [c for c in names if c.startswith("cat_")]
    num = [c for c in names if c.startswith("num_")]
    return cat, num


def _maybe_spark():
    """Return a UTC SparkSession or None."""
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
            SparkSession.builder.appName("tabular-preproc-demo")
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


def compare(a, b, label: str) -> None:
    """Assert two transformed tables match: ids exactly, numerics within 1e-4."""
    a = a.sort_values("epk_id").reset_index(drop=True)
    b = b.sort_values("epk_id").reset_index(drop=True)
    assert sorted(a.columns) == sorted(b.columns), (
        label,
        sorted(a.columns),
        sorted(b.columns),
    )
    for i in range(len(a)):
        assert list(a["cat_features"].iloc[i]) == list(b["cat_features"].iloc[i]), (
            label,
            i,
        )
        na = np.asarray(a["num_features"].iloc[i], dtype=np.float32)
        nb = np.asarray(b["num_features"].iloc[i], dtype=np.float32)
        assert np.allclose(na, nb, atol=1e-4, equal_nan=True), (label, i)
    print(f"  OK  {label}  ({len(a)} rows)")


def main() -> None:
    """Fit both backends, cross-load their artifacts and compare the output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--identity", nargs="*", default=["cat_0"])
    args = ap.parse_args()

    if not os.path.isdir(args.data) or not os.listdir(args.data):
        print("[data] generating synthetic dataset ...")
        sys.path.insert(0, HERE)
        import generate_data

        os.makedirs(args.data, exist_ok=True)
        tbl = generate_data.build_table()
        step = -(-tbl.num_rows // 6)
        for i, s in enumerate(range(0, tbl.num_rows, step)):
            pq.write_table(
                tbl.slice(s, step), os.path.join(args.data, f"part-{i:03d}.parquet")
            )

    cat_cols, num_cols = _load_columns(args.data)
    log_cols = num_cols[::4]  # matches generate_data's heavy-tailed columns
    print(
        f"categorical={len(cat_cols)}  numeric={len(num_cols)}  "
        f"to_log={log_cols}  identity={args.identity}"
    )

    kw = dict(
        categorical_columns=cat_cols,
        numeric_columns=num_cols,
        spec_tokens={"pad": 0},
        label_encoder_kwargs={"frequency_encoder": True},
        standard_scaler_kwargs={"to_log_columns": log_cols},
    )

    # ---- local backend --------------------------------------------------
    from avatar.preprocessing.local import TabularPreprocessor as LocalTab

    local_pp = LocalTab(**kw, batch_rows=50_000)
    local_pp.fit(args.data)
    local_out = local_pp.transform(args.data, identity_cols=args.identity).to_pandas()
    print(
        f"\n[local] vocab_size={local_pp.vocab_size}  offset_map={local_pp.offset_map}"
    )

    spark = _maybe_spark()
    if spark is None:
        print("\n[local-only] transform head:")
        print(local_out.head(3).to_string())
        return

    # ---- spark backend ------------------------------------------------------
    from avatar.preprocessing.spark.pipeline import TabularPreprocessor as SparkTab

    sdf = spark.read.parquet(args.data)
    spark_pp = SparkTab(**kw)
    spark_pp.fit(sdf)
    spark_out = spark_pp.transform(sdf, identity_cols=args.identity).toPandas()
    print(f"[spark] vocab_size={spark_pp.vocab_size}  offset_map={spark_pp.offset_map}")

    # ---- fit-stat parity -------------------------------------------------
    assert spark_pp.vocab_size == local_pp.vocab_size
    assert spark_pp.offset_map == local_pp.offset_map
    for c in cat_cols:
        assert (
            spark_pp.label_encoder.values_to_id[c]
            == local_pp.label_encoder.values_to_id[c]
        )
    for c in num_cols:
        sp, lo = (
            spark_pp.standard_scaler.mean_std[c],
            local_pp.standard_scaler.mean_std[c],
        )
        assert np.isclose(sp["mean"], lo["mean"], atol=1e-6, equal_nan=True)
        assert np.isclose(sp["std"], lo["std"], atol=1e-6, equal_nan=True)
    print("\n[fit] statistics identical across backends")

    # ---- cross-load ----------------------------------------------------
    print("\n[cross-load] dump() from one backend -> load() into the other:")
    local_from_spark = LocalTab.load(spark_pp.dump())
    spark_from_local = SparkTab.load(local_pp.dump())

    compare(spark_out, local_out, "spark.fit          vs local.fit")
    compare(
        spark_out,
        local_from_spark.transform(args.data, identity_cols=args.identity).to_pandas(),
        "spark.fit          vs local.load(spark.dump())",
    )
    compare(
        spark_from_local.transform(sdf, identity_cols=args.identity).toPandas(),
        local_out,
        "spark.load(local.dump()) vs local.fit",
    )

    print("\nAll backends and cross-loaded artifacts produce identical output.")
    spark.stop()


if __name__ == "__main__":
    main()
