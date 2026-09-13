"""Parity between the local backend and the Spark backend.

Skipped when a Spark / JDK environment is not available (see conftest). What is
checked: fit statistics, cross-loading an artifact from one backend into the
other, and identical ``transform`` output.

Note on ordering: Spark's ``sort().groupBy().collect_list()`` does not preserve
event order after the group-by shuffle, so sequence parity is asserted on the
per-group event *multiset* (each parallel list co-sorted by timestamp). The
local backend is always deterministically time-sorted.
"""

import numpy as np

CAT = ["cat_a", "cat_b"]
NUM = ["num_1", "num_2", "num_3"]
LOG = ["num_2"]


def _spark_df(spark_session, path):
    return spark_session.read.parquet(path)


def _cmp_tabular(a, b):
    a = a.sort_values("epk_id").reset_index(drop=True)
    b = b.sort_values("epk_id").reset_index(drop=True)
    assert sorted(a.columns) == sorted(b.columns)
    for i in range(len(a)):
        assert list(a["cat_features"].iloc[i]) == list(b["cat_features"].iloc[i]), i
        na = np.asarray(a["num_features"].iloc[i], dtype=np.float32)
        nb = np.asarray(b["num_features"].iloc[i], dtype=np.float32)
        assert np.allclose(na, nb, atol=1e-4, equal_nan=True), i


def test_tabular_fit_and_cross_load(spark_session, write_parquet, tabular_table):
    from avatar.preprocessing.local import TabularPreprocessor as Local
    from avatar.preprocessing.spark.pipeline import TabularPreprocessor as Spark

    d = write_parquet(tabular_table)
    kw = dict(
        categorical_columns=CAT,
        numeric_columns=NUM,
        spec_tokens={"pad": 0},
        label_encoder_kwargs={"frequency_encoder": True},
        standard_scaler_kwargs={"to_log_columns": LOG},
    )
    sdf = _spark_df(spark_session, d)
    spark_pp = Spark(**kw)
    spark_pp.fit(sdf)
    local_pp = Local(**kw, batch_rows=1000)
    local_pp.fit(d)

    assert spark_pp.vocab_size == local_pp.vocab_size
    assert spark_pp.offset_map == local_pp.offset_map
    for c in CAT:
        assert (
            spark_pp.label_encoder.values_to_id[c]
            == local_pp.label_encoder.values_to_id[c]
        ), c
    for c in NUM:
        sm = spark_pp.standard_scaler.mean_std[c]
        lm = local_pp.standard_scaler.mean_std[c]
        assert np.isclose(sm["mean"], lm["mean"], atol=1e-6, equal_nan=True)
        assert np.isclose(sm["std"], lm["std"], atol=1e-6, equal_nan=True)

    # spark artifact -> local
    local_from_spark = Local.load(spark_pp.dump())
    _cmp_tabular(
        spark_pp.transform(sdf, identity_cols=["cat_b"]).toPandas(),
        local_from_spark.transform(d, identity_cols=["cat_b"]).to_pandas(),
    )
    # local artifact -> spark
    spark_from_local = Spark.load(local_pp.dump())
    _cmp_tabular(
        spark_from_local.transform(sdf, identity_cols=["cat_b"]).toPandas(),
        local_pp.transform(d, identity_cols=["cat_b"]).to_pandas(),
    )


def _canon(row, cols):
    order = sorted(
        range(len(row["timestamps"])),
        key=lambda j: (row["timestamps"][j], *tuple(row[c][j] for c in cols)),
    )
    return {c: [row[c][j] for j in order] for c in cols}


def test_sequence_fit_and_cross_load(spark_session, write_parquet, sequence_table):
    from avatar.preprocessing.local import EventSequencePreprocessor as Local
    from avatar.preprocessing.spark.pipeline import EventSequencePreprocessor as Spark

    d = write_parquet(sequence_table)
    kw = dict(
        categorical_columns=["mcc", "direction"],
        numeric_columns=["price"],
        event_time_column="timestamps",
        id_column="epk_id",
        event_type_ids_column="event_ids",
        time_unit="days",
        label_encoder_kwargs={"frequency_encoder": True},
    )
    sdf = _spark_df(spark_session, d)
    spark_pp = Spark(**kw)
    spark_pp.fit(sdf)
    local_pp = Local(**kw, batch_rows=700)
    local_pp.fit(d)

    for c in ["mcc", "direction"]:
        assert (
            spark_pp.label_encoder.values_to_id[c]
            == local_pp.label_encoder.values_to_id[c]
        )
    assert spark_pp.columns_meta == local_pp.columns_meta

    sout = (
        spark_pp.transform(sdf).toPandas().sort_values("epk_id").reset_index(drop=True)
    )
    lout = (
        Local
        .load(spark_pp.dump())
        .transform(d)
        .to_pandas()
        .sort_values("epk_id")
        .reset_index(drop=True)
    )
    list_cols = ["price", "mcc", "direction", "timestamps", "event_ids"]
    assert len(sout) == len(lout)
    for i in range(len(sout)):
        assert sout["epk_id"].iloc[i] == lout["epk_id"].iloc[i]
        ca, cb = _canon(sout.iloc[i], list_cols), _canon(lout.iloc[i], list_cols)
        for c in ["mcc", "direction", "event_ids"]:
            assert ca[c] == cb[c], (c, i)
        for c in ["price", "timestamps"]:
            assert np.allclose(
                np.asarray(ca[c], np.float32), np.asarray(cb[c], np.float32), atol=1e-4
            )
        assert np.all(np.diff(np.asarray(lout["timestamps"].iloc[i])) >= 0)
