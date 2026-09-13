"""Local-only tests for ``avatar.preprocessing.local.EventSequencePreprocessor``."""

import numpy as np
import pytest

from avatar.preprocessing.local import EventSequencePreprocessor

KW = dict(
    categorical_columns=["mcc", "direction"],
    numeric_columns=["price"],
    event_time_column="timestamps",
    id_column="epk_id",
    event_type_ids_column="event_ids",
    time_unit="days",
    label_encoder_kwargs={"frequency_encoder": True},
)


def test_output_structure_and_time_sorted(write_parquet, sequence_table):
    d = write_parquet(sequence_table)
    pp = EventSequencePreprocessor(**KW).fit(d)
    out = pp.transform(d).to_pandas()
    assert len(out) == 200
    for col in ["mcc", "direction", "price", "timestamps", "event_ids"]:
        assert col in out.columns
    for ts in out["timestamps"]:
        assert np.all(np.diff(np.asarray(ts)) >= 0), "events must be time-sorted"
    # every parallel list in a row has the same length
    for _, row in out.iterrows():
        lengths = {len(row[c]) for c in ["mcc", "direction", "price", "timestamps"]}
        assert len(lengths) == 1


@pytest.mark.parametrize("n_buckets", [1, 3, 16])
def test_bucketing_invariance(write_parquet, sequence_table, n_buckets):
    d = write_parquet(sequence_table)
    pp = EventSequencePreprocessor(**KW, batch_rows=500).fit(d)
    ref = (
        pp
        .transform(d, n_buckets=1)
        .to_pandas()
        .sort_values("epk_id")
        .reset_index(drop=True)
    )
    got = (
        pp
        .transform(d, n_buckets=n_buckets)
        .to_pandas()
        .sort_values("epk_id")
        .reset_index(drop=True)
    )
    assert list(ref["epk_id"]) == list(got["epk_id"])
    for c in ["mcc", "price", "timestamps", "event_ids"]:
        for i in range(len(ref)):
            assert list(np.asarray(ref[c].iloc[i])) == list(np.asarray(got[c].iloc[i]))


def test_columns_meta(write_parquet, sequence_table):
    d = write_parquet(sequence_table)
    pp = EventSequencePreprocessor(**KW).fit(d)
    meta = pp.columns_meta
    assert meta["price"] == {"type": "numeric", "n_classes": 1}
    assert meta["mcc"]["type"] == "categorical"
    assert meta["mcc"]["n_classes"] == len(pp.label_encoder.values_to_id["mcc"])


def test_dump_load_roundtrip(write_parquet, sequence_table):
    import yaml

    d = write_parquet(sequence_table)
    pp = EventSequencePreprocessor(**KW).fit(d)
    pp2 = EventSequencePreprocessor.load(yaml.safe_load(yaml.safe_dump(pp.dump())))
    a = pp.transform(d).to_pandas().sort_values("epk_id").reset_index(drop=True)
    b = pp2.transform(d).to_pandas().sort_values("epk_id").reset_index(drop=True)
    for c in ["mcc", "price", "timestamps"]:
        for i in range(len(a)):
            assert list(np.asarray(a[c].iloc[i])) == list(np.asarray(b[c].iloc[i]))


def test_groupby_columns_with_null_key(write_parquet, sequence_table):
    d = write_parquet(sequence_table)
    pp = EventSequencePreprocessor(
        categorical_columns=["mcc"],
        numeric_columns=None,
        event_time_column="timestamps",
        id_column="epk_id",
        event_type_ids_column="event_ids",
        groupby_columns=["direction"],
        label_encoder_kwargs={"frequency_encoder": True},
    ).fit(d)
    out = pp.transform(d).to_pandas()
    assert "direction" in out.columns
    assert out["direction"].isna().any()  # null-direction group is kept
