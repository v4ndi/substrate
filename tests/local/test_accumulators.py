"""Unit tests for the backend-agnostic streaming accumulators."""

import numpy as np
import pyarrow as pa
import pytest

from avatar.preprocessing.base.accumulators import (
    CardinalityError,
    MeanStdAccumulator,
    ValueCountAccumulator,
)
from avatar.preprocessing.base.encode import CategoricalMapper, signed_log1p


def _feed(acc, table, chunk):
    for b in table.to_batches(max_chunksize=chunk):
        acc.update(b)
    return acc


@pytest.mark.parametrize("chunk", [1, 7, 100, 10_000])
def test_mean_std_matches_numpy_sample_std(chunk):
    rng = np.random.default_rng(0)
    x = rng.normal(50, 12, size=9999)
    t = pa.table({"a": x})
    ms = _feed(MeanStdAccumulator(["a"]), t, chunk).finalize()["a"]
    assert ms["mean"] == pytest.approx(x.mean(), abs=1e-9)
    assert ms["std"] == pytest.approx(x.std(ddof=1), abs=1e-9)


def test_mean_std_excludes_nulls_keeps_data_nan():
    t = pa.table({"a": pa.array([1.0, 2.0, 3.0, None])})
    ms = _feed(MeanStdAccumulator(["a"]), t, 2).finalize()["a"]
    assert ms["mean"] == pytest.approx(2.0)
    t2 = pa.table({"a": pa.array([1.0, 2.0, np.nan])})
    ms2 = _feed(MeanStdAccumulator(["a"]), t2, 2).finalize()["a"]
    assert np.isnan(ms2["mean"]) and np.isnan(ms2["std"])


def test_mean_std_applies_signed_log1p():
    xs = np.array([0.0, 1.0, 3.0, -2.0])
    t = pa.table({"a": xs})
    ms = _feed(MeanStdAccumulator(["a"], to_log_columns=["a"]), t, 3).finalize()["a"]
    logged = signed_log1p(xs)
    assert ms["mean"] == pytest.approx(logged.mean())
    assert ms["std"] == pytest.approx(logged.std(ddof=1))


def test_mean_std_single_value_warns_and_nans():
    t = pa.table({"a": [5.0]})
    with pytest.warns(UserWarning):
        ms = _feed(MeanStdAccumulator(["a"]), t, 1).finalize()["a"]
    assert ms["mean"] == 5.0 and np.isnan(ms["std"])


@pytest.mark.parametrize("chunk", [1, 3, 1000])
def test_value_counts_frequency_order_deterministic(chunk):
    t = pa.table({"c": ["B", "B", "B", "A", "A", None, "C"]})
    vc = _feed(ValueCountAccumulator(["c"]), t, chunk)
    assert vc.finalize({"unk": 0}, frequency_encoder=True)["c"] == {
        "unk": 0, "B": 1, "A": 2, "C": 3,
    }
    assert vc.finalize({"unk": 0}, order="sorted")["c"] == {
        "unk": 0, "A": 1, "B": 2, "C": 3,
    }


def test_value_counts_cardinality_guard():
    t = pa.table({"c": list(range(50))})
    vc = ValueCountAccumulator(["c"], max_cardinality=10)
    with pytest.raises(CardinalityError):
        _feed(vc, t, 8)
    vc2 = _feed(ValueCountAccumulator(["c"], max_cardinality=10, on_overflow="topk"),
                pa.table({"c": [1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}), 4)
    kept = vc2.finalize({"unk": 0}, order="count_desc")["c"]
    assert len(kept) == 11 and kept[1] == 1  # most frequent gets id 1


def test_value_counts_spec_token_collision():
    vc = _feed(ValueCountAccumulator(["c"]), pa.table({"c": ["unk", "a"]}), 2)
    with pytest.raises(ValueError):
        vc.finalize({"unk": 0})


def test_categorical_mapper_int_and_str_paths():
    m = CategoricalMapper({"a": 2, "b": 1, "c": 3})
    np.testing.assert_array_equal(m(["a", "b", "x", None]), [2, 1, 0, 0])
    mi = CategoricalMapper({10: 1, 20: 2, -1: 3})
    np.testing.assert_array_equal(mi([10, -1, 999, 20]), [1, 3, 0, 2])
    np.testing.assert_array_equal(
        mi([10, None], null_mask=np.array([False, True])), [1, 0]
    )
