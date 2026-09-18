"""A parallel fit and transform must be the sequential one, only faster.

The two claims this file makes precise:

* the **vocabulary** is identical, bit for bit, because it is counted in
  integers and the ids come from an order that does not depend on traversal;
* the **numeric statistics** agree to rounding, not bit for bit, because
  floating-point addition is not associative and a different grouping of
  Chan's formula rounds differently. That is a property of the arithmetic, not
  a bug to be fixed, and stating it here is cheaper than rediscovering it.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fmlib.preprocessing.base.accumulators import (
    CardinalityError,
    MeanStdAccumulator,
    ValueCountAccumulator,
)
from fmlib.preprocessing.base.io import shard_files
from fmlib.preprocessing.local import TabularPreprocessor

CAT = ["cat_a", "cat_b"]
NUM = ["num_1", "num_2", "num_3"]
KW = dict(
    categorical_columns=CAT,
    numeric_columns=NUM,
    spec_tokens={"pad": 0},
    standard_scaler_kwargs={"to_log_columns": ["num_2"]},
)


def _batch(**columns) -> pa.RecordBatch:
    return pa.record_batch({name: pa.array(values) for name, values in columns.items()})


# --------------------------------------------------------------------------- #
# merge()                                                                      #
# --------------------------------------------------------------------------- #
def test_merging_mean_std_states_matches_one_pass_to_rounding():
    rng = np.random.default_rng(3)
    left, right = rng.normal(50, 12, 400), rng.exponential(3, 600)

    whole = MeanStdAccumulator(["x"])
    whole.update(_batch(x=left))
    whole.update(_batch(x=right))

    first, second = MeanStdAccumulator(["x"]), MeanStdAccumulator(["x"])
    first.update(_batch(x=left))
    second.update(_batch(x=right))
    merged = first.merge(second)

    assert merged.finalize()["x"]["mean"] == pytest.approx(
        whole.finalize()["x"]["mean"], rel=1e-12
    )
    assert merged.finalize()["x"]["std"] == pytest.approx(
        whole.finalize()["x"]["std"], rel=1e-12
    )
    combined = np.concatenate([left, right])
    assert merged.finalize()["x"]["mean"] == pytest.approx(combined.mean(), rel=1e-12)
    assert merged.finalize()["x"]["std"] == pytest.approx(
        combined.std(ddof=1), rel=1e-12
    )


def test_merging_an_empty_state_changes_nothing():
    fed = MeanStdAccumulator(["x"])
    fed.update(_batch(x=[1.0, 2.0, 3.0]))
    before = fed.finalize()
    merged = fed.merge(MeanStdAccumulator(["x"])).finalize()
    assert merged == before


def test_merging_value_counts_is_exact():
    left, right = ValueCountAccumulator(["c"]), ValueCountAccumulator(["c"])
    left.update(_batch(c=["a", "b", "a"]))
    right.update(_batch(c=["b", "c", "b"]))
    merged = left.merge(right)
    assert dict(merged._counts["c"]) == {"a": 2, "b": 3, "c": 1}
    assert merged.finalize({"pad": 0}) == {"c": {"pad": 0, "a": 1, "b": 2, "c": 3}}


def test_merging_can_trip_the_cardinality_guard():
    left = ValueCountAccumulator(["c"], max_cardinality=2)
    right = ValueCountAccumulator(["c"], max_cardinality=2)
    left.update(_batch(c=["a", "b"]))
    right.update(_batch(c=["c"]))
    with pytest.raises(CardinalityError, match="while merging parallel fit states"):
        left.merge(right)


@pytest.mark.parametrize(
    ("other", "message"),
    [
        (MeanStdAccumulator(["y"]), "different columns"),
        (MeanStdAccumulator(["x"], ["x"]), "to_log_columns"),
    ],
)
def test_merging_incompatible_mean_std_states_is_refused(other, message):
    with pytest.raises(ValueError, match=message):
        MeanStdAccumulator(["x"]).merge(other)


def test_merging_incompatible_value_count_states_is_refused():
    with pytest.raises(ValueError, match="cardinality policies"):
        ValueCountAccumulator(["c"]).merge(
            ValueCountAccumulator(["c"], max_cardinality=5)
        )


# --------------------------------------------------------------------------- #
# sharding                                                                     #
# --------------------------------------------------------------------------- #
def test_shards_are_contiguous_whole_files_in_sorted_order(
    write_parquet, tabular_table
):
    directory = write_parquet(tabular_table, n_files=7)
    groups = shard_files(directory, 3)
    assert [len(group) for group in groups] == [3, 2, 2]
    assert [path for group in groups for path in group] == sorted(
        glob.glob(os.path.join(directory, "*.parquet"))
    )


def test_more_shards_than_files_yields_one_shard_per_file(write_parquet, tabular_table):
    directory = write_parquet(tabular_table, n_files=2)
    assert len(shard_files(directory, 16)) == 2


def test_sharding_refuses_a_built_dataset(write_parquet, tabular_table):
    import pyarrow.dataset as ds

    directory = write_parquet(tabular_table, n_files=2)
    with pytest.raises(TypeError, match="needs file paths"):
        shard_files(ds.dataset(directory, format="parquet"), 2)


# --------------------------------------------------------------------------- #
# parallel fit                                                                 #
# --------------------------------------------------------------------------- #
def test_parallel_fit_reproduces_the_sequential_artifact(write_parquet, tabular_table):
    """Vocabulary exactly, statistics to rounding -- the two halves of the claim."""
    directory = write_parquet(tabular_table, n_files=6)
    sequential = TabularPreprocessor(**KW).fit(directory)
    parallel = TabularPreprocessor(**KW).fit(directory, num_workers=3)

    assert parallel.label_encoder.values_to_id == sequential.label_encoder.values_to_id
    assert parallel.offset_map == sequential.offset_map
    assert parallel.vocab_size == sequential.vocab_size

    expected = sequential.standard_scaler.mean_std
    observed = parallel.standard_scaler.mean_std
    assert set(observed) == set(expected)
    for column, stats in expected.items():
        assert observed[column]["mean"] == pytest.approx(stats["mean"], rel=1e-12)
        assert observed[column]["std"] == pytest.approx(stats["std"], rel=1e-12)


def test_one_worker_is_the_sequential_path_bit_for_bit(write_parquet, tabular_table):
    directory = write_parquet(tabular_table, n_files=6)
    sequential = TabularPreprocessor(**KW).fit(directory).dump()
    single = TabularPreprocessor(**KW).fit(directory, num_workers=1).dump()
    assert single == sequential


def test_the_one_traversal_dependent_order_is_refused_in_parallel(
    write_parquet, tabular_table
):
    directory = write_parquet(tabular_table, n_files=4)
    pipeline = TabularPreprocessor(
        **KW | {"label_encoder_kwargs": {"order": "first_seen"}}
    )
    with pytest.raises(ValueError, match="first_seen"):
        pipeline.fit(directory, num_workers=2)
    assert pipeline.fit(directory, num_workers=1) is pipeline


def test_count_descending_order_survives_a_parallel_fit(write_parquet, tabular_table):
    directory = write_parquet(tabular_table, n_files=6)
    kwargs = KW | {"label_encoder_kwargs": {"order": "count_desc"}}
    sequential = TabularPreprocessor(**kwargs).fit(directory)
    parallel = TabularPreprocessor(**kwargs).fit(directory, num_workers=2)
    assert parallel.label_encoder.values_to_id == sequential.label_encoder.values_to_id


# --------------------------------------------------------------------------- #
# parallel transform                                                           #
# --------------------------------------------------------------------------- #
def test_parallel_transform_writes_one_part_per_shard_and_the_same_rows(
    write_parquet, tabular_table, tmp_path
):
    directory = write_parquet(tabular_table, n_files=6)
    pipeline = TabularPreprocessor(**KW).fit(directory)

    single = tmp_path / "one.parquet"
    pipeline.transform(directory, str(single), identity_cols=["epk_id"])

    many = tmp_path / "parts"
    parts = pipeline.transform(
        directory, str(many), identity_cols=["epk_id"], num_workers=3
    )

    assert [os.path.basename(path) for path in parts] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    expected = pq.read_table(single)
    observed = pa.concat_tables([pq.read_table(path) for path in parts])
    assert observed.schema == expected.schema
    assert observed.num_rows == expected.num_rows
    assert observed.sort_by("epk_id").equals(expected.sort_by("epk_id"))


def test_parallel_transform_needs_somewhere_to_write(write_parquet, tabular_table):
    directory = write_parquet(tabular_table, n_files=4)
    pipeline = TabularPreprocessor(**KW).fit(directory)
    with pytest.raises(ValueError, match="needs an output directory"):
        pipeline.transform(directory, None, num_workers=2)


def test_fit_transform_carries_the_worker_count_through(
    write_parquet, tabular_table, tmp_path
):
    directory = write_parquet(tabular_table, n_files=4)
    parts = TabularPreprocessor(**KW).fit_transform(
        directory, str(tmp_path / "out"), identity_cols=["epk_id"], num_workers=2
    )
    assert len(parts) == 2
    assert sum(pq.ParquetFile(path).metadata.num_rows for path in parts) == (
        tabular_table.num_rows
    )


def test_peak_rows_per_part_stay_bounded_by_the_shard(
    write_parquet, tabular_table, tmp_path
):
    """Row counts follow the file split, which is what makes reads parallel."""
    directory = write_parquet(tabular_table, n_files=8)
    pipeline = TabularPreprocessor(**KW).fit(directory)
    parts = pipeline.transform(directory, str(tmp_path / "out"), num_workers=2)
    counts = [pq.ParquetFile(path).metadata.num_rows for path in parts]
    assert sum(counts) == tabular_table.num_rows
    assert all(count > 0 for count in counts)
    assert max(counts) <= math.ceil(tabular_table.num_rows / 2)
