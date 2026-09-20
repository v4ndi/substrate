from pathlib import Path

import polars as pl
import pytest

from fmlib.automl.data import ParquetSource
from fmlib.automl.exceptions import SchemaError


def test_parquet_manifest_is_deterministic(tmp_path):
    pl.DataFrame({"value": [2]}).write_parquet(tmp_path / "b.parquet")
    pl.DataFrame({"value": [1]}).write_parquet(tmp_path / "a.parquet")

    source = ParquetSource.resolve(tmp_path)

    assert [path.name for path in source.files] == ["a.parquet", "b.parquet"]
    assert source.read()["value"].to_list() == [1, 2]


def test_hive_partition_columns_above_and_below_split_root_are_preserved(tmp_path):
    global_root = tmp_path / "global_name=pr"
    split_root = global_root / "split_type=train"
    first_month = split_root / "month_part=2025-10-31"
    second_month = split_root / "month_part=2025-11-30"
    first_month.mkdir(parents=True)
    second_month.mkdir(parents=True)
    pl.DataFrame({"epk_id": [1], "feature": [0.1]}).write_parquet(
        first_month / "part-00000.parquet"
    )
    pl.DataFrame({"epk_id": [2], "feature": [0.2]}).write_parquet(
        first_month / "part-00001.parquet"
    )
    pl.DataFrame({"epk_id": [3], "feature": [0.3]}).write_parquet(
        second_month / "part-00000.parquet"
    )

    frame = ParquetSource.resolve(split_root).read().sort("epk_id")

    assert frame.columns == [
        "epk_id",
        "feature",
        "global_name",
        "split_type",
        "month_part",
    ]
    assert frame["global_name"].to_list() == ["pr", "pr", "pr"]
    assert frame["split_type"].to_list() == ["train", "train", "train"]
    assert frame["month_part"].to_list() == ["2025-10-31", "2025-10-31", "2025-11-30"]

    source = ParquetSource.resolve(split_root)
    assert source.unique_column_values("month_part") == (
        ("2025-10-31", "2025-11-30"),
        False,
    )


def test_unique_column_values_reads_only_requested_physical_column_and_detects_nulls(
    tmp_path,
):
    pl.DataFrame({"group": ["b", None], "unused": [1, 2]}).write_parquet(
        tmp_path / "first.parquet"
    )
    pl.DataFrame({"group": ["a", "b"], "unused": [3, 4]}).write_parquet(
        tmp_path / "second.parquet"
    )

    values, has_nulls = ParquetSource.resolve(tmp_path).unique_column_values("group")

    assert values == ("a", "b")
    assert has_nulls is True


def test_missing_path_is_rejected(tmp_path):
    with pytest.raises(SchemaError):
        ParquetSource.resolve(Path(tmp_path) / "missing")


def test_lazy_projection_preserves_hive_union_nulls_and_physical_precedence(
    tmp_path, monkeypatch
):
    first = tmp_path / "group=a"
    second = tmp_path / "group=b"
    first.mkdir()
    second.mkdir()
    pl.DataFrame({
        "id": [2, 1],
        "x": [1.0, None],
        "unused": ["large", "payload"],
    }).write_parquet(first / "a.parquet")
    pl.DataFrame({
        "id": [3],
        "group": ["physical"],
        "unused": ["payload"],
    }).write_parquet(second / "b.parquet")
    source = ParquetSource.resolve(tmp_path)

    def no_eager_read(*args, **kwargs):
        pytest.fail("Source must project a lazy scan before collecting")

    monkeypatch.setattr(pl, "read_parquet", no_eager_read)
    plan = source.scan().select("id", "x", "group")
    assert "PROJECT 2/3 COLUMNS" in plan.explain()
    result = source.read(("id", "x", "group"))
    assert result.to_dict(as_series=False) == {
        "id": [2, 1, 3],
        "x": [1.0, None, None],
        "group": ["a", "a", "physical"],
    }


# --------------------------------------------------------------------------- #
# What a source refuses, and the partition-only column path                    #
# --------------------------------------------------------------------------- #
def test_a_directory_without_parquet_is_refused_by_name(tmp_path):
    """Pointing at the parent of the data instead of the data is a common slip."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    (empty / "readme.txt").write_text("not parquet", encoding="utf-8")

    with pytest.raises(SchemaError, match="No parquet files found under"):
        ParquetSource.resolve(empty)


def test_a_file_that_is_not_parquet_is_refused(tmp_path):
    """The suffix is the check, so a renamed csv fails here rather than later."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(SchemaError, match="not a parquet file"):
        ParquetSource.resolve(path)


def test_a_corrupt_shard_names_the_source_it_could_not_read(tmp_path):
    """Polars' own message is kept, but wrapped so the caller learns which source."""
    root = tmp_path / "broken"
    root.mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(root / "good.parquet")
    (root / "bad.parquet").write_bytes(b"PAR1 definitely not a parquet file")

    source = ParquetSource.resolve(root)
    with pytest.raises(SchemaError, match="Failed to read parquet source"):
        source.read()


def test_a_column_missing_from_every_shard_is_refused(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    pl.DataFrame({"a": [1, 2]}).write_parquet(root / "part-0.parquet")

    source = ParquetSource.resolve(root)
    with pytest.raises(SchemaError, match="Column 'missing' is missing"):
        source.unique_column_values("missing")


def test_a_partition_only_column_yields_its_values_without_reading_the_data(tmp_path):
    """A Hive key is a column even though no shard contains it.

    Group routing asks for the distinct values of the group column, and when
    the data is partitioned by that column the answer lives in the path, not in
    the file. Returning nothing here would silently route every row to the
    global model.
    """
    root = tmp_path / "data"
    for group in ("retail", "corp"):
        directory = root / f"group={group}"
        directory.mkdir(parents=True)
        pl.DataFrame({"a": [1, 2]}).write_parquet(directory / "part-0.parquet")

    values, has_nulls = ParquetSource.resolve(root).unique_column_values("group")

    assert values == ("corp", "retail")
    assert has_nulls is False


def test_a_shard_outside_the_partition_layout_counts_as_a_null(tmp_path):
    """One stray file without the key means the column is not complete."""
    root = tmp_path / "data"
    (root / "group=retail").mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(root / "group=retail" / "part-0.parquet")
    pl.DataFrame({"a": [2]}).write_parquet(root / "loose.parquet")

    values, has_nulls = ParquetSource.resolve(root).unique_column_values("group")

    assert values == ("retail",)
    assert has_nulls is True


def test_a_percent_encoded_partition_value_is_decoded(tmp_path):
    """Hive escapes what it cannot put in a path; the value must come back whole."""
    directory = root_directory = tmp_path / "data" / "group=retail%2Fnorth"
    directory.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(directory / "part-0.parquet")

    values, _ = ParquetSource.resolve(root_directory.parent).unique_column_values(
        "group"
    )
    assert values == ("retail/north",)
