from pathlib import Path

import polars as pl
import pytest

from avatar.automl.data import ParquetSource
from avatar.automl.exceptions import SchemaError


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
