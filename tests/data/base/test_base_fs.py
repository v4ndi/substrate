"""Filesystem resolution and fs-aware parquet access.

The HDFS path cannot be exercised without a live namenode, so what is asserted
here is that URI resolution dispatches correctly and that every read path
behaves identically whether the filesystem is inferred or passed explicitly —
which is the whole contract the HDFS backend relies on.
"""

import os

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from fmlib.data.base import BaseParquetDataset
from fmlib.data.base.fs import (
    discover_parquet_files,
    file_size,
    is_local_filesystem,
    resolve_filesystem,
    resolve_filesystems,
)
from fmlib.data.base.parquet import parquet_num_rows, read_parquet_file


@pytest.fixture
def dataset_dir(tmp_path):
    """Two hive-partitioned parquet files plus the sidecars a writer leaves."""
    for index in range(2):
        part = tmp_path / f"partition={index}"
        part.mkdir()
        table = pa.table({
            "col1": [index * 10 + row for row in range(5)],
            "col2": ["a", "b", "c", "d", "e"],
        })
        pq.write_table(table, part / "data.parquet")
    (tmp_path / "_SUCCESS").write_text("")
    (tmp_path / ".hidden.parquet").write_text("not parquet")
    return tmp_path


def test_local_paths_resolve_to_the_local_filesystem(tmp_path):
    filesystem, path = resolve_filesystem(str(tmp_path))
    assert is_local_filesystem(filesystem)
    assert path == str(tmp_path)

    filesystem, path = resolve_filesystem(f"file://{tmp_path}")
    assert is_local_filesystem(filesystem)
    assert path == str(tmp_path)


def test_relative_paths_are_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    _, path = resolve_filesystem("sub")
    assert path == os.path.join(str(tmp_path), "sub")


def _resolve_hdfs_or_skip(uri: str = "hdfs://namenode:8020/user/team/data"):
    """Resolve an HDFS URI, skipping when the JVM client is not installed."""
    try:
        return resolve_filesystem(uri)
    except (OSError, ImportError) as error:
        pytest.skip(f"libhdfs is not available in this environment: {error}")


def test_hdfs_uri_selects_the_hadoop_filesystem():
    """The URI must route to HadoopFileSystem without contacting a namenode."""
    filesystem, path = _resolve_hdfs_or_skip()
    assert isinstance(filesystem, pafs.HadoopFileSystem)
    assert path == "/user/team/data"


def test_explicit_filesystem_overrides_the_uri(tmp_path):
    local = pafs.LocalFileSystem()
    filesystem, path = resolve_filesystem(
        f"hdfs://namenode:8020{tmp_path}", filesystem=local
    )
    assert filesystem is local
    assert path == str(tmp_path)


def test_unsupported_filesystem_options_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported filesystem type"):
        resolve_filesystem(str(tmp_path), filesystem={"type": "s3"})
    with pytest.raises(TypeError):
        resolve_filesystem(str(tmp_path), filesystem="hdfs")


def test_mixed_filesystems_are_rejected(tmp_path):
    _resolve_hdfs_or_skip("hdfs://namenode:8020/data")
    with pytest.raises(ValueError, match="same filesystem"):
        resolve_filesystems([str(tmp_path), "hdfs://namenode:8020/data"])


def test_discovery_skips_sidecars_and_sorts(dataset_dir):
    filesystem, path = resolve_filesystem(str(dataset_dir))
    files = discover_parquet_files(filesystem, path)
    assert len(files) == 2
    assert all(file.endswith("data.parquet") for file in files)
    assert files == sorted(files)
    assert not any("_SUCCESS" in file or "/.hidden" in file for file in files)


def test_discovery_rejects_a_missing_directory(tmp_path):
    filesystem, _ = resolve_filesystem(str(tmp_path))
    with pytest.raises(AssertionError, match="doesn't exist"):
        discover_parquet_files(filesystem, str(tmp_path / "nope"))


def test_reads_match_with_and_without_an_explicit_filesystem(dataset_dir):
    filesystem, path = resolve_filesystem(str(dataset_dir))
    files = discover_parquet_files(filesystem, path)

    for file in files:
        inferred = list(read_parquet_file(file, shuffle=False))
        explicit = list(read_parquet_file(file, shuffle=False, filesystem=filesystem))
        assert [record["col1"] for record in inferred] == [
            record["col1"] for record in explicit
        ]
        assert parquet_num_rows(file) == parquet_num_rows(file, filesystem) == 5
        assert file_size(filesystem, file) == os.path.getsize(file)


def test_dataset_accepts_a_file_uri(dataset_dir):
    dataset = BaseParquetDataset(
        path=f"file://{dataset_dir}", shuffle_files=False, shuffle_pq=False
    )
    assert len(dataset.files) == 2
    assert len(list(dataset)) == 10


def test_dataset_accepts_an_explicit_filesystem(dataset_dir):
    dataset = BaseParquetDataset(
        path=str(dataset_dir),
        filesystem=pafs.LocalFileSystem(),
        shuffle_files=False,
        shuffle_pq=False,
    )
    assert len(list(dataset)) == 10


def test_dataset_accepts_filesystem_options(dataset_dir):
    dataset = BaseParquetDataset(
        path=str(dataset_dir),
        filesystem={"type": "local"},
        shuffle_files=False,
        shuffle_pq=False,
    )
    assert len(list(dataset)) == 10
