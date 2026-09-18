import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader

from fmlib.data.base import BaseParquetDataset


# Fixture to create a temporary directory with multiple parquet files
@pytest.fixture
def temp_parquet_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple parquet files with more rows
        num_files = 5  # Increase the number of files
        rows_per_file = 10  # Increase the number of rows per file

        for i in range(num_files):
            data = {
                "col1": list(range(i * rows_per_file + 1, (i + 1) * rows_per_file + 1)),
                "col2": [
                    chr(97 + j) for j in range(rows_per_file)
                ],  # 'a', 'b', 'c', ...
                "col3": [
                    float((i * rows_per_file + j + 1) * 1.1)
                    for j in range(rows_per_file)
                ],
            }
            table = pa.Table.from_pydict(data)
            pq.write_table(table, os.path.join(temp_dir, f"file_{i}.parquet"))

        yield temp_dir  # Provide the directory path to the test


# Test for BaseIterDataset
def test_base_iter_dataset(temp_parquet_dir):
    # Initialize the dataset
    dataset = BaseParquetDataset(
        path=temp_parquet_dir, shuffle_files=False, shuffle_pq=False
    )
    # Test __len__
    assert len(dataset) == 50  # 5 files * 10 rows per file

    # Test files_per_worker without workers
    iter_start, iter_end = dataset.files_per_worker()
    assert iter_start == 0
    assert iter_end == 5  # 5 files in total

    # Test shuffle_files
    original_files = dataset.files.copy()
    dataset.shuffle_files()
    assert dataset.files != original_files  # Files should be shuffled


# Test for IterDataset without shuffling
def test_iter_dataset_no_shuffle(temp_parquet_dir):
    # Initialize the dataset
    dataset = BaseParquetDataset(
        path=temp_parquet_dir, shuffle_files=False, shuffle_pq=False
    )
    dataset.files = sorted(
        dataset.files, key=lambda x: int(x.split("/")[-1][5])
    )  # Number of files must be less then 10
    # Collect all records
    records = list(dataset)

    # Verify the number of records
    assert len(records) == 50

    # Verify the content of the records (order should match the original files)
    expected_records = []
    for i in range(5):  # 5 files
        for j in range(10):  # 10 rows per file
            expected_records.append({
                "col1": i * 10 + j + 1,
                "col2": chr(97 + j),
                "col3": float((i * 10 + j + 1) * 1.1),
            })
    assert records == expected_records


# Test for IterDataset with shuffling
def test_iter_dataset_with_shuffle(temp_parquet_dir):
    # Initialize the dataset
    dataset = BaseParquetDataset(
        path=temp_parquet_dir, shuffle_files=True, shuffle_pq=True
    )

    # Collect all records
    records = list(dataset)

    # Verify the number of records
    assert len(records) == 50

    # Verify that the records are shuffled (order is different from the original)
    expected_records = []
    for i in range(5):  # 5 files
        for j in range(10):  # 10 rows per file
            expected_records.append({
                "col1": i * 10 + j + 1,
                "col2": chr(97 + j),
                "col3": float((i * 10 + j + 1) * 1.1),
            })
    assert records != expected_records  # Order should be different
    assert sorted(records, key=lambda x: x["col1"]) == sorted(
        expected_records, key=lambda x: x["col1"]
    )  # Content should match


# Test for IterDataset with specific columns
def test_iter_dataset_specific_columns(temp_parquet_dir):
    # Initialize the dataset with specific columns
    dataset = BaseParquetDataset(
        path=temp_parquet_dir,
        read_columns=["col1", "col3"],
        shuffle_files=False,
        shuffle_pq=False,
    )
    dataset.files = sorted(dataset.files, key=lambda x: int(x.split("/")[-1][5]))
    # Collect all records
    records = list(dataset)

    # Verify the number of records
    assert len(records) == 50

    # Verify that only the specified columns are present
    assert all(set(record.keys()) == {"col1", "col3"} for record in records)

    # Verify the content of the records
    expected_records = []
    for i in range(5):  # 5 files
        for j in range(10):  # 10 rows per file
            expected_records.append({
                "col1": i * 10 + j + 1,
                "col3": float((i * 10 + j + 1) * 1.1),
            })
    assert records == expected_records


# Test for IterDataset with multiple workers
def test_iter_dataset_multiple_workers(temp_parquet_dir):
    # Initialize the dataset
    dataset = BaseParquetDataset(
        path=temp_parquet_dir, shuffle_files=False, shuffle_pq=False
    )

    # Create a DataLoader with 2 workers
    dataloader = DataLoader(dataset, num_workers=2, batch_size=None)

    # Collect all records
    records = list(dataloader)

    # Verify the number of records
    assert len(records) == 50

    # Verify the content of the records (order may vary due to parallel processing)
    expected_records = []
    for i in range(5):  # 5 files
        for j in range(10):  # 10 rows per file
            expected_records.append({
                "col1": i * 10 + j + 1,
                "col2": chr(97 + j),
                "col3": float((i * 10 + j + 1) * 1.1),
            })
    assert sorted(records, key=lambda x: x["col1"]) == sorted(
        expected_records, key=lambda x: x["col1"]
    )
