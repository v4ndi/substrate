import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from avatar.data.parquet import parquet_num_rows, read_parquet_file


# Fixture to create a temporary parquet file for testing
@pytest.fixture
def temp_parquet_file():
    # Create a sample table
    data = {
        "col1": [1, 2, 3, 4, 5],
        "col2": ["a", "b", "c", "d", "e"],
        "col3": [1.1, 2.2, 3.3, 4.4, 5.5],
    }
    table = pa.Table.from_pydict(data)

    # Write the table to a temporary parquet file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
        pq.write_table(table, f.name)
        yield f.name  # Provide the file path to the test

    # Clean up the temporary file after the test
    os.unlink(f.name)


# Test for `parquet_num_rows`
def test_parquet_num_rows(temp_parquet_file):
    # Test that the function returns the correct number of rows
    assert parquet_num_rows(temp_parquet_file) == 5


# Test for `read_parquet_file` without shuffling
def test_read_parquet_file_no_shuffle(temp_parquet_file):
    records = list(read_parquet_file(temp_parquet_file, shuffle=False))
    # Verify the number of records
    assert len(records) == 5
    # Verify the content of the first record
    assert records[0] == {"col1": 1, "col2": "a", "col3": 1.1}
    # Verify the content of the last record
    assert records[-1] == {"col1": 5, "col2": "e", "col3": 5.5}


# Test for `read_parquet_file` with shuffling
def test_read_parquet_file_with_shuffle(temp_parquet_file):
    records = list(read_parquet_file(temp_parquet_file, shuffle=True))
    # Verify the number of records
    assert len(records) == 5
    # Verify that the records are shuffled (order is different from the original)
    original_order = [
        {"col1": 1, "col2": "a", "col3": 1.1},
        {"col1": 2, "col2": "b", "col3": 2.2},
        {"col1": 3, "col2": "c", "col3": 3.3},
        {"col1": 4, "col2": "d", "col3": 4.4},
        {"col1": 5, "col2": "e", "col3": 5.5},
    ]
    assert records != original_order
    # Verify that all records are present (just in a different order)
    assert sorted(records, key=lambda x: x["col1"]) == sorted(
        original_order, key=lambda x: x["col1"]
    )


# Test for `read_parquet_file` with specific columns
def test_read_parquet_file_specific_columns(temp_parquet_file):
    records = list(
        read_parquet_file(temp_parquet_file, columns=["col1", "col3"], shuffle=False)
    )
    # Verify the number of records
    assert len(records) == 5
    # Verify that only the specified columns are present
    assert all(set(record.keys()) == {"col1", "col3"} for record in records)
    # Verify the content of the first record
    assert records[0] == {"col1": 1, "col3": 1.1}
