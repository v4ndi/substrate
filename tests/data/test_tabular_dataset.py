import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from avatar.data.dataset import TabularDataset


@pytest.fixture
def create_parquet_file():
    """
    Creates a temporary Parquet file with required columns and returns the file path.
    """
    with TemporaryDirectory() as temp_dir:
        data = {
            "id": [1, 2, 3],
            "target": [0, 1, 0],
            "cat_features": [[0, 1], [2, 3], [4, 5]],  # Categorical features (lists)
            "num_features": [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],  # Numerical features (lists)
        }
        df = pd.DataFrame(data)
        # Convert to Arrow Table
        table = pa.Table.from_pandas(df)
        # Write the Arrow Table to a Parquet file
        parquet_path = os.path.join(temp_dir, "test_data.parquet")
        pq.write_table(table, parquet_path)

        yield parquet_path


def test_check_tabular_features(create_parquet_file):
    """
    Test the check_tabular_features method.
    """
    dataset = TabularDataset(path=os.path.dirname(create_parquet_file))

    # Check if the tabular features exist
    assert dataset.check_tabular_features() is True


def test_process_tabular(create_parquet_file):
    """
    Test the process_tabular method.
    """
    dataset = TabularDataset(path=os.path.dirname(create_parquet_file))

    # Load a sample record from the Parquet file
    sample_record = {
        "cat_features": np.array([0, 1]),
        "num_features": np.array([0.1, 0.2]),
    }

    dataset.process_tabular(sample_record)

    # Check if the processed features are torch tensors and have correct types
    assert isinstance(sample_record["tab_features"]["cat_features"], torch.Tensor)
    assert isinstance(sample_record["tab_features"]["num_features"], torch.Tensor)
    assert sample_record["tab_features"]["cat_features"].dtype == torch.long
    assert sample_record["tab_features"]["num_features"].dtype == torch.float


def test_process_tabular_hidden_state(create_parquet_file):
    """
    Test the process_tabular method with hidden_state
    """
    dataset = TabularDataset(path=os.path.dirname(create_parquet_file))

    # Load a sample record from the Parquet file
    sample_record = {
        "cat_features": np.array([0, 1]),
        "num_features": np.array([0.1, 0.2]),
        "hidden_state": np.array([0.1, 0.2, 0.12, 0.17, 0.25]),
    }

    dataset.process_tabular(
        record=sample_record, hidden_state_columns=["hidden_state"]
    )
    # Check if the processed features are torch tensors and have correct types
    assert isinstance(sample_record["tab_features"]["cat_features"], torch.Tensor)
    assert isinstance(sample_record["tab_features"]["num_features"], torch.Tensor)
    assert sample_record["tab_features"]["cat_features"].dtype == torch.long
    assert sample_record["tab_features"]["num_features"].dtype == torch.float
    hidden_states = sample_record["tab_features"]["_hidden_states"]
    assert set(hidden_states) == {"hidden_state"}
    assert hidden_states["hidden_state"].dtype == torch.float


def test_process(create_parquet_file):
    """
    Test the process method.
    """
    dataset = TabularDataset(path=os.path.dirname(create_parquet_file))

    # Load a sample record from the Parquet file
    sample_record = {
        "id": 1,
        "target": 0,
        "cat_features": np.array([0, 1]),
        "num_features": np.array([0.1, 0.2]),
    }

    dataset.process(sample_record)

    # Check if the processed record contains all keys, including "tab_features"
    assert "tab_features" in sample_record
    assert isinstance(sample_record["tab_features"], dict)
    assert "cat_features" in sample_record["tab_features"]
    assert "num_features" in sample_record["tab_features"]

    # Check that other keys are passed through
    assert sample_record["id"] == 1
    assert sample_record["target"] == 0


if __name__ == "__main__":
    pytest.main()
