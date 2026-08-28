import pytest
import torch
from pyspark.sql import SparkSession

from avatar.data.dataset import FixedHorizonDataset
from avatar.synth import generate_fixed_horizon_dataset


@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.appName("EventSequenceDataset").getOrCreate()


@pytest.fixture
def output_path(spark):
    output_path = generate_fixed_horizon_dataset(
        spark=spark,
        num_records=100,
        num_output_partitions=8,
        num_events_range=(1, 100),
        tabular_features=True,
    )
    return output_path


def test_min_max_lenght_and_types(output_path):
    """Tests for min/max_lenght and sequence dtypes"""
    dataset = FixedHorizonDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        target_columns=["targets_events", "targets_timestamps"],
        ctx_columns=["ctx_events", "ctx_timestamps"],
        selected_event_ids=None,
        min_length=10,
        max_length=20,
        has_tabular=False,
        read_columns=None,
        event_time_column=None,
        event_ids_column="event_ids",
        shuffle_files=False,
        shuffle_pq=False,
        ctx_max_length=15,
        ctx_min_length=5,
    )
    sample = next(iter(dataset))
    assert sample["mcc"].dtype == torch.long
    assert sample["price"].dtype == torch.float
    assert sample["ctx_events"].dtype == torch.long
    assert sample["ctx_timestamps"].dtype == torch.float
    assert sample["targets_events"].dtype == torch.long
    assert sample["targets_timestamps"].dtype == torch.float
    assert sample["mcc"].shape[0] >= 10
    assert sample["mcc"].shape[0] <= 20
    assert sample["ctx_events"].shape[0] >= 5
    assert sample["ctx_events"].shape[0] <= 15
    assert sample["ctx_timestamps"].shape[0] >= 5
    assert sample["ctx_timestamps"].shape[0] <= 15


def test_min_max_lenght_and_types_no_tabular(output_path):
    """Tests for min/max_lenght and sequence dtypes"""
    dataset = FixedHorizonDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        target_columns=[],
        ctx_columns=["ctx_events", "ctx_timestamps"],
        selected_event_ids=None,
        min_length=10,
        max_length=20,
        has_tabular=False,
        read_columns=None,
        event_time_column=None,
        event_ids_column="event_ids",
        shuffle_files=False,
        shuffle_pq=False,
        ctx_max_length=15,
        ctx_min_length=5,
    )
    sample = next(iter(dataset))
    assert sample["mcc"].dtype == torch.long
    assert sample["price"].dtype == torch.float
    assert sample["ctx_events"].dtype == torch.long
    assert sample["ctx_timestamps"].dtype == torch.float
    assert sample["mcc"].shape[0] >= 10
    assert sample["mcc"].shape[0] <= 20
    assert sample["ctx_events"].shape[0] >= 5
    assert sample["ctx_events"].shape[0] <= 15
    assert sample["ctx_timestamps"].shape[0] >= 5
    assert sample["ctx_timestamps"].shape[0] <= 15
