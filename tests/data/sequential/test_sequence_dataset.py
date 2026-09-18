import numpy as np
import pandas as pd
import pytest
import torch

from fmlib.data.sequential import EventSequenceCollateFn, EventSequenceDataset


@pytest.fixture
def output_path(synth_sequence_dataset):
    return synth_sequence_dataset(
        num_records=100,
        num_output_partitions=8,
        num_events_range=(1, 100),
        target_column="classification",
        tabular_features=True,
    )


def test_min_max_length_and_types(output_path):
    """Tests for min/max_length and sequence dtypes"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        selected_event_ids=None,
        min_length=10,
        max_length=20,
        has_tabular=False,
        read_columns=None,
        event_time_column=None,
        event_ids_column="event_ids",
        shuffle_files=False,
        shuffle_pq=False,
    )
    collate_fn = EventSequenceCollateFn(
        sequence_columns=["mcc", "price"],
        create_attention_mask=True,
        target_column=None,
        is_regression=False,
        has_tabular=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, collate_fn=collate_fn, batch_size=1
    )
    for features in dataloader:
        assert features["seq_features"]["mcc"].dtype == torch.long
        assert features["seq_features"]["price"].dtype == torch.float
        assert features["seq_features"]["mcc"].shape[1] >= 10
        assert features["seq_features"]["mcc"].shape[1] <= 20


def test_has_tab_features(output_path):
    """Tests for min/max_length and sequence dtypes"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        selected_event_ids=None,
        min_length=10,
        max_length=20,
        has_tabular=True,
        read_columns=None,
        shuffle_files=False,
        shuffle_pq=False,
        event_time_column=None,
        event_ids_column="event_ids",
    )
    collate_fn = EventSequenceCollateFn(
        sequence_columns=["mcc", "price"],
        create_attention_mask=True,
        target_column=None,
        is_regression=False,
        has_tabular=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, collate_fn=collate_fn, batch_size=10
    )

    batch = next(iter(dataloader))
    assert batch["tab_features"].cat_features.shape == (10, 10)
    assert batch["tab_features"].num_features.shape == (10, 10)


def test_slice_by_event_ids(synth_sequence_dataset):
    """Tests for min/max_length and sequence dtypes"""
    output_path = synth_sequence_dataset(
        num_records=100,
        num_output_partitions=8,
        num_events_range=(100, 100),
        target_column="classification",
        tabular_features=True,
    )

    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        selected_event_ids=[0, 1],
        min_length=1,
        max_length=100,
        has_tabular=True,
        read_columns=None,
        shuffle_files=False,
        event_time_column=None,
        event_ids_column="event_ids",
        shuffle_pq=False,
    )

    first_pq = pd.read_parquet(dataset.files[0]).to_dict("records")[0]
    first_record = next(iter(dataset))
    first_sec_size = first_record["_event_ids"].shape[0]

    assert torch.all(np.array(first_pq["mcc"]) == first_record["mcc"])
    assert set(first_record["_event_ids"].tolist()) == set([0, 1])
    assert first_pq["mcc"].shape[0] == first_record["_event_ids"].shape[0]
    assert first_pq["price"].shape[0] == first_record["_event_ids"].shape[0]

    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        selected_event_ids=[0],
        min_length=1,
        max_length=100,
        has_tabular=True,
        event_time_column=None,
        event_ids_column="event_ids",
        read_columns=None,
        shuffle_files=False,
        shuffle_pq=False,
    )

    first_pq = pd.read_parquet(dataset.files[0]).to_dict("records")[0]
    first_record = next(iter(dataset))
    first_size = first_record["_event_ids"].shape[0]

    assert set(first_record["_event_ids"].tolist()) == set([0])
    assert first_pq["mcc"].shape[0] > first_record["_event_ids"].shape[0]
    assert first_pq["price"].shape[0] > first_record["_event_ids"].shape[0]
    assert first_size < first_sec_size
    assert (first_pq["mcc"][first_pq["event_ids"] == 0] == first_record["mcc"]).all()

    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        selected_event_ids=[1],
        min_length=1,
        max_length=100,
        has_tabular=True,
        read_columns=None,
        event_time_column=None,
        event_ids_column="event_ids",
        shuffle_files=False,
        shuffle_pq=False,
    )

    first_pq = pd.read_parquet(dataset.files[0]).to_dict("records")[0]
    first_record = next(iter(dataset))
    second_size = first_record["_event_ids"].shape[0]

    assert set(first_record["_event_ids"].tolist()) == set([1])
    assert first_pq["mcc"].shape[0] > first_record["_event_ids"].shape[0]
    assert first_pq["price"].shape[0] > first_record["_event_ids"].shape[0]
    assert second_size < first_sec_size
    assert (first_pq["mcc"][first_pq["event_ids"] == 1] == first_record["mcc"]).all()


@pytest.fixture
def base_record():
    return {
        "mcc": np.array([1, 2, 3, 4, 5]),
        "price": np.array([10, 20, 30, 40, 50]),
        "event_ids": np.array([0, 1, 0, 1, 2]),
        "timestamp": np.array([1000, 2000, 3000, 4000, 5000]),
        "tabular_feature": 42,
    }


def test_no_event_ids_column(base_record, output_path):
    """Test when no event IDs column is configured"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        event_ids_column=None,
        selected_event_ids=None,
        min_length=1,
        max_length=10,
        has_tabular=True,
    )

    result = dataset.process(base_record.copy())
    assert result is not None
    assert "_event_ids" not in result
    assert len(result["mcc"]) == 5
    assert "tabular_feature" in result


def test_event_ids_without_selection(base_record, output_path):
    """Test event IDs column without selected event IDs"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        event_ids_column="event_ids",
        selected_event_ids=None,
        min_length=1,
        max_length=10,
    )

    result = dataset.process(base_record.copy())
    assert result is not None
    assert torch.equal(result["_event_ids"], torch.tensor([0, 1, 0, 1, 2]))
    assert len(result["mcc"]) == 5
    assert "event_ids" not in result


def test_event_ids_with_selection(base_record, output_path):
    """Test filtering with selected event IDs"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        event_ids_column="event_ids",
        selected_event_ids=[0, 1],
        min_length=1,
        max_length=10,
    )

    result = dataset.process(base_record.copy())
    assert result is not None
    assert torch.equal(result["_event_ids"], torch.tensor([0, 1, 0, 1]))
    assert torch.equal(result["mcc"], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(result["price"], torch.tensor([10, 20, 30, 40]))


def test_insufficient_events_after_filtering(base_record, output_path):
    """Test record skipping when filtered events < min_length"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        event_ids_column="event_ids",
        selected_event_ids=[2],  # Only one matching event
        min_length=2,
        max_length=10,
    )

    result = dataset.process(base_record.copy())
    assert result is None


def test_slicing_with_event_filtering(base_record, output_path):
    """Test interaction between event filtering and length-based slicing"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        event_ids_column="event_ids",
        selected_event_ids=[0, 1],
        min_length=1,
        max_length=3,
        random_slicing=False,
    )

    result = dataset.process(base_record.copy())
    assert result is not None
    assert len(result["_event_ids"]) == 3
    assert torch.equal(result["mcc"], torch.tensor([2, 3, 4]))
    assert torch.equal(result["price"], torch.tensor([20, 30, 40]))


def test_timestamp_processing(base_record, output_path):
    """Test timestamp processing with event filtering"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        event_ids_column="event_ids",
        event_time_column="timestamp",
        selected_event_ids=[0, 1],
        min_length=1,
        max_length=10,
    )

    result = dataset.process(base_record.copy())
    assert result is not None
    assert "_timestamps" in result
    assert len(result["_timestamps"]) == 4
    assert torch.equal(
        result["_timestamps"],
        torch.tensor([1000, 2000, 3000, 4000], dtype=torch.float32),
    )


def test_lazy_processing(base_record, output_path):
    """Test lazy processing returns function instead of processed result"""
    dataset = EventSequenceDataset(
        path=output_path,
        sequence_columns=["mcc", "price"],
        lazy_process=True,
        min_length=1,
        max_length=10,
    )

    result = dataset.process(base_record.copy())
    assert callable(result)
    processed = result()
    assert isinstance(processed, dict)
    assert "mcc" in processed
