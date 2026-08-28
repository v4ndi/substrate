from functools import lru_cache
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from pyspark.sql import SparkSession

from avatar.data.dataset import ShardEventSequenceDataset
from avatar.data.dataset.collate_fn import EventSequenceCollateFn
from avatar.synth import generate_sequence_dataset


@pytest.fixture(scope="session")
def spark_session():
    """Session-scoped Spark fixture"""
    spark = SparkSession.builder.appName("SequencePreprocessorTests").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def data_sample(spark_session):
    """Function-scoped data sample fixture"""

    @lru_cache(maxsize=1_000)
    def _generate_data(num_records=128, num_output_partitions=8):
        return generate_sequence_dataset(
            spark=spark_session,
            num_records=num_records,
            num_output_partitions=num_output_partitions,
            num_events_range=(1, 100),
        )

    return _generate_data


def test_dataset_length(data_sample):
    """Test that the dataset returns the correct number of records."""
    dataset = ShardEventSequenceDataset(
        path=data_sample(num_records=128, num_output_partitions=8),
        sequence_columns=["mcc", "price"],
        event_time_column=None,
        event_ids_column="event_ids",
        selected_event_ids=None,
        min_length=1,
        max_length=100,
        random_slicing=False,
        has_tabular=False,
        shuffle_files=False,
        shuffle_pq=False,
    )
    length = len([1 for _ in dataset])
    assert length == 128
    assert length == len(dataset)


def test_filter_by_length(data_sample):
    """Test that the dataset correctly filters sequences by minimum length."""
    output_dir = data_sample(num_records=2000, num_output_partitions=8)
    min_length = 30
    lengths = pd.read_parquet(output_dir)["mcc"].apply(len)
    result_num_records = 2000 - (lengths < min_length).sum()

    dataset = ShardEventSequenceDataset(
        path=output_dir,
        sequence_columns=["mcc", "price"],
        event_time_column=None,
        event_ids_column="event_ids",
        selected_event_ids=None,
        min_length=min_length,
        max_length=100,
        random_slicing=False,
        has_tabular=False,
        shuffle_files=False,
        shuffle_pq=False,
    )
    length = len([1 for _ in dataset])
    assert length == result_num_records


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_dataloader_length(num_workers, data_sample):
    """Test that the dataloader returns the correct number of batches with varying num_workers."""
    output_dir = data_sample(num_records=128, num_output_partitions=8)
    dataset = ShardEventSequenceDataset(
        path=output_dir,
        sequence_columns=["mcc", "price"],
        event_time_column=None,
        event_ids_column="event_ids",
        selected_event_ids=None,
        min_length=1,
        max_length=100,
        random_slicing=False,
        has_tabular=False,
        shuffle_files=False,
        shuffle_pq=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=EventSequenceCollateFn(sequence_columns=["mcc", "price"]),
        drop_last=False,
    )

    length = len([1 for _ in dataloader])
    assert length == 128


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_multi_gpu_sharding_with_mocks(num_workers, data_sample):
    """Test dataset sharding in multi-GPU environment using mocks.

    Verifies that:
    1. The total number of samples is correct across all ranks
    2. There are no duplicate samples across ranks
    """
    world_size = 2
    total_samples = 128
    all_samples = []
    path = data_sample(num_records=total_samples, num_output_partitions=8)
    dataset = ShardEventSequenceDataset(
        path=path,
        sequence_columns=["mcc", "price"],
        event_time_column=None,
        event_ids_column="event_ids",
        selected_event_ids=None,
        min_length=1,
        max_length=100,
        random_slicing=False,
        has_tabular=False,
        shuffle_files=False,
        shuffle_pq=False,
    )
    for rank in range(world_size):
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=rank),
        ):
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=num_workers,
                collate_fn=EventSequenceCollateFn(sequence_columns=["mcc", "price"]),
                drop_last=False,
            )

            rank_samples = [x for x in dataloader]
            all_samples.extend(rank_samples)

    assert len(all_samples) == total_samples
    sample_ids = [x["epk_id"][0].item() for x in all_samples]
    assert len(set(sample_ids)) == total_samples


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_multi_gpu_sharding_remainder_files(num_workers, data_sample):
    """Test dataset sharding when number of files isn't divisible by world_size.

    Verifies correct behavior when there are remainder files after division.
    """
    world_size = 2
    total_samples = 32 * 9
    all_samples = []
    path = data_sample(num_records=total_samples, num_output_partitions=9)
    dataset = ShardEventSequenceDataset(
        path=path,
        sequence_columns=["mcc", "price"],
        event_time_column=None,
        event_ids_column="event_ids",
        selected_event_ids=None,
        min_length=1,
        max_length=100,
        random_slicing=False,
        has_tabular=False,
        shuffle_files=False,
        shuffle_pq=False,
    )
    for rank in range(world_size):
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=rank),
        ):
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=num_workers,
                collate_fn=EventSequenceCollateFn(sequence_columns=["mcc", "price"]),
                drop_last=False,
            )

            rank_samples = [x for x in dataloader]
            all_samples.extend(rank_samples)

    assert len(all_samples) == (total_samples - 32)
    sample_ids = [x["epk_id"][0].item() for x in all_samples]
    assert len(set(sample_ids)) == (total_samples - 32)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_workers,world_size,batch_size",
    [(j, i, k) for i in range(1, 4) for j in range(3) for k in [1, 2, 3, 4]],
)
def test_multi_gpu_batches_per_rank(num_workers, world_size, batch_size, data_sample):
    """Test that each rank gets the same number of batches in multi-GPU setup.

    Verifies balanced batch distribution across ranks for various configurations.
    """
    total_samples = 2345
    all_samples = []
    path = data_sample(num_records=total_samples, num_output_partitions=27)

    dataset = ShardEventSequenceDataset(
        path=path,
        sequence_columns=["mcc", "price"],
        event_time_column=None,
        event_ids_column="event_ids",
        selected_event_ids=None,
        min_length=1,
        max_length=100,
        random_slicing=False,
        has_tabular=False,
        shuffle_files=False,
        shuffle_pq=False,
    )
    for rank in range(world_size):
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=rank),
        ):
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=EventSequenceCollateFn(sequence_columns=["mcc", "price"]),
                drop_last=False,
            )

            rank_samples = [x for x in dataloader]
            all_samples.append(rank_samples)

    rank_samples_length = [len(x) for x in all_samples]
    assert all(x == rank_samples_length[0] for x in rank_samples_length)

    unioned = []
    for x in all_samples:
        unioned.extend(x)
    sample_ids = [x["epk_id"][0].item() for x in unioned]
    assert len(set(sample_ids)) == len(unioned)
