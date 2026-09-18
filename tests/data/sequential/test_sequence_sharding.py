"""Sharding behaviour of :class:`EventSequenceDataset`.

Ranks are simulated through the ``WORLD_SIZE`` / ``RANK`` variables ``torchrun``
exports, so these exercise the real rank-resolution path rather than a mocked
``torch.distributed``. The multi-process counterpart, which runs actual
collectives, lives in ``tests/data/test_distributed_sharding.py``.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import torch

from fmlib.data.base import ShardPlanner
from fmlib.data.sequential import EventSequenceCollateFn, EventSequenceDataset

SEQUENCE_COLUMNS = ["mcc", "price"]


@pytest.fixture
def data_sample(synth_sequence_dataset):
    """Function-scoped synthetic-data factory (pandas/pyarrow, no Spark)."""

    @lru_cache(maxsize=1_000)
    def _generate_data(
        num_records=128, num_output_partitions=8, num_events_range=(1, 100)
    ):
        return synth_sequence_dataset(
            num_records=num_records,
            num_output_partitions=num_output_partitions,
            num_events_range=num_events_range,
        )

    return _generate_data


def build(path, world_size=1, rank=0, monkeypatch=None, **kwargs):
    """Construct a dataset as if launched by torchrun with the given geometry."""
    if monkeypatch is not None:
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("RANK", str(rank))
    options = {
        "sequence_columns": SEQUENCE_COLUMNS,
        "event_time_column": None,
        "event_ids_column": "event_ids",
        "min_length": 1,
        "max_length": 100,
        "random_slicing": False,
        "has_tabular": False,
        "shuffle_files": False,
        "shuffle_pq": False,
    }
    options.update(kwargs)
    return EventSequenceDataset(path=path, **options)


def ids_from_loader(dataset, num_workers=0, batch_size=1):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=EventSequenceCollateFn(sequence_columns=SEQUENCE_COLUMNS),
        drop_last=False,
    )
    batches = list(loader)
    ids = [int(value) for batch in batches for value in batch["epk_id"]]
    return ids, len(batches)


# --------------------------------------------------------------------------- #
# Single process                                                               #
# --------------------------------------------------------------------------- #
def test_dataset_length(data_sample):
    dataset = build(data_sample(num_records=128, num_output_partitions=8))
    assert len([1 for _ in dataset]) == 128
    assert len(dataset) == 128


def test_filter_by_length(data_sample):
    output_dir = data_sample(num_records=2000, num_output_partitions=8)
    min_length = 30
    lengths = pd.read_parquet(output_dir)["mcc"].apply(len)
    expected = 2000 - int((lengths < min_length).sum())

    dataset = build(output_dir, min_length=min_length)
    assert len([1 for _ in dataset]) == expected
    assert len(dataset) == expected


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_dataloader_yields_every_record(num_workers, data_sample):
    dataset = build(data_sample(num_records=128, num_output_partitions=8))
    ids, _ = ids_from_loader(dataset, num_workers=num_workers)
    assert sorted(ids) == list(range(1, 129))


# --------------------------------------------------------------------------- #
# Multi rank                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 4, 5])
def test_ranks_cover_everything_without_duplicates(
    num_workers, data_sample, monkeypatch
):
    world_size, total = 2, 128
    path = data_sample(num_records=total, num_output_partitions=8)

    all_ids = []
    for rank in range(world_size):
        dataset = build(path, world_size, rank, monkeypatch)
        ids, _ = ids_from_loader(dataset, num_workers=num_workers)
        assert len(ids) == total // world_size
        all_ids.extend(ids)

    assert len(all_ids) == total
    assert len(set(all_ids)) == total


@pytest.mark.slow
def test_remainder_costs_records_not_whole_files(data_sample, monkeypatch):
    """A file count that does not divide by world_size must not lose a file.

    The previous file-level split truncated the file list to a multiple of
    ``world_size``, silently dropping every record of the remainder files. Only
    the global record remainder may be lost.
    """
    world_size, total = 2, 32 * 9
    path = data_sample(num_records=total, num_output_partitions=9)

    all_ids = []
    for rank in range(world_size):
        dataset = build(path, world_size, rank, monkeypatch)
        ids, _ = ids_from_loader(dataset)
        all_ids.extend(ids)

    assert len(all_ids) == total - total % world_size == total
    assert len(set(all_ids)) == len(all_ids)


@pytest.mark.slow
@pytest.mark.parametrize(
    "num_workers,world_size,batch_size",
    [(j, i, k) for i in range(1, 4) for j in range(3) for k in [1, 2, 3, 4]],
)
def test_every_rank_gets_the_same_number_of_batches(
    num_workers, world_size, batch_size, data_sample, monkeypatch
):
    """Unequal batch counts are what deadlock DDP at the first all-reduce."""
    total = 2345
    path = data_sample(num_records=total, num_output_partitions=27)

    batch_counts, all_ids = [], []
    for rank in range(world_size):
        dataset = build(path, world_size, rank, monkeypatch)
        ids, num_batches = ids_from_loader(
            dataset, num_workers=num_workers, batch_size=batch_size
        )
        batch_counts.append(num_batches)
        all_ids.extend(ids)

    assert len(set(batch_counts)) == 1, batch_counts
    assert len(set(all_ids)) == len(all_ids)
    assert len(all_ids) == total - total % world_size


@pytest.mark.slow
def test_uneven_files_still_balance(synth_sequence_dataset, monkeypatch):
    """Files with very different valid-record counts must still split evenly."""
    path = synth_sequence_dataset(
        num_records=900, num_output_partitions=7, num_events_range=(0, 120)
    )
    min_length, world_size = 40, 3

    counts, all_ids = [], []
    for rank in range(world_size):
        dataset = build(path, world_size, rank, monkeypatch, min_length=min_length)
        ids, _ = ids_from_loader(dataset)
        counts.append(len(ids))
        all_ids.extend(ids)

    assert len(set(counts)) == 1, counts
    assert len(set(all_ids)) == len(all_ids)


# --------------------------------------------------------------------------- #
# Event-type filtering (previously unsupported when sharding)                  #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_selected_event_ids_shard_correctly(data_sample, monkeypatch):
    path = data_sample(num_records=256, num_output_partitions=8)
    min_length, world_size = 5, 2

    frame = pd.read_parquet(path)
    expected = int(
        sum(
            (len(row["mcc"]) >= min_length)
            and (np.isin(np.asarray(row["event_ids"]), [1]).sum() >= min_length)
            for _, row in frame.iterrows()
        )
    )

    all_ids = []
    for rank in range(world_size):
        dataset = build(
            path,
            world_size,
            rank,
            monkeypatch,
            selected_event_ids=[1],
            min_length=min_length,
        )
        ids, _ = ids_from_loader(dataset)
        all_ids.extend(ids)

    assert len(all_ids) == expected - expected % world_size
    assert len(set(all_ids)) == len(all_ids)


def test_selected_event_ids_keeps_only_those_events(data_sample):
    dataset = build(
        data_sample(num_records=64, num_output_partitions=4),
        selected_event_ids=[1],
        min_length=1,
    )
    for record in dataset:
        assert set(record["_event_ids"].tolist()) <= {1}
        assert len(record["mcc"]) == len(record["_event_ids"])


# --------------------------------------------------------------------------- #
# Epoch handling and the divergence guard                                      #
# --------------------------------------------------------------------------- #
def test_set_epoch_rotates_which_records_are_dropped(data_sample, monkeypatch):
    path = data_sample(num_records=101, num_output_partitions=4)
    world_size = 4

    def ids_for_epoch(epoch):
        collected = []
        for rank in range(world_size):
            dataset = build(path, world_size, rank, monkeypatch, rotate_tail=True)
            dataset.set_epoch(epoch)
            collected.extend(int(record["epk_id"]) for record in dataset)
        return set(collected)

    first, second = ids_for_epoch(0), ids_for_epoch(1)
    assert len(first) == len(second) == 100
    assert first != second


def test_divergence_between_scan_and_iteration_is_fatal(data_sample):
    """A predicate mismatch must fail loudly instead of hanging DDP."""
    dataset = build(data_sample(num_records=32, num_output_partitions=2))
    # Pretend the scan counted more records than the files can produce.
    dataset._file_counts = dataset._file_counts + 5
    dataset._planner = ShardPlanner(
        file_counts=dataset._file_counts, world_size=1, rank=0
    )
    with pytest.raises(RuntimeError, match="diverged from count_valid_records"):
        list(dataset)
