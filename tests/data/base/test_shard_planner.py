"""Spec for the record-level sharding math.

The properties asserted here are what keep DDP from deadlocking: every rank
owns the same number of records, no record is owned twice, and nothing but the
global remainder is lost.
"""

import itertools

import numpy as np
import pytest

from fmlib.data.base import ShardPlanner

# Deliberately uneven file sizes — equal-sized files hide every sharding bug.
UNEVEN = np.array([13, 1, 40, 7, 22, 3, 91, 5], dtype=np.int64)


def owned_ranges(planner: ShardPlanner, num_workers: int, epoch: int = 0):
    """Global record indexes this rank owns, as a flat list."""
    cum = np.concatenate([[0], np.cumsum(planner.file_counts)])
    owned = []
    for worker_id in range(num_workers):
        for segment in planner.worker_segments(
            worker_id=worker_id, num_workers=num_workers, epoch=epoch
        ):
            base = int(cum[segment.file_index])
            owned.extend(range(base + segment.start, base + segment.stop))
    return owned


def planners(file_counts, world_size, **kwargs):
    return [
        ShardPlanner(
            file_counts=file_counts, world_size=world_size, rank=rank, **kwargs
        )
        for rank in range(world_size)
    ]


@pytest.mark.parametrize("world_size", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("num_workers", [1, 2, 3, 4])
def test_ranks_are_disjoint_and_equal_sized(world_size, num_workers):
    all_owned = []
    for planner in planners(UNEVEN, world_size):
        owned = owned_ranges(planner, num_workers)
        assert len(owned) == planner.rank_record_count()
        assert len(owned) == UNEVEN.sum() // world_size
        all_owned.append(owned)

    flat = list(itertools.chain.from_iterable(all_owned))
    assert len(flat) == len(set(flat)), "a record is owned by more than one shard"
    assert len(flat) == UNEVEN.sum() - UNEVEN.sum() % world_size


@pytest.mark.parametrize("world_size", [2, 3, 5])
def test_drop_tail_disabled_keeps_every_record(world_size):
    all_owned = []
    for planner in planners(UNEVEN, world_size, drop_tail=False):
        all_owned.extend(owned_ranges(planner, num_workers=1))
    assert sorted(all_owned) == list(range(int(UNEVEN.sum())))


@pytest.mark.parametrize("num_workers", [1, 2, 3])
def test_workers_partition_their_rank(num_workers):
    planner = ShardPlanner(file_counts=UNEVEN, world_size=3, rank=1)
    seen = []
    for worker_id in range(num_workers):
        segments = planner.worker_segments(worker_id=worker_id, num_workers=num_workers)
        count = sum(len(segment) for segment in segments)
        assert count == planner.worker_record_count(worker_id, num_workers)
        cum = np.concatenate([[0], np.cumsum(planner.file_counts)])
        for segment in segments:
            base = int(cum[segment.file_index])
            seen.extend(range(base + segment.start, base + segment.stop))

    assert len(seen) == len(set(seen))
    assert len(seen) == planner.rank_record_count()


def test_rotate_tail_moves_the_dropped_records():
    counts = np.array([10, 10, 3], dtype=np.int64)  # 23 records, 4 ranks -> tail 3
    world_size = 4

    def owned_for_epoch(epoch):
        owned = []
        for planner in planners(counts, world_size, rotate_tail=True):
            owned.extend(owned_ranges(planner, num_workers=1, epoch=epoch))
        return set(owned)

    first, second = owned_for_epoch(0), owned_for_epoch(1)
    assert len(first) == len(second) == 20
    assert first != second, "rotation must change which records are dropped"


def test_no_rotation_without_the_flag():
    counts = np.array([10, 10, 3], dtype=np.int64)
    owned = [
        {
            index
            for planner in planners(counts, 4)
            for index in owned_ranges(planner, num_workers=1, epoch=epoch)
        }
        for epoch in (0, 1, 2)
    ]
    assert owned[0] == owned[1] == owned[2]


def test_empty_corpus_yields_nothing():
    planner = ShardPlanner(
        file_counts=np.zeros(4, dtype=np.int64), world_size=2, rank=1
    )
    assert planner.total == 0
    assert planner.rank_record_count() == 0
    assert planner.worker_segments(worker_id=0, num_workers=2) == []


def test_more_workers_than_records():
    planner = ShardPlanner(
        file_counts=np.array([3], dtype=np.int64), world_size=1, rank=0
    )
    counts = [
        len(planner.worker_segments(worker_id=worker_id, num_workers=8))
        for worker_id in range(8)
    ]
    assert sum(counts) == 3, "idle workers are fine; lost records are not"


def test_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        ShardPlanner(file_counts=UNEVEN, world_size=0, rank=0)
    with pytest.raises(ValueError):
        ShardPlanner(file_counts=UNEVEN, world_size=2, rank=2)
    with pytest.raises(ValueError):
        ShardPlanner(file_counts=np.array([-1]), world_size=1, rank=0)

    planner = ShardPlanner(file_counts=UNEVEN, world_size=1, rank=0)
    with pytest.raises(ValueError):
        planner.worker_segments(worker_id=2, num_workers=2)
    with pytest.raises(ValueError):
        planner.worker_segments(worker_id=0, num_workers=0)
