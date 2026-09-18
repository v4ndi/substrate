"""The decisions :func:`predict` makes before it runs anything.

The multi-rank consequences are covered by ``test_distributed_predict.py``;
these are the single-process rules that pick which of the three modes a run
gets, and the file naming that makes the cheapest one safe.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from fmlib.metrics.base import ArtifactMetric, ScalarMetric
from fmlib.train.dist import DistEnv
from fmlib.train.evaluate import metrics_need_population, warn_about_dropped_tail


class Numbers(ScalarMetric):
    def update(self, inputs, outputs):
        pass

    def compute(self):
        return {}

    def reset(self):
        pass


class Rows(ArtifactMetric):
    def flush(self):
        pass

    def update(self, inputs, outputs):
        pass

    def reset(self):
        pass


def test_a_collector_does_not_need_the_population():
    assert metrics_need_population([Rows(path_to_save="/tmp/does-not-matter")]) is False


def test_a_scalar_metric_does():
    assert metrics_need_population([Numbers()]) is True


def test_one_metric_that_needs_it_is_enough():
    """The union is the safe side: a mixed list still gathers."""
    metrics = [Rows(path_to_save="/tmp/does-not-matter"), Numbers()]
    assert metrics_need_population(metrics) is True


def test_no_metrics_need_nothing():
    assert metrics_need_population(None) is False
    assert metrics_need_population([]) is False


def test_a_metric_that_says_nothing_is_assumed_to_need_it():
    """An out-of-tree metric predating the flag must not be scored on a slice."""
    legacy = types.SimpleNamespace(update=None, compute=None, reset=None)
    assert metrics_need_population([legacy]) is True


def test_parts_carry_the_rank_when_there_is_more_than_one(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    metric = Rows(path_to_save=str(tmp_path))
    assert "rank-002" in metric.next_path()


def test_parts_are_named_as_before_on_a_single_process(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    metric = Rows(path_to_save=str(tmp_path))
    assert "rank-" not in metric.next_path()


def test_two_ranks_no_longer_choose_the_same_name(tmp_path, monkeypatch):
    """What the gather used to hide: identical stamps and identical counters."""
    names = []
    for rank in (0, 1):
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("RANK", str(rank))
        metric = Rows(path_to_save=str(tmp_path))
        metric._run_stamp = "2026_01_01_00_00_00"  # same second, as in a real launch
        names.append(metric.next_path())
    assert names[0] != names[1]


def test_a_collector_writes_what_it_is_given(tmp_path):
    """save_dataframe is the only place a part file is created."""
    metric = Rows(path_to_save=str(tmp_path))
    path = metric.save_dataframe(pd.DataFrame({"id": [1, 2, 3]}))
    assert pd.read_parquet(path)["id"].tolist() == [1, 2, 3]


class FakeDataset:
    shard = True

    def __init__(self, drop_tail, tail, total):
        self.drop_tail = drop_tail
        self._planner = types.SimpleNamespace(tail=tail, total=total)


def loader_over(dataset):
    return types.SimpleNamespace(dataset=dataset)


def test_the_dropped_tail_is_reported_with_its_size():
    env = DistEnv(rank=0, world_size=2)
    with pytest.warns(UserWarning, match="1 of 101 records will not be scored"):
        warn_about_dropped_tail(
            loader_over(FakeDataset(drop_tail=True, tail=1, total=101)), env
        )


def test_nothing_is_said_when_nothing_is_lost(recwarn):
    env = DistEnv(rank=0, world_size=2)
    warn_about_dropped_tail(
        loader_over(FakeDataset(drop_tail=False, tail=1, total=101)), env
    )
    warn_about_dropped_tail(
        loader_over(FakeDataset(drop_tail=True, tail=0, total=100)), env
    )
    warn_about_dropped_tail(
        loader_over(FakeDataset(drop_tail=True, tail=1, total=101)),
        DistEnv(rank=0, world_size=1),
    )
    assert [str(warning.message) for warning in recwarn] == []
