import numpy as np
import pytest
import torch

from avatar.metrics import SequenceStats


class DummySeqFeatures:
    def __init__(self, event_ids, seq_len, event_stats):
        """
        event_ids: torch.Tensor shape [num_events, batch_size]
        seq_len: torch.Tensor shape [batch_size]
        event_stats: dict[event_id] = torch.Tensor (num_items for each sample)
        """
        self.event_ids = event_ids
        self.seq_len = seq_len
        self._event_stats = event_stats

    def num_items(self, event_id):
        # Возвращает сумму по батчу для конкретного event_id
        return self._event_stats[event_id]


@pytest.fixture
def sample_inputs_outputs():
    event_ids = torch.tensor(
        [[0, 0, 1, -1, -1], [1, 0, 1, 1, -1]], dtype=torch.int64
    )  # shape [2, 5]
    seq_len = torch.tensor([3, 4], dtype=torch.float32)
    event_stats = {
        0: torch.tensor([2, 1], dtype=torch.float32),
        1: torch.tensor([1, 3], dtype=torch.float32),
    }
    seq_features = DummySeqFeatures(event_ids, seq_len, event_stats)
    inputs = {"seq_features": seq_features}
    outputs = None
    return inputs, outputs


@pytest.fixture
def sample_inputs_outputs_p2():
    event_ids = torch.tensor(
        [[0, 0, 1, 1, 1], [1, 0, 0, 1, -1]], dtype=torch.int64
    )  # shape [2, 5]
    seq_len = torch.tensor([5, 4], dtype=torch.float32)
    event_stats = {
        0: torch.tensor([2, 2], dtype=torch.float32),
        1: torch.tensor([3, 2], dtype=torch.float32),
    }
    seq_features = DummySeqFeatures(event_ids, seq_len, event_stats)
    inputs = {"seq_features": seq_features}
    outputs = None
    return inputs, outputs


@pytest.fixture
def sample_inputs_outputs_p3():
    event_ids = torch.tensor(
        [[0, -1, -1, -1, -1], [1, -1, -1, -1, -1]], dtype=torch.int64
    )  # shape [2, 5]
    seq_len = torch.tensor([1, 1], dtype=torch.float32)
    event_stats = {
        0: torch.tensor([1, 0], dtype=torch.float32),
        1: torch.tensor([0, 1], dtype=torch.float32),
    }
    seq_features = DummySeqFeatures(event_ids, seq_len, event_stats)
    inputs = {"seq_features": seq_features}
    outputs = None
    return inputs, outputs


def test_update_and_compute(sample_inputs_outputs):
    stats = SequenceStats()
    inputs, outputs = sample_inputs_outputs
    stats.update(inputs, outputs)

    assert stats._batch_size == 2
    assert set(stats._events_lengths.keys()) == {0, 1}
    assert stats._context_lengths[0] == 7.0  # 3 + 4

    assert len(stats._events_lengths[0]) == 1
    assert len(stats._events_lengths[1]) == 1

    result = stats.compute()

    assert np.isclose(result["mean_context_length"], 3.5)
    assert np.isclose(result["event_0"], 3 / 7)
    assert np.isclose(result["event_1"], 4 / 7)


def test_reset(sample_inputs_outputs):
    stats = SequenceStats()
    inputs, outputs = sample_inputs_outputs
    stats.update(inputs, outputs)
    stats.compute()  # для заполнения состояния
    stats.reset()
    assert stats._context_lengths == []
    assert stats._events_lengths[0] == []
    assert stats._events_lengths[1] == []
    assert stats._batch_size is None


def test_compute_without_update():
    stats = SequenceStats()
    assert stats.compute() == {}


def test_update_with_negative_event_ids():
    stats = SequenceStats()
    # Добавим отрицательный id
    event_ids = torch.tensor([[-1, 0], [1, 1]], dtype=torch.int64)
    seq_len = torch.tensor([2, 2], dtype=torch.float32)
    event_stats = {
        0: torch.tensor([1, 0], dtype=torch.float32),
        1: torch.tensor([0, 2], dtype=torch.float32),
    }
    seq_features = DummySeqFeatures(event_ids, seq_len, event_stats)
    stats.update({"seq_features": seq_features}, None)
    # -1 не должен попасть в _event_ids
    assert set(stats._events_lengths.keys()) == {0, 1}


def test_few_updates(
    sample_inputs_outputs, sample_inputs_outputs_p2, sample_inputs_outputs_p3
):
    stats = SequenceStats()
    inputs, outputs = sample_inputs_outputs
    stats.update(inputs, outputs)
    inputs, outputs = sample_inputs_outputs_p2
    stats.update(inputs, outputs)
    inputs, outputs = sample_inputs_outputs_p3
    stats.update(inputs, outputs)

    result = stats.compute()
    assert np.isclose(result["mean_context_length"], 3.0)
    assert np.isclose(result["event_0"], 8 / 18)
    assert np.isclose(result["event_1"], 10 / 18)
