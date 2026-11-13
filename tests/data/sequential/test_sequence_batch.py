import torch

from avatar.data import EventSequenceBatch

test_events = {
    "evt_attr_4": torch.tensor([[1, 1, 1, 1, 1]]),
    "evt_attr_5": torch.tensor([[1, 0, 2, 0, 3]]),
    "evt_attr_15": torch.tensor([[0.0, 1.0, 2.0, 0.0, 4.0]]),
}
test_seq_batch = EventSequenceBatch(
    events=test_events,
    timestamps=torch.tensor([[1, 3, 6, 10, 15]]),
    attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
    event_ids=torch.tensor([[0, 1, 1, 2, 1]]),
)


def test_timedeltas():
    assert torch.equal(test_seq_batch.get_timedeltas(), torch.tensor([[0, 2, 3, 4, 0]]))
    assert torch.equal(
        test_seq_batch.get_timedeltas(step=2), torch.tensor([[0, 0, 5, 7, 0]])
    )


def test_event_attn_mask():
    assert torch.equal(
        test_seq_batch.event_attn_mask(event_id=0), torch.tensor([[1, 0, 0, 0, 0]])
    )
    assert torch.equal(
        test_seq_batch.event_attn_mask(event_id=1), torch.tensor([[0, 1, 1, 0, 1]])
    )
    assert torch.equal(
        test_seq_batch.event_attn_mask(event_id=2), torch.tensor([[0, 0, 0, 1, 0]])
    )
    assert torch.equal(
        test_seq_batch.event_attn_mask(event_id=3), torch.tensor([[0, 0, 0, 0, 0]])
    )


def test_num_items():
    assert test_seq_batch.num_items(event_id=0) == 1
    assert test_seq_batch.num_items(event_id=1) == 3
    assert test_seq_batch.num_items(event_id=2) == 1
    assert test_seq_batch.num_items(event_id=3) == 0


edge_events = {
    "evt_attr_4": torch.tensor([[1, 1, 1, 1, 1]]),
    "evt_attr_5": torch.tensor([[1, 0, 2, 0, 3]]),
    "evt_attr_15": torch.tensor([[0.0, 1.0, 2.0, 0.0, 4.0]]),
}
edge_seq_batch = EventSequenceBatch(
    events=test_events,
    timestamps=torch.tensor([[5, 4, 3, 2, 1]]),
    attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
    event_ids=torch.tensor([[1, 1, 1, -1, -1]]),
)


def edge_cases():
    assert torch.equal(
        edge_seq_batch.event_attn_mask(event_id=0), torch.tensor([[0, 0, 0, 0, 0]])
    )
    assert test_seq_batch.num_items(event_id=0) == 0
    assert torch.equal(test_seq_batch.get_timedeltas(), torch.tensor([[0, 0, 0, 0, 0]]))
