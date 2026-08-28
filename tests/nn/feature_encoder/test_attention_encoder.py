import torch

from avatar.data import EventSequenceBatch
from avatar.nn.embedding import EventSequenceEmbedding
from avatar.nn.feature_encoder.attention_encoder import (
    FeatureAttentionEncoder,
    IntraFeatureAttention,
)


def test_default():
    hidden_size = 4
    emb_example = EventSequenceEmbedding(
        hidden_size=hidden_size,
        columns_meta={
            "attr_1": {
                "type": "categorical",
                "n_classes": 2,
                "event_id": 0,
            },
            "attr_2": {"type": "categorical", "n_classes": 2, "event_id": 1},
            "attr_3": {"type": "categorical", "n_classes": 2, "event_id": 2},
        },
    )
    fae_example = FeatureAttentionEncoder(embedding=emb_example)
    event_ids_tensor = torch.LongTensor([[0, 0, 1, 1, 2, 2]])
    events_tensor = {
        "attr_1": torch.ones_like(event_ids_tensor),
        "attr_2": torch.ones_like(event_ids_tensor),
        "attr_3": torch.ones_like(event_ids_tensor),
    }
    attention_mask_tensor = torch.ones(
        event_ids_tensor.size(0), event_ids_tensor.size(1)
    )
    input_batch = EventSequenceBatch(
        events=events_tensor,
        attention_mask=attention_mask_tensor,
        event_ids=event_ids_tensor,
    )
    test_event_attn = fae_example.event_attention_mask(input_batch)
    expected_tesnor = torch.LongTensor([
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    ])
    assert torch.equal(expected_tesnor, test_event_attn)
    ifa_example = IntraFeatureAttention(hidden_size=emb_example.hidden_size)
    test_scores = torch.where(
        ifa_example.calcuate_attn_scores(
            q=test_event_attn, k=test_event_attn, event_attention_mask=test_event_attn
        )
        != 0,
        1,
        0,
    )
    assert torch.equal(test_scores, test_event_attn)


def test_mix_modalities():
    hidden_size = 4
    emb_example = EventSequenceEmbedding(
        hidden_size=hidden_size,
        columns_meta={
            "attr_1": {
                "type": "categorical",
                "n_classes": 2,
                "event_id": 0,
            },
            "attr_2": {"type": "categorical", "n_classes": 2, "event_id": [1, 2]},
            "attr_3": {"type": "categorical", "n_classes": 2, "event_id": 2},
        },
    )
    fae_example = FeatureAttentionEncoder(embedding=emb_example)
    event_ids_tensor = torch.LongTensor([[0, 2, 1, 2, 0]])
    events_tensor = {
        "attr_1": torch.ones_like(event_ids_tensor),
        "attr_2": torch.ones_like(event_ids_tensor),
        "attr_3": torch.ones_like(event_ids_tensor),
    }
    attention_mask_tensor = torch.ones(
        event_ids_tensor.size(0), event_ids_tensor.size(1)
    )
    input_batch = EventSequenceBatch(
        events=events_tensor,
        attention_mask=attention_mask_tensor,
        event_ids=event_ids_tensor,
    )
    test_event_attn = fae_example.event_attention_mask(input_batch)
    expected_tesnor = torch.LongTensor([
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    ])
    assert torch.equal(expected_tesnor, test_event_attn)


def test_compute_scores():
    ifa_example = IntraFeatureAttention(hidden_size=5)
    test_input = torch.randn(3, 3, 3, 3)
    test_attn_mask = torch.randint(0, 2, (3, 3, 3, 3))
    test_scores = ifa_example.calcuate_attn_scores(
        q=test_input, k=test_input, event_attention_mask=test_attn_mask
    )
    test_scores = torch.where(torch.nan_to_num(test_scores, nan=0.0) == 0, 0, 1)
    assert torch.equal(test_attn_mask, test_scores)


def test_time_attention():
    hidden_size = 4
    emb_example = EventSequenceEmbedding(
        hidden_size=hidden_size,
        columns_meta={
            "attr_1": {
                "type": "categorical",
                "n_classes": 2,
                "event_id": 0,
            },
            "attr_2": {"type": "categorical", "n_classes": 2, "event_id": [1, 2]},
            "attr_3": {"type": "categorical", "n_classes": 2, "event_id": 2},
        },
    )
    fae_example = FeatureAttentionEncoder(embedding=emb_example, time_encoding="delta")
    event_ids_tensor = torch.LongTensor([[0, 2, 1, 2, 0]])
    events_tensor = {
        "attr_1": torch.ones_like(event_ids_tensor),
        "attr_2": torch.ones_like(event_ids_tensor),
        "attr_3": torch.ones_like(event_ids_tensor),
    }
    attention_mask_tensor = torch.ones(
        event_ids_tensor.size(0), event_ids_tensor.size(1)
    )
    input_batch = EventSequenceBatch(
        events=events_tensor,
        attention_mask=attention_mask_tensor,
        event_ids=event_ids_tensor,
    )
    test_event_attn = fae_example.event_attention_mask(input_batch)
    print(test_event_attn)
    print(test_event_attn.shape)
