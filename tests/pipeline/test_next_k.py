import torch

from avatar.nn.embedding import BaseEmbedding
from avatar.nn.feature_encoder import BaseSequenceFeatureEncoder
from avatar.nn.sequence import BaseSequenceBackbone, BaseSequenceModel
from avatar.pipeline.sequence import NextKTokensPrediction

feature_encoder = BaseSequenceFeatureEncoder(embedding=BaseEmbedding(hidden_size=128))
feature_encoder.embedding.columns_meta = {
    "pos_geo_evt_attr_1": {"n_classes": 87, "type": "categorical"},
    "txn_evt_attr_15": {"n_classes": 1, "type": "numeric"},
}

model = NextKTokensPrediction(
    model=BaseSequenceModel(
        feature_encoder=feature_encoder, backbone=BaseSequenceBackbone()
    ),
    horizon=3,
)


def test_create_labels_logits():
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    logits = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
    ])
    horizon_offset = 1
    n_classes = 3

    shifted_logits, shifted_labels = model.create_labels(
        input_ids, logits, horizon_offset, n_classes
    )

    assert torch.equal(
        shifted_logits,
        torch.tensor([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27]],
        ]),
    )
    assert torch.equal(shifted_labels, torch.tensor([[2, 3, 4, 5], [7, 8, 9, 10]]))


def test_horizon():
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    logits = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
    ])
    horizon_offset = 3
    n_classes = 3

    shifted_logits, shifted_labels = model.create_labels(
        input_ids, logits, horizon_offset, n_classes
    )

    assert torch.equal(
        shifted_logits,
        torch.tensor([[[1, 2, 3], [4, 5, 6]], [[16, 17, 18], [19, 20, 21]]]),
    )
    assert torch.equal(shifted_labels, torch.tensor([[4, 5], [9, 10]]))


def test_padding():
    input_ids = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    logits = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
    ])
    horizon_offset = 3
    n_classes = 3

    _, shifted_labels = model.create_labels(
        input_ids, logits, horizon_offset, n_classes
    )

    assert torch.equal(shifted_labels, torch.tensor([[-100, -100], [-100, -100]]))

    input_ids = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    logits = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
    ])
    horizon_offset = 3
    n_classes = 1

    _, shifted_labels = model.create_labels(
        input_ids, logits, horizon_offset, n_classes
    )

    assert torch.equal(shifted_labels, torch.tensor([[0, 0], [0, 0]]))


def test_null_label():
    input_ids = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    logits = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
            [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]],
        ],
        dtype=torch.float32,
    )
    horizon_offset = 3
    n_classes = 3

    shifted_logits, shifted_labels = model.create_labels(
        input_ids, logits, horizon_offset, n_classes
    )

    loss, num_items = model.calculate_loss(
        shifted_logits=shifted_logits,
        shifted_labels=shifted_labels,
        n_classes=n_classes,
    )
    assert loss == 0
    assert num_items == 0


def test_apply_attention_mask():
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    logits = torch.tensor(
        [
            [[1], [2], [3], [4], [5]],
            [[6], [7], [8], [9], [10]],
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    horizon_offset = 1
    n_classes = 1

    shifted_logits, shifted_labels = model.create_labels(
        input_ids, logits, horizon_offset, n_classes
    )
    _, num_items = model.calculate_loss(
        shifted_logits=shifted_logits,
        shifted_labels=shifted_labels,
        n_classes=n_classes,
        attention_mask=attention_mask,
        horizon_offset=horizon_offset,
    )

    assert num_items == 3


def test_numeric_loss():
    test_logits, test_labels = (
        torch.randint(0, 11, (2, 4, 1)),
        torch.randint(2, 5, (2, 4)),
    )
    expected_output = abs(test_labels - test_logits.squeeze(-1)).sum()
    loss, _ = model.calculate_loss(
        shifted_logits=test_logits,
        shifted_labels=test_labels,
        n_classes=1,
        attention_mask=torch.ones_like(test_labels),
        horizon_offset=0,
    )
    assert torch.isclose(loss, expected_output, atol=1e-5)


def test_numeric_loss_with_mask():
    test_logits = torch.FloatTensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    test_labels = torch.FloatTensor([[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5]])
    test_attn_mask = torch.LongTensor([
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ])
    expected_output = abs(test_labels - test_logits.squeeze(-1)).sum() * 0.75
    loss, _ = model.calculate_loss(
        shifted_logits=test_logits,
        shifted_labels=test_labels,
        n_classes=1,
        attention_mask=test_attn_mask,
        horizon_offset=0,
    )
    assert torch.isclose(loss, expected_output, atol=1e-5)
    test_attn_mask = torch.LongTensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])
    expected_output = abs(test_labels - test_logits.squeeze(-1)).sum() * 0.5
    loss, _ = model.calculate_loss(
        shifted_logits=test_logits,
        shifted_labels=test_labels,
        n_classes=1,
        attention_mask=test_attn_mask,
        horizon_offset=0,
    )
    assert torch.isclose(loss, expected_output, atol=1e-5)
    test_attn_mask = torch.LongTensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    expected_output = torch.tensor([0.0])
    loss, _ = model.calculate_loss(
        shifted_logits=test_logits,
        shifted_labels=test_labels,
        n_classes=1,
        attention_mask=test_attn_mask,
        horizon_offset=0,
    )
    assert torch.isclose(loss, expected_output, atol=1e-5)


def test_catigorical_loss():
    shifted_logits = torch.tensor([
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [1.0, 1.0, 1.0]]
    ])
    shifted_labels = torch.tensor([[2, 0, -100, -100]])

    loss, num_items = model.calculate_loss(
        shifted_logits=shifted_logits,
        shifted_labels=shifted_labels,
        n_classes=3,
        horizon_offset=0,
    )

    assert loss > 0
    assert num_items == 2
