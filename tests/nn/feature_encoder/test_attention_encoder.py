import torch

from avatar.nn.feature_encoder.attention_encoder import IntraFeatureAttention


def test_compute_scores():
    ifa_example = IntraFeatureAttention(hidden_size=5)
    test_input = torch.randn(3, 3, 3, 3)
    test_attn_mask = torch.randint(0, 2, (3, 3, 3, 3))
    test_scores = ifa_example.calcuate_attn_scores(
        q=test_input, k=test_input, event_attention_mask=test_attn_mask
    )
    test_scores = torch.where(torch.nan_to_num(test_scores, nan=0.0) == 0, 0, 1)
    assert torch.equal(test_attn_mask, test_scores)
