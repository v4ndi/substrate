"""TabularTransformer contract: shape, output type, masking, hidden states."""

import torch

from avatar.nn.tabular import TabularTransformer, build_feature_padding_mask
from avatar.outputs import BaseTabularOutput

B, F, D = 4, 9, 16


def _encoder():
    torch.manual_seed(0)
    return TabularTransformer(
        hidden_size=D, num_heads=2, num_layers=3, attn_dropout=0.0
    )


def test_returns_base_tabular_output_with_same_shape():
    enc = _encoder().eval()
    out = enc(torch.randn(B, F, D))

    assert isinstance(out, BaseTabularOutput)
    assert out.last_hidden_state.shape == (B, F, D)
    assert out.hidden_states is None


def test_output_hidden_states_returns_one_per_layer():
    enc = _encoder().eval()
    out = enc(torch.randn(B, F, D), output_hidden_states=True)

    assert len(out.hidden_states) == 3
    assert all(h.shape == (B, F, D) for h in out.hidden_states)
    assert torch.equal(out.hidden_states[-1], out.last_hidden_state)


def test_attention_mask_changes_output():
    enc = _encoder().eval()
    x = torch.randn(B, F, D)
    mask = torch.ones(B, F, dtype=torch.long)
    mask[:, -3:] = 0

    unmasked = enc(x).last_hidden_state
    masked = enc(x, attention_mask=mask).last_hidden_state

    # kept positions must differ once padded positions are excluded from attention
    assert not torch.allclose(unmasked[:, :-3], masked[:, :-3], atol=1e-5)


def test_build_feature_padding_mask():
    assert build_feature_padding_mask(None) is None
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    kpm = build_feature_padding_mask(mask)
    assert kpm.dtype == torch.bool
    assert kpm.tolist() == [[False, False, True]]


def test_hidden_size_attribute():
    assert _encoder().hidden_size == D
