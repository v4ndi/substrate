import torch

from fmlib.nn.utils import (
    BaseAggregation,
    LastHiddenState,
    MeanHiddenState,
)
from fmlib.outputs import BaseSequenceOutput


class TestSample:
    """Shared fixtures for the aggregation tests, as plain class attributes.

    Not a dataclass: none of these are annotated, so ``@dataclass`` generated
    no fields and was a no-op.
    """

    __test__ = False
    states = BaseSequenceOutput(
        last_hidden_state=torch.LongTensor([
            [[1, 2, 3, 7], [3, 4, 5, 7], [6, 7, 8, 7]],
            [[1, 2, 3, 7], [3, 4, 5, 7], [6, 7, 8, 7]],
            [[1, 2, 3, 7], [3, 4, 5, 7], [6, 7, 8, 7]],
        ])
    )
    attn_msk = torch.LongTensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    aggregation = BaseAggregation()
    seq_len, expand_attn = aggregation.expand_attn_mask(
        states.last_hidden_state, attn_msk
    )

    sum_result = torch.LongTensor([[7, 9, 11, 14], [3, 4, 5, 7], [10, 13, 16, 21]])

    mean_result = torch.tensor([
        [3.5000, 4.5000, 5.5000, 7.0000],
        [3.0000, 4.0000, 5.0000, 7.0000],
        [3.33333333333333333, 4.33333333333333, 5.3333333333333, 7.0000],
    ])


def test_expanded_attn_sum():
    sample = TestSample()
    assert (
        (sample.states.last_hidden_state * sample.expand_attn).sum(dim=1)
        == sample.sum_result
    ).all()


def test_expanded_attn_seq_len_mean():
    sample = TestSample()
    assert (
        (
            (
                (sample.states.last_hidden_state * sample.expand_attn).sum(dim=1)
                / sample.seq_len.unsqueeze(1)
            ).expand(
                sample.states.last_hidden_state.shape[0],
                sample.states.last_hidden_state.shape[-1],
            )
        )
        - sample.mean_result
    ).sum() < 1e-40


def test_apply_expanded_mask():
    sample = TestSample()
    states_output, seq_len_output = sample.aggregation.apply_expanded_mask(
        sample.states, sample.attn_msk
    )
    seq_len_output = seq_len_output.unsqueeze(1).expand(
        sample.states.last_hidden_state.shape[0],
        sample.states.last_hidden_state.shape[-1],
    )
    assert (states_output.sum(dim=1) == sample.sum_result).all()
    assert (
        (states_output.sum(dim=1) / seq_len_output) - sample.mean_result
    ).sum() < 1e-40


def test_last_hidden_state():
    decoder_embedds = BaseSequenceOutput(last_hidden_state=torch.randn(3, 4, 2))
    attn_msk = torch.LongTensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]])
    last_agg = LastHiddenState()
    agg = last_agg(decoder_embedds, attn_msk)
    assert torch.all(
        decoder_embedds.last_hidden_state[[0, 1, 2], [3, 1, 0]] == agg
    ).item()


def test_mean_hidden_state():
    mean_aggregation = MeanHiddenState()
    states = BaseSequenceOutput(
        last_hidden_state=torch.LongTensor([
            [[1, 2, 3, 7], [3, 4, 5, 7], [6, 7, 8, 7]],
            [[1, 2, 3, 7], [3, 4, 5, 7], [6, 7, 8, 7]],
            [[1, 2, 3, 7], [3, 4, 5, 7], [6, 7, 8, 7]],
        ])
    )
    attn_msk = torch.LongTensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
    mean_result = torch.tensor([
        [3.5000, 4.5000, 5.5000, 7.0000],
        [3.0000, 4.0000, 5.0000, 7.0000],
        [3.33333333333333333, 4.33333333333333, 5.3333333333333, 7.0000],
    ])
    aggregated = mean_aggregation(states, attn_msk)
    assert (aggregated - mean_result).sum() < 1e-40
