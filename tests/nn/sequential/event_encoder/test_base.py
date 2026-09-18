import pytest

from fmlib.nn.embedding import BaseEmbedding
from fmlib.nn.sequential.event_encoder.base import BaseEventEncoder


def _embedding():
    return BaseEmbedding(hidden_size=16)


def test_time_encoding_none_builds_no_layer():
    enc = BaseEventEncoder(embedding=_embedding(), time_encoding=None)
    assert enc.time_encoding is None
    assert not hasattr(enc, "time_encoding_layer")


def test_time_encoding_default_is_none():
    assert BaseEventEncoder(embedding=_embedding()).time_encoding is None


@pytest.mark.parametrize("mode", ["delta", "absolute"])
def test_time_encoding_builds_layer(mode):
    enc = BaseEventEncoder(embedding=_embedding(), time_encoding=mode)
    assert enc.time_encoding == mode
    assert hasattr(enc, "time_encoding_layer")


def test_time_encoding_invalid_raises():
    with pytest.raises(ValueError, match="time_encoding"):
        BaseEventEncoder(embedding=_embedding(), time_encoding="bogus")
