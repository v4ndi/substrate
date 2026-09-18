"""Public API surface of fmlib.nn.sequential."""

import warnings

import pytest


def test_flat_public_api():
    import fmlib.nn.sequential as seq

    for name in [
        "BaseEventEncoder",
        "EventEncoder",
        "EventAggregator",
        "IntraFeatureAttention",
        "build_event_attention_mask",
        "BaseBackbone",
        "SequenceBackbone",
        "BaseSequenceModel",
        "TransformersWrapper",
    ]:
        assert hasattr(seq, name), name
        assert name in seq.__all__, name


def test_subpackage_paths():
    from fmlib.nn.sequential.backbone.base import BaseBackbone, SequenceBackbone
    from fmlib.nn.sequential.event_encoder.attention import (
        EventAggregator,
        build_event_attention_mask,
    )
    from fmlib.nn.sequential.event_encoder.base import BaseEventEncoder
    from fmlib.nn.sequential.event_encoder.event import EventEncoder
    from fmlib.nn.sequential.model.base import BaseSequenceModel
    from fmlib.nn.sequential.model.transformers import TransformersWrapper

    assert issubclass(EventEncoder, BaseEventEncoder)
    assert issubclass(TransformersWrapper, BaseSequenceModel)
    assert callable(build_event_attention_mask)
    assert BaseBackbone is not None and SequenceBackbone is not None
    assert EventAggregator is not None


def test_feature_encoder_attribute_alias_on_model():
    import torch.nn as nn

    from fmlib.nn.sequential import BaseSequenceModel

    encoder, backbone = nn.Identity(), nn.Identity()
    model = BaseSequenceModel(event_encoder=encoder, backbone=backbone)
    assert model.feature_encoder is model.event_encoder is encoder


def test_no_deprecation_warning_from_top_level_import():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import importlib

        importlib.reload(importlib.import_module("fmlib.nn.sequential"))


def test_deprecation_shims_are_gone():
    """fmlib.nn.sequence / fmlib.nn.feature_encoder were dropped after migration."""
    import importlib

    for name in ("fmlib.nn.sequence", "fmlib.nn.feature_encoder"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
