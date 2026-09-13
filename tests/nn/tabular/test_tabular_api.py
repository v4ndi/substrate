"""Public API surface of avatar.nn.tabular."""

import importlib
import warnings

import pytest


def test_flat_public_api():
    import avatar.nn.tabular as tab

    for name in [
        "BaseTabularEncoder",
        "EncoderBlock",
        "MLPEmbedding",
        "SublayerConnection",
        "TabularTransformer",
        "build_feature_padding_mask",
    ]:
        assert hasattr(tab, name), name
        assert name in tab.__all__, name


def test_subpackage_paths():
    from avatar.nn.tabular.base.encoder import BaseTabularEncoder
    from avatar.nn.tabular.base.layers import EncoderBlock, SublayerConnection
    from avatar.nn.tabular.models.transformer import TabularTransformer
    from avatar.nn.tabular.utils.masking import build_feature_padding_mask

    assert issubclass(TabularTransformer, BaseTabularEncoder)
    assert EncoderBlock is not None and SublayerConnection is not None
    assert callable(build_feature_padding_mask)


def test_removed_symbols_are_gone():
    import avatar.nn.tabular as tab

    for name in [
        "BaseTabularBackbone",
        "DCNv2",
        "CrossNetV2",
        "STEv2",
        "STEv2Block",
        "MoE",
        "MultiTreatmentSTE",
    ]:
        assert not hasattr(tab, name), name


def test_no_deprecation_warning_from_top_level_import():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(importlib.import_module("avatar.nn.tabular"))


def test_deprecation_shim_is_gone():
    """avatar.nn.tabular.ste was dropped after configs moved to TabularTransformer."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("avatar.nn.tabular.ste")
