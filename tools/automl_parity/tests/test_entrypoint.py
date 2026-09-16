"""Parity adapter contracts."""

from tools.automl_parity.autocampaign_entrypoint import _decode_name, _encode_path


def test_reference_artifact_names_are_windows_safe_and_reversible() -> None:
    logical = "configs_product/<'catboost_binary_clf'>_product.txt"

    encoded = _encode_path(logical)

    assert "<" not in encoded and ">" not in encoded
    assert _decode_name(encoded.split("/")[-1]) == "<'catboost_binary_clf'>_product.txt"
