"""Deprecated. ``avatar.nn.tabular.ste`` has moved to ``avatar.nn.tabular``.

``STEv2Block`` and ``STEv2`` are both replaced by
:class:`avatar.nn.tabular.TabularTransformer` (a feature-token encoder that takes
embeddings, not a ``TabularBatch``, and returns a ``BaseTabularOutput``).

This shim is kept for one release for out-of-repo Hydra configs. ``STEv2``'s old
``embedding=``-owning signature is *not* reproduced - configs that instantiated
``STEv2`` with a nested ``embedding`` must move the embedding up to the pipeline
(see ``TabularWithAggregatedStates``).
"""

import warnings

from avatar.nn.tabular.base.layers import EncoderBlock, SublayerConnection
from avatar.nn.tabular.models.transformer import TabularTransformer

warnings.warn(
    "avatar.nn.tabular.ste has moved to avatar.nn.tabular; "
    "STEv2Block/STEv2 are now avatar.nn.tabular.TabularTransformer",
    DeprecationWarning,
    stacklevel=2,
)

STEv2Block = TabularTransformer
STEv2 = TabularTransformer

__all__ = ["EncoderBlock", "STEv2", "STEv2Block", "SublayerConnection"]
