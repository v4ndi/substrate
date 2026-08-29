"""Deprecated -- moved to :mod:`avatar.nn.sequential`.

Kept for one release so out-of-repo Hydra configs referencing
``avatar.nn.sequence.*`` keep resolving. Import from ``avatar.nn.sequential``
instead.
"""

import warnings

from avatar.nn.sequential import (
    BaseBackbone as BaseSequenceBackbone,
)
from avatar.nn.sequential import (
    BaseSequenceModel,
    TransformersWrapper,
)

warnings.warn(
    "avatar.nn.sequence has moved to avatar.nn.sequential; "
    "update imports and Hydra _target_ paths.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseSequenceBackbone", "BaseSequenceModel", "TransformersWrapper"]
