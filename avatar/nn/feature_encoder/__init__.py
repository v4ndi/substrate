"""Deprecated -- moved to :mod:`avatar.nn.sequential.event_encoder`.

Kept for one release so out-of-repo Hydra configs referencing
``avatar.nn.feature_encoder.*`` keep resolving. Import from
``avatar.nn.sequential`` instead.

Name changes:
``BaseSequenceFeatureEncoder`` -> ``BaseEventEncoder``,
``FeatureEncoder`` / ``FeatureAttentionEncoder`` -> ``EventEncoder``.
"""

import warnings

from avatar.nn.sequential import (
    BaseEventEncoder as BaseSequenceFeatureEncoder,
)
from avatar.nn.sequential import (
    EventEncoder as FeatureAttentionEncoder,
)
from avatar.nn.sequential import (
    EventEncoder as FeatureEncoder,
)

warnings.warn(
    "avatar.nn.feature_encoder has moved to avatar.nn.sequential; "
    "update imports and Hydra _target_ paths.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BaseSequenceFeatureEncoder",
    "FeatureAttentionEncoder",
    "FeatureEncoder",
]
