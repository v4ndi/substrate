"""Deprecated alias for :mod:`avatar.training_arguments` (module name typo fix).

Kept for one release so external imports of the misspelled name keep working.
"""

import warnings

from avatar.training_arguments import *  # noqa: F403
from avatar.training_arguments import TrainingArguments  # noqa: F401

warnings.warn(
    "avatar.training_agruments is deprecated; import from avatar.training_arguments",
    DeprecationWarning,
    stacklevel=2,
)
