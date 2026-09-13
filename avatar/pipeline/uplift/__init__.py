"""Deprecated import path. Uplift lives in :mod:`avatar.pipeline.tabular`.

``SLearner`` consumes a ``TabularBatch`` and is built from tabular blocks, so
``tabular`` — the shape of the batch — is where it belongs; ``uplift`` named
the task, which is a different axis. The class, its arguments and its
checkpoint keys are unchanged.

This shim exists because ``avatar.pipeline.uplift.SLearner`` is named by some
forty-five configs, several of them records of production runs, and by cluster
configs outside the repository. Configs in the repository have been updated;
this path is kept for one release.

Names are resolved lazily, so importing the module warns about nothing — only
reaching a class through it does.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # resolved lazily below; declared here for static tools
    from avatar.pipeline.tabular import SLearner
    from avatar.pipeline.tabular.interaction import IgnoreTreatmentInteraction

__all__ = ["IgnoreTreatmentInteraction", "SLearner"]


def __getattr__(name: str) -> Any:
    """Resolve a name from the new location, with a deprecation warning.

    Raises:
        AttributeError: The name was never part of this package.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from avatar.pipeline import tabular

    warnings.warn(
        f"avatar.pipeline.uplift.{name} has moved to "
        f"avatar.pipeline.tabular.{name}; the old path will be removed in the "
        "next release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(tabular, name)


def __dir__() -> list[str]:
    return sorted(__all__)
