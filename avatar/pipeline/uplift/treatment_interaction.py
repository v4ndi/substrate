"""Deprecated import path for :mod:`avatar.pipeline.tabular.interaction`.

The interaction blocks serve the group embedding as well as the treatment one,
so they were never uplift-specific; they sat here only because uplift was the
first caller. Ten configs name this module directly, which is why the path is
kept for one release.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # resolved lazily below; declared here for static tools
    from avatar.pipeline.tabular.interaction import (
        BaseTreatmentInteraction,
        ConcatTreatmentInteraction,
        ElementwiseTreatmentInteraction,
        IgnoreTreatmentInteraction,
        SumTreatmentInteraction,
    )

__all__ = [
    "BaseTreatmentInteraction",
    "ConcatTreatmentInteraction",
    "ElementwiseTreatmentInteraction",
    "IgnoreTreatmentInteraction",
    "SumTreatmentInteraction",
]


def __getattr__(name: str) -> Any:
    """Resolve a name from the new location, with a deprecation warning.

    Raises:
        AttributeError: The name was never part of this module.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from avatar.pipeline.tabular import interaction

    warnings.warn(
        f"avatar.pipeline.uplift.treatment_interaction.{name} has moved to "
        f"avatar.pipeline.tabular.interaction.{name}; the old path will be "
        "removed in the next release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(interaction, name)


def __dir__() -> list[str]:
    return sorted(__all__)
