"""Deprecated alias for :mod:`avatar.infer`.

Kept so ``python -m avatar.inference`` keeps working while configs and launch
scripts move over.
"""

import warnings

from avatar.infer import log_scores, main

warnings.warn(
    "avatar.inference is deprecated; use avatar.infer instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["log_scores", "main"]


if __name__ == "__main__":
    main()
