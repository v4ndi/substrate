"""Model building blocks.

Nothing here computes a loss: that belongs to :mod:`fmlib.pipeline`, which
is what lets the same encoder be reused across tasks and run at inference
with no targets. See ``docs/guides/models.md`` and
``docs/reference/nn.md``.
"""

from . import embedding, sequential, tabular, utils

__all__ = ["embedding", "sequential", "tabular", "utils"]
