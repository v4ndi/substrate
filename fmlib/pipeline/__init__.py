"""Task pipelines: what goes in ``model:``.

A pipeline is the top-level module of a run. It owns the composition — which
embedding, which encoder, which head — and it is the only layer that computes a
loss: everything under :mod:`fmlib.nn` is loss-free by design, so the same
blocks can be reused across tasks.
:class:`~fmlib.pipeline.base.BasePipeline` states the contract every pipeline
owes the trainer.

One family today, over tabular batches:
:class:`~fmlib.pipeline.tabular.SupervisedLearner` and
:class:`~fmlib.pipeline.tabular.SLearner`. The directory level is kept because
the axis is the shape of the batch, not the task — a sequence pipeline would be
a sibling of ``tabular``, not a second kind of supervised learner.

See ``docs/guides/models.md`` for how the pieces compose,
``docs/reference/pipeline.md`` for the catalogue and
``docs/decisions/pipeline_boundaries.md`` for why the package looks like this.
"""

from fmlib.pipeline.base import BasePipeline
from fmlib.pipeline.tabular import SLearner, SupervisedLearner

__all__ = ["BasePipeline", "SLearner", "SupervisedLearner"]
