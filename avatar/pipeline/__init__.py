"""Task pipelines: what goes in ``model:``.

A pipeline is the top-level module of a run. It owns the composition — which
embedding, which encoder, which head — and it is the only layer that computes a
loss: everything under :mod:`avatar.nn` is loss-free by design, so the same
blocks can be reused across tasks.

Two families, by what the task is. Both take a tabular batch — one row per
record:

* :mod:`~avatar.pipeline.tabular` — supervised: binary, multi-class, regression.
* :mod:`~avatar.pipeline.uplift` — treatment/control, scored on the difference
  between the two passes.

See ``docs/guides/models.md`` for how the pieces compose and
``docs/reference/pipeline.md`` for the catalogue.
"""
