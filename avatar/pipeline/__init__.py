"""Task pipelines: what goes in ``model:``.

A pipeline is the top-level module of a run. It owns the composition — which
embedding, which encoder, which head — and it is the only layer that computes a
loss: everything under :mod:`avatar.nn` is loss-free by design, so the same
blocks can be reused across tasks.

Four families, by what the batch looks like and what the task is:

* :mod:`~avatar.pipeline.tabular` — one row per record, supervised.
* :mod:`~avatar.pipeline.sequence` — event sequences, supervised or
  self-supervised.
* :mod:`~avatar.pipeline.uplift` — treatment/control, scored on the difference
  between the two heads.
* :mod:`~avatar.pipeline.multi_task` — several tasks over shared experts.

See ``docs/guides/models.md`` for how the pieces compose and
``docs/reference/pipeline.md`` for the catalogue.
"""
