"""The contract every ``model:`` must satisfy.

A pipeline is the top-level module of a run: the trainer builds it from the
``model:`` block and calls it with the batch. Three rules follow from how it is
called, and all three have been broken at least once:

1. **``forward`` is called as ``model(**batch)``.** Every key the collate
   function emits arrives as a keyword argument, so ``forward`` must end in
   ``**kwargs`` — a batch that carries one extra column must not be a
   ``TypeError``.
2. **Targets are optional.** Inference has no labels. A pipeline that touches
   ``targets`` unconditionally dies on the first inference batch, which is
   exactly what ``SupervisedLearner`` used to do with ``_ = targets.device``.
3. **The return value carries ``.loss``.** That is all
   :mod:`avatar.train` reads (or ``losses`` plus ``num_items`` for the
   multi-head case). Everything else on the output exists for metrics.

:data:`BasePipeline.required_inputs` states which batch keys the pipeline
cannot work without, the way :class:`~avatar.metrics.base.BaseMetric` states
what a metric reads. It is what makes a missing column a readable error rather
than a ``TypeError`` three frames down, and
``tests/pipeline/test_pipeline_contract.py`` checks every subclass against it.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["BasePipeline"]


class BasePipeline(nn.Module):
    """Base class for task pipelines — the object named in ``model:``.

    Subclasses implement :meth:`forward` and declare
    :attr:`required_inputs`. There is deliberately nothing else here: the
    trainer needs no registry, since Hydra builds the pipeline from its
    ``_target_``, and a base class that did more would have to know about
    tasks, which is the one thing the layer below it must not.
    """

    #: Batch keys :meth:`forward` cannot work without. Optional inputs
    #: (``targets``, ``group``) are *not* listed: they have defaults and the
    #: pipeline runs without them.
    required_inputs: tuple[str, ...] = ()

    def forward(self, **batch):  # pragma: no cover - abstract
        """Score the batch, and compute the loss when targets are present.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def missing_inputs(self, batch: dict) -> list[str]:
        """Which of :attr:`required_inputs` the batch does not carry.

        Args:
            batch: The keyword arguments ``forward`` is about to receive.

        Returns:
            The missing key names, in declaration order.
        """
        return [name for name in self.required_inputs if batch.get(name) is None]
