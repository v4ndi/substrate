"""A worked example of a custom :class:`~avatar.train.TrainerCallback`.

Counts how often the gradient norm exceeded a threshold, and reports the total
across ranks once per epoch. Small on purpose — what it demonstrates is the two
rules that are easy to get wrong:

* hooks fire on **every** rank, so anything that must happen once guards on
  ``ctx.env.is_main`` itself;
* a collective must therefore be called *outside* that guard, on every rank —
  calling it inside would hang the run.

Register it from a config::

    callbacks:
      - _target_: avatar.train.MLflowCallback
      - _target_: avatar.train.ProgressBarCallback
      - _target_: examples.custom_callback.callback.GradientNormAlarm
        threshold: 1.0
      - _target_: avatar.train.CheckpointCallback
        checkpoint_dir: best_models/my_experiment/my_run

An explicit ``callbacks:`` list replaces the default one entirely, so the
checkpoint callback has to be listed too.
"""

from __future__ import annotations

from avatar.train import TrainerCallback
from avatar.train.state import CallbackContext


class GradientNormAlarm(TrainerCallback):
    """Count optimizer steps whose gradient norm exceeded ``threshold``.

    Args:
        threshold: Norm above which a step is counted.

    Logs ``grad/exceeded_steps`` (summed over ranks) and
    ``grad/exceeded_fraction`` once per epoch.

    Note:
        ``ctx.grad_norm`` is only populated when ``train.clip_grad_norm`` is
        set — clipping is what computes the norm. Without it this callback has
        nothing to measure and stays silent.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.exceeded = 0
        self.steps = 0

    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        self.exceeded = 0
        self.steps = 0

    def on_optimizer_step(self, ctx: CallbackContext) -> None:
        if ctx.grad_norm is None:
            return
        self.steps += 1
        if float(ctx.grad_norm) > self.threshold:
            self.exceeded += 1

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        # Every rank must reach the collective, so it goes before the is_main
        # guard, not inside it.
        total_exceeded = ctx.env.all_reduce_sum(float(self.exceeded))
        total_steps = ctx.env.all_reduce_sum(float(self.steps))
        if not (ctx.env.is_main and ctx.log is not None and total_steps):
            return
        ctx.log(
            {
                "grad/exceeded_steps": total_exceeded,
                "grad/exceeded_fraction": total_exceeded / total_steps,
            },
            step=ctx.state.epoch,
        )
