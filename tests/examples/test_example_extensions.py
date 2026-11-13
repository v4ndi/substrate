"""The custom_callback and custom_loss examples must keep working.

They are documentation that happens to be executable, and they are the only
worked examples of the two extension points. Both are exercised here through
the same route a user takes — Hydra ``_target_`` instantiation — so a rename in
``avatar.train`` or ``avatar.losses`` breaks this test rather than the reader.
"""

from __future__ import annotations

import torch
from hydra.utils import instantiate

from avatar.train import DistEnv, RunConfig
from avatar.train.state import CallbackContext, TrainerControl, TrainerState

CALLBACK_TARGET = "examples.custom_callback.callback.GradientNormAlarm"
LOSS_TARGET = "examples.custom_loss.loss.FocalLoss"


def make_context(**extra) -> CallbackContext:
    """A context shaped like the one the loop builds, with a capturing log."""
    logged: dict = {}
    ctx = CallbackContext(
        env=DistEnv(),
        state=TrainerState(),
        control=TrainerControl(),
        run_config=RunConfig(),
        log=lambda values, step=None: logged.update(values),
        **extra,
    )
    ctx.extra["logged"] = logged
    return ctx


# --------------------------------------------------------------------------- #
# custom_loss                                                                  #
# --------------------------------------------------------------------------- #
def test_focal_loss_instantiates_from_its_documented_target():
    loss = instantiate({"_target_": LOSS_TARGET, "gamma": 2.0})
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.1, -0.1]])
    targets = torch.tensor([0, 1, 0])

    result = loss(logits, targets)

    assert result.loss.ndim == 0
    assert "cross_entropy" in result.components
    assert result.loss.requires_grad is False  # inputs do not require grad


def test_focal_loss_down_weights_easy_examples():
    """gamma > 0 must shrink the loss relative to plain cross-entropy."""
    loss = instantiate({"_target_": LOSS_TARGET, "gamma": 2.0})
    confident = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    targets = torch.tensor([0, 1])

    result = loss(confident, targets)

    assert float(result.loss) < float(result.components["cross_entropy"])


def test_focal_loss_accepts_the_model_argument_the_pipeline_passes():
    loss = instantiate({"_target_": LOSS_TARGET})
    result = loss(
        torch.tensor([[1.0, 0.0]]), torch.tensor([0]), model=torch.nn.Linear(2, 2)
    )
    assert result.loss.ndim == 0


# --------------------------------------------------------------------------- #
# custom_callback                                                              #
# --------------------------------------------------------------------------- #
def test_gradient_alarm_instantiates_from_its_documented_target():
    callback = instantiate({"_target_": CALLBACK_TARGET, "threshold": 1.0})
    assert callback.threshold == 1.0


def test_gradient_alarm_counts_and_logs_the_fraction():
    callback = instantiate({"_target_": CALLBACK_TARGET, "threshold": 1.0})
    ctx = make_context()

    callback.on_epoch_begin(ctx)
    for norm in (0.5, 2.0, 3.0, 0.1):
        ctx.grad_norm = norm
        callback.on_optimizer_step(ctx)
    callback.on_epoch_end(ctx)

    logged = ctx.extra["logged"]
    assert logged["grad/exceeded_steps"] == 2.0
    assert logged["grad/exceeded_fraction"] == 0.5


def test_gradient_alarm_is_silent_without_clipping():
    """`ctx.grad_norm` is None unless train.clip_grad_norm is set."""
    callback = instantiate({"_target_": CALLBACK_TARGET})
    ctx = make_context()

    callback.on_epoch_begin(ctx)
    for _ in range(3):
        ctx.grad_norm = None
        callback.on_optimizer_step(ctx)
    callback.on_epoch_end(ctx)

    assert ctx.extra["logged"] == {}


def test_gradient_alarm_resets_between_epochs():
    callback = instantiate({"_target_": CALLBACK_TARGET, "threshold": 1.0})
    ctx = make_context()

    callback.on_epoch_begin(ctx)
    ctx.grad_norm = 5.0
    callback.on_optimizer_step(ctx)
    callback.on_epoch_begin(ctx)

    assert callback.exceeded == 0
    assert callback.steps == 0
