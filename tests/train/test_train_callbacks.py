"""Callbacks in isolation: dispatch, ordering guarantees and individual units."""

from __future__ import annotations

import pytest
import torch
from tiny_training import TinyModel

from avatar.train import (
    CallbackHandler,
    DistEnv,
    EarlyStopping,
    EarlyStoppingCallback,
    RunConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainStatsCallback,
)
from avatar.train.callbacks.ema import EMACallback
from avatar.train.callbacks.mlflow import sanitize_param_key, sanitize_params
from avatar.train.callbacks.profiler import ProfilerCallback
from avatar.train.loss_reduce import calculate_output_loss
from avatar.train.state import CallbackContext


@pytest.fixture
def ctx():
    logged: list[tuple[dict, int]] = []
    context = CallbackContext(
        env=DistEnv(device=torch.device("cpu")),
        state=TrainerState(),
        control=TrainerControl(),
        run_config=RunConfig(),
        log=lambda values, step=None: logged.append((values, step)),
    )
    context.extra["logged"] = logged
    return context


# -- dispatch ---------------------------------------------------------------


def test_handler_fires_callbacks_in_registration_order(ctx):
    seen: list[str] = []

    class Named(TrainerCallback):
        def __init__(self, name):
            self.name = name

        def on_step_end(self, ctx):
            seen.append(self.name)

    handler = CallbackHandler([Named("first"), Named("second")])
    handler.fire("on_step_end", ctx)
    assert seen == ["first", "second"]


def test_handler_rejects_unknown_events(ctx):
    with pytest.raises(ValueError, match="Unknown callback event"):
        CallbackHandler([]).fire("on_something_else", ctx)


def test_base_callback_hooks_are_all_no_ops(ctx):
    handler = CallbackHandler([TrainerCallback()])
    from avatar.train.callbacks.base import EVENTS

    for event in EVENTS:
        handler.fire(event, ctx)


# -- early stopping ----------------------------------------------------------


def test_early_stopping_allows_a_save_on_improvement(ctx):
    callback = EarlyStoppingCallback(
        EarlyStopping(main_metric="loss", patience=2, strategy="min")
    )
    ctx.metrics = {"loss": 1.0}
    ctx.control.should_save = True
    callback.on_evaluate(ctx)
    assert ctx.control.should_save is True
    assert ctx.control.should_training_stop is False


def test_early_stopping_vetoes_a_save_and_eventually_stops(ctx):
    callback = EarlyStoppingCallback(
        EarlyStopping(main_metric="loss", patience=2, strategy="min")
    )
    for loss in (1.0, 2.0, 3.0):
        ctx.metrics = {"loss": loss}
        ctx.control.should_save = True
        callback.on_evaluate(ctx)
    assert ctx.control.should_save is False
    assert ctx.control.should_training_stop is True


def test_early_stopping_ignores_empty_metrics(ctx):
    """Off the main rank there are no scores, and no decision to make."""
    callback = EarlyStoppingCallback(EarlyStopping(main_metric="loss"))
    ctx.metrics = {}
    ctx.control.should_save = True
    callback.on_evaluate(ctx)
    assert ctx.control.should_save is True


# -- train stats -------------------------------------------------------------


def test_train_stats_logs_epoch_means(ctx):
    callback = TrainStatsCallback()
    callback.on_epoch_begin(ctx)
    for loss in (1.0, 3.0):
        ctx.loss = torch.tensor(loss)
        callback.on_forward_end(ctx)
    ctx.state.learning_rate = 0.1
    ctx.grad_norm = torch.tensor(2.0)
    callback.on_optimizer_step(ctx)
    ctx.state.epoch_step = 1
    callback.on_epoch_end(ctx)

    values, step = ctx.extra["logged"][-1]
    assert values["train_loss"] == pytest.approx(2.0)
    assert values["mean_epoch_lr"] == pytest.approx(0.1)
    assert values["grad_norm"] == pytest.approx(2.0)
    assert step == ctx.state.epoch


def test_train_stats_resets_between_epochs(ctx):
    callback = TrainStatsCallback()
    callback.on_epoch_begin(ctx)
    ctx.loss = torch.tensor(10.0)
    callback.on_forward_end(ctx)
    callback.on_epoch_begin(ctx)
    ctx.loss = torch.tensor(1.0)
    callback.on_forward_end(ctx)
    ctx.state.epoch_step = 1
    callback.on_epoch_end(ctx)
    assert ctx.extra["logged"][-1][0]["train_loss"] == pytest.approx(1.0)


# -- EMA ---------------------------------------------------------------------


def test_ema_updates_on_schedule_and_publishes_the_averaged_model(ctx):
    model = TinyModel()
    swa = torch.optim.swa_utils.AveragedModel(model)
    callback = EMACallback(swa, update_every=2, min_epoch=0)
    ctx.model = model

    ctx.state.global_step = 1
    callback.on_step_end(ctx)
    assert "eval_model" not in ctx.extra

    ctx.state.global_step = 2
    callback.on_step_end(ctx)
    assert ctx.extra["eval_model"] is swa


def test_ema_waits_for_min_epoch(ctx):
    model = TinyModel()
    swa = torch.optim.swa_utils.AveragedModel(model)
    callback = EMACallback(swa, update_every=1, min_epoch=3)
    ctx.model = model
    ctx.state.epoch = 1
    ctx.state.global_step = 1
    callback.on_step_end(ctx)
    assert "eval_model" not in ctx.extra


def test_ema_rejects_a_zero_update_interval():
    model = TinyModel()
    with pytest.raises(ValueError, match="update_every"):
        EMACallback(torch.optim.swa_utils.AveragedModel(model), update_every=0)


# -- profiler ----------------------------------------------------------------


def test_profiler_is_started_stepped_and_stopped(ctx):
    class FakeProfiler:
        def __init__(self):
            self.calls: list[str] = []

        def start(self):
            self.calls.append("start")

        def step(self):
            self.calls.append("step")

        def stop(self):
            self.calls.append("stop")

    profiler = FakeProfiler()
    callback = ProfilerCallback(profiler)
    callback.on_train_begin(ctx)
    callback.on_step_end(ctx)
    callback.on_train_end(ctx)
    assert profiler.calls == ["start", "step", "stop"]


# -- MLflow param sanitising -------------------------------------------------


def test_mlflow_param_keys_are_sanitised():
    assert sanitize_param_key("model.encoder[0]") == "model.encoder_0_"
    assert sanitize_param_key("train_dataloader/batch") == "train_dataloader/batch"


def test_mlflow_drops_oversized_values():
    with pytest.warns(UserWarning, match="Skipping MLflow param"):
        params = sanitize_params({"short": 1, "long": "x" * 501})
    assert params == {"short": "1"}


# -- loss reduction ----------------------------------------------------------


class MultiHeadOutput:
    def __init__(self, losses, num_items):
        self.losses = losses
        self.num_items = num_items
        self.loss = None


def test_plain_loss_passes_through():
    class Output:
        loss = torch.tensor(1.5)
        num_items = None

    env = DistEnv(device=torch.device("cpu"))
    assert calculate_output_loss(Output(), env) is Output.loss


def test_multi_head_loss_is_token_weighted():
    env = DistEnv(device=torch.device("cpu"))
    output = MultiHeadOutput(
        losses={"a": torch.tensor(4.0), "b": torch.tensor(6.0)},
        num_items={"a": torch.tensor([2]), "b": torch.tensor([3])},
    )
    # (4/2 + 6/3) / 2 == 2.0 with world size 1.
    assert calculate_output_loss(output, env) == pytest.approx(2.0)


def test_heads_with_no_items_are_skipped():
    env = DistEnv(device=torch.device("cpu"))
    output = MultiHeadOutput(
        losses={"a": torch.tensor(4.0), "b": torch.tensor(6.0)},
        num_items={"a": torch.tensor([2]), "b": torch.tensor([0])},
    )
    # The empty head contributes nothing but still counts in the average.
    assert calculate_output_loss(output, env) == pytest.approx(1.0)
