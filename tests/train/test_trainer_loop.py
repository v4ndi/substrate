"""Single-process behaviour of the loop, its control flow and its callbacks."""

from __future__ import annotations

import torch
from tiny_training import TinyModel, TinyShardedDataset, collate
from torch.utils.data import DataLoader

from avatar.train import (
    CallbackContext,
    CheckpointCallback,
    DistEnv,
    EarlyStopping,
    EarlyStoppingCallback,
    RunConfig,
    Trainer,
    TrainerCallback,
    TrainStatsCallback,
    load_checkpoint,
)
from avatar.train.checkpoint import checkpoint_path, latest_checkpoint_step
from avatar.train.config import DistributedConfig
from avatar.training_arguments import TrainingArguments


def newest_checkpoint(directory) -> str | None:
    """The highest-numbered step directory.

    Not ``sorted(glob(...))``: step directories are named by number, so
    lexicographic order puts "16" before "8".
    """
    step = latest_checkpoint_step(str(directory))
    return None if step is None else checkpoint_path(str(directory), step)


class RecordingCallback(TrainerCallback):
    """Note every event, in order, so the loop's contract can be asserted on."""

    def __init__(self):
        self.events: list[str] = []

    def __getattribute__(self, name):
        if name.startswith("on_"):
            events = object.__getattribute__(self, "events")
            return lambda ctx: events.append(name)
        return object.__getattribute__(self, name)


def build_trainer(tmp_path, callbacks=None, num_records=64, valid=True, **kwargs):
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
    loader = DataLoader(
        TinyShardedDataset(num_records=num_records),
        batch_size=8,
        collate_fn=collate,
        drop_last=True,
    )
    valid_loader = (
        DataLoader(
            TinyShardedDataset(num_records=16, seed=1),
            batch_size=8,
            collate_fn=collate,
            drop_last=True,
        )
        if valid
        else None
    )
    arguments = TrainingArguments(
        num_epochs=kwargs.pop("num_epochs", 2), seed=1, **kwargs
    )
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=loader,
        valid_dataloader=valid_loader,
        training_arguments=arguments,
        run_config=RunConfig(distributed=DistributedConfig(backend="gloo")),
        env=DistEnv(device=torch.device("cpu")),
        callbacks=callbacks or [],
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )


def test_loss_decreases_over_epochs(tmp_path):
    stats = TrainStatsCallback()
    trainer = build_trainer(tmp_path, callbacks=[stats], num_epochs=5)
    trainer.train()
    losses = [
        entry["train_loss"]
        for entry in trainer.state.log_history
        if "train_loss" in entry
    ]
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_state_counts_steps_batches_and_samples(tmp_path):
    trainer = build_trainer(tmp_path, num_records=64, num_epochs=1)
    trainer.train()
    # 64 records, batch size 8 -> 8 batches, one optimizer step each.
    assert trainer.state.epoch_batches == 8
    assert trainer.state.global_step == 8
    assert trainer.state.samples_seen == 64


def test_gradient_accumulation_reduces_optimizer_steps(tmp_path):
    trainer = build_trainer(tmp_path, num_records=64, num_epochs=1)
    trainer.grad_accum = 2
    trainer.train()
    assert trainer.state.epoch_batches == 8
    assert trainer.state.global_step == 4


def test_callback_events_fire_in_order(tmp_path):
    recorder = RecordingCallback()
    trainer = build_trainer(tmp_path, callbacks=[recorder], num_epochs=1)
    trainer.train()
    events = recorder.events

    assert events[0] == "on_train_begin"
    assert "on_train_end" in events
    assert events.index("on_epoch_begin") < events.index("on_batch_begin")
    assert events.index("on_batch_begin") < events.index("on_forward_end")
    assert events.index("on_forward_end") < events.index("on_backward_end")
    assert events.index("on_backward_end") < events.index("on_optimizer_step")
    assert events.index("on_optimizer_step") < events.index("on_step_end")
    # Per-epoch evaluation happens after the epoch is closed out.
    assert events.index("on_epoch_end") < events.index("on_evaluate")


def test_evaluation_produces_validation_scores(tmp_path):
    trainer = build_trainer(tmp_path, num_epochs=2)
    trainer.train()
    valid = [entry for entry in trainer.state.log_history if "valid_loss" in entry]
    assert len(valid) == 2


def test_checkpoint_is_written_and_restores_the_weights(tmp_path):
    trainer = build_trainer(
        tmp_path, callbacks=[CheckpointCallback(str(tmp_path / "checkpoints"))]
    )
    trainer.train()

    saved = newest_checkpoint(tmp_path / "checkpoints")
    assert saved is not None, "no checkpoint written"

    fresh = TinyModel()
    payload = load_checkpoint(saved, model=fresh, restore_rng=False)
    assert payload["state"]["global_step"] == trainer.state.global_step
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key])


def test_early_stopping_halts_training(tmp_path):
    # patience 0 with a metric that cannot improve on its very first reading.
    stopping = EarlyStopping(main_metric="loss", patience=1, strategy="min")
    stopping.best_score = 10.0  # any real loss is worse than this
    trainer = build_trainer(
        tmp_path,
        callbacks=[EarlyStoppingCallback(stopping)],
        num_epochs=6,
    )
    trainer.train()
    assert trainer.control.should_training_stop
    assert trainer.state.epoch < 5, "training should have stopped early"


def test_early_stopping_vetoes_a_non_improving_save(tmp_path):
    stopping = EarlyStopping(main_metric="loss", patience=10, strategy="min")
    stopping.best_score = 10.0
    checkpointer = CheckpointCallback(str(tmp_path / "checkpoints"))
    trainer = build_trainer(
        tmp_path,
        callbacks=[EarlyStoppingCallback(stopping), checkpointer],
        num_epochs=2,
    )
    trainer.train()
    # The metric never improves, so nothing should have been written.
    assert not list((tmp_path / "checkpoints").glob("*/checkpoint.pt"))


def test_a_callback_can_stop_training_early(tmp_path):
    class StopAfterFirstEpoch(TrainerCallback):
        def on_epoch_end(self, ctx: CallbackContext) -> None:
            ctx.control.should_training_stop = True

    trainer = build_trainer(tmp_path, callbacks=[StopAfterFirstEpoch()], num_epochs=5)
    trainer.train()
    assert trainer.state.epoch == 0


def test_step_level_evaluation_uses_a_literal_step_budget(tmp_path):
    """No ``// num_processes`` fudge: every N optimizer steps means exactly that."""
    trainer = build_trainer(
        tmp_path, num_records=64, num_epochs=1, steps_before_evaluation=4
    )
    trainer.train()
    valid = [entry for entry in trainer.state.log_history if "valid_loss" in entry]
    # 8 steps, evaluating on steps 4 and 8.
    assert len(valid) == 2


def test_resume_restores_step_and_weights(tmp_path):
    first = build_trainer(
        tmp_path,
        callbacks=[CheckpointCallback(str(tmp_path / "checkpoints"))],
        num_epochs=2,
    )
    first.train()
    saved = newest_checkpoint(tmp_path / "checkpoints")

    resumed = build_trainer(tmp_path / "second", num_epochs=2, checkpoint_state=saved)
    assert resumed.state.global_step == first.state.global_step
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
