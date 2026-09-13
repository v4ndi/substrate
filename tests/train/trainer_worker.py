"""Run a real :class:`~avatar.train.loop.Trainer` under ``torchrun``.

Launched by ``tests/train/test_distributed_training.py``. Each rank trains the
tiny synthetic task, then reports what it saw; rank 0 gathers the reports and
writes them as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tiny_training import (
    TinyModel,
    TinyShardedDataset,
    collate,
    parameter_fingerprint,
)

from avatar.train import (
    CheckpointCallback,
    DistEnv,
    RunConfig,
    Trainer,
    TrainStatsCallback,
    load_checkpoint,
    unwrap_model,
)
from avatar.train.config import DDPConfig, DistributedConfig
from avatar.training_arguments import TrainingArguments


def build_loader(num_records, batch_size, env, seed, num_workers=0):
    dataset = TinyShardedDataset(
        num_records=num_records,
        seed=seed,
        world_size=env.world_size,
        rank=env.rank,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        drop_last=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-records", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp", default="no")
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    run_config = RunConfig(
        distributed=DistributedConfig(
            backend="gloo", gradient_accumulation_steps=args.grad_accum
        ),
        ddp=DDPConfig(find_unused_parameters=args.find_unused_parameters),
        amp=args.amp,
    )
    env = DistEnv.from_env(backend="gloo")

    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)

    train_loader = build_loader(
        args.num_records, args.batch_size, env, seed=0, num_workers=args.num_workers
    )
    valid_loader = build_loader(args.num_records // 4, args.batch_size, env, seed=1)

    checkpoint_dir = args.checkpoint_dir or str(Path(args.out).parent / "checkpoints")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        training_arguments=TrainingArguments(
            num_epochs=args.epochs,
            seed=42,
            clip_grad_norm=args.clip_grad_norm,
        ),
        run_config=run_config,
        env=env,
        callbacks=[TrainStatsCallback(), CheckpointCallback(checkpoint_dir)],
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train()

    train_losses = [
        entry["train_loss"]
        for entry in trainer.state.log_history
        if "train_loss" in entry
    ]
    valid_losses = [
        entry["valid_loss"]
        for entry in trainer.state.log_history
        if "valid_loss" in entry
    ]

    # A saved checkpoint must restore bit-identical weights.
    roundtrip_matches = None
    # Step directories are numbered, so sort them numerically, not by name.
    saved = sorted(
        Path(checkpoint_dir).glob("*/checkpoint.pt"),
        key=lambda path: int(path.parent.name),
    )
    if saved:
        fresh = TinyModel()
        load_checkpoint(str(saved[-1]), model=fresh, restore_rng=False)
        trained = unwrap_model(trainer.model).state_dict()
        restored = fresh.state_dict()
        roundtrip_matches = all(
            torch.equal(value.cpu(), restored[key].cpu())
            for key, value in trained.items()
        )

    report = {
        "rank": env.rank,
        "world_size": env.world_size,
        "global_step": trainer.state.global_step,
        "epoch_batches": trainer.state.epoch_batches,
        "samples_seen": trainer.state.samples_seen,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "params": parameter_fingerprint(unwrap_model(trainer.model)),
        "checkpoint_roundtrip": roundtrip_matches,
        "checkpoints": [path.parent.name for path in saved],
    }
    gathered = env.gather_objects(report)

    if env.is_main:
        Path(args.out).write_text(
            json.dumps(
                {
                    "world_size": env.world_size,
                    "ranks": sorted(gathered, key=lambda item: item["rank"]),
                },
                indent=2,
            )
        )
    env.barrier()
    env.destroy()


if __name__ == "__main__":
    main()
