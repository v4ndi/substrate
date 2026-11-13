"""Worker launched by ``torchrun`` for the real distributed sharding test.

Each rank builds the dataset, iterates its shard through a DataLoader, and
reports what it saw. The results are gathered with a genuine ``all_gather_object``
over the gloo backend, so the assertions run against real collectives rather
than a mocked ``torch.distributed``.

Not a pytest module: it is spawned by ``tests/data/test_distributed_sharding.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from avatar.data.sequential import EventSequenceCollateFn, EventSequenceDataset
from avatar.data.tabular import TabularCollateFn, TabularDataset

SEQUENCE_COLUMNS = ["mcc", "price"]


def build_dataset(
    modality: str, path: str, min_length: int, selected_event_ids, rotate_tail: bool
):
    if modality == "sequence":
        return EventSequenceDataset(
            path=path,
            sequence_columns=SEQUENCE_COLUMNS,
            event_ids_column="event_ids",
            selected_event_ids=selected_event_ids,
            min_length=min_length,
            max_length=100,
            shuffle_files=False,
            shuffle_pq=False,
            rotate_tail=rotate_tail,
        )
    return TabularDataset(
        path=path, shuffle_files=False, shuffle_pq=False, rotate_tail=rotate_tail
    )


def build_collate(modality: str):
    if modality == "sequence":
        return EventSequenceCollateFn(sequence_columns=SEQUENCE_COLUMNS)
    return TabularCollateFn()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--modality", default="sequence")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rotate-tail", action="store_true")
    parser.add_argument("--selected-event-ids", default="")
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    selected = (
        [int(value) for value in args.selected_event_ids.split(",")]
        if args.selected_event_ids
        else None
    )
    dataset = build_dataset(
        args.modality, args.path, args.min_length, selected, args.rotate_tail
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=build_collate(args.modality),
        drop_last=False,
    )

    epochs = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        batches = list(loader)
        ids = [int(value) for batch in batches for value in batch["epk_id"]]
        epochs.append({
            "ids": ids,
            "num_batches": len(batches),
            "len": len(dataset),
        })

    # A real collective: if the ranks disagreed on cardinality this is where a
    # DDP job would hang, so exercising it is the point of the test.
    gathered: list = [None] * world_size
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "epochs": epochs,
            "total": dataset.planner.total,
            "tail": dataset.planner.tail,
        },
    )

    if rank == 0:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"world_size": world_size, "ranks": gathered}, handle)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
