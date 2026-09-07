"""Write a small preprocessed tabular dataset for the distributed example.

Produces exactly what :class:`avatar.data.TabularDataset` expects *after*
preprocessing — ``cat_features`` (list<int64>), ``num_features``
(list<float32>), a binary ``target`` and an ``epk_id`` — so the example needs no
Spark, no preprocessing pass and no access to internal storage.

The target is a noisy linear function of the features, so the run has something
learnable to show and ROC AUC climbs above 0.5 within a couple of epochs.

Run::

    python examples/distributed_training/generate_data.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))

N_CAT = 8
N_NUM = 16
VOCAB_SIZE = 64


def build_table(n_rows: int, seed: int) -> pa.Table:
    """Build one split: ids, categorical ids, standardised numerics, target."""
    rng = np.random.default_rng(seed)

    # Cumulative-offset ids, as the real preprocessor emits: every categorical
    # column indexes into one shared embedding table of size VOCAB_SIZE.
    per_column = VOCAB_SIZE // N_CAT
    cat = np.stack(
        [
            rng.integers(i * per_column, (i + 1) * per_column, n_rows)
            for i in range(N_CAT)
        ],
        axis=1,
    ).astype(np.int64)
    num = rng.standard_normal((n_rows, N_NUM)).astype(np.float32)

    # The target function must be the SAME in every split — drawn from a fixed
    # seed, not from the split's. Redrawing it per split would give train and
    # valid different labelling rules, and validation would sit at chance.
    weights_rng = np.random.default_rng(20240101)
    cat_weights = weights_rng.standard_normal(N_CAT)
    num_weights = weights_rng.standard_normal(N_NUM)
    signal = (cat % per_column) @ cat_weights / per_column + num @ num_weights
    noise = rng.standard_normal(n_rows) * 0.5
    target = ((signal + noise) > np.median(signal)).astype(np.int64)

    return pa.table({
        "epk_id": pa.array(np.arange(n_rows), pa.int64()),
        "cat_features": pa.array(cat.tolist(), pa.list_(pa.int64())),
        "num_features": pa.array(num.tolist(), pa.list_(pa.float32())),
        "target": pa.array(target, pa.int64()),
    })


def write_split(table: pa.Table, directory: str, n_files: int) -> None:
    """Write ``table`` as ``n_files`` parquet parts under ``directory``."""
    os.makedirs(directory, exist_ok=True)
    step = -(-table.num_rows // n_files)
    for index, start in enumerate(range(0, table.num_rows, step)):
        pq.write_table(
            table.slice(start, step),
            os.path.join(directory, f"part-{index:03d}.parquet"),
        )


def main() -> None:
    """Write train and valid splits, then print the launch command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "data"))
    parser.add_argument("--train-rows", type=int, default=40_000)
    parser.add_argument("--valid-rows", type=int, default=8_000)
    parser.add_argument("--files", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    write_split(
        build_table(args.train_rows, args.seed), f"{args.out}/train", args.files
    )
    write_split(
        build_table(args.valid_rows, args.seed + 1), f"{args.out}/valid", args.files
    )

    print(f"wrote {args.train_rows} train / {args.valid_rows} valid rows to {args.out}")
    print(f"  {N_CAT} categorical (vocab_size={VOCAB_SIZE}), {N_NUM} numeric")
    print("\nnow launch training with:\n")
    print(
        f"  torchrun --standalone --nproc_per_node=2 -m avatar.train \\\n"
        f"      --config-dir={os.path.join(HERE, 'configs')} "
        f"--config-name=synthetic \\\n"
        f"      data_dir={args.out}"
    )


if __name__ == "__main__":
    main()
