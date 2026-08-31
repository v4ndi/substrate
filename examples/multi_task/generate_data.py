"""Write a small multi-task tabular dataset for the MMoE / PLE example.

Three tasks over the same feature space, with deliberately different target
functions: two of them share most of their signal, the third is nearly
independent. That is what makes the gating visible — a mixture of experts is
only interesting when the tasks disagree about which features matter.

Emits what :class:`avatar.data.TabularDataset` expects after preprocessing,
plus the two columns the multi-task pipelines need: ``task_name`` and a
campaign ``group_id``.

Run::

    python examples/multi_task/generate_data.py
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
N_TASKS = 3
N_GROUPS = 2


def build_table(n_rows: int, seed: int) -> pa.Table:
    """Build one split: features, task id, campaign group, per-task target."""
    rng = np.random.default_rng(seed)

    per_column = VOCAB_SIZE // N_CAT
    cat = np.stack(
        [
            rng.integers(i * per_column, (i + 1) * per_column, n_rows)
            for i in range(N_CAT)
        ],
        axis=1,
    ).astype(np.int64)
    num = rng.standard_normal((n_rows, N_NUM)).astype(np.float32)

    task = rng.integers(0, N_TASKS, n_rows)
    group = rng.integers(0, N_GROUPS, n_rows)

    # The target function must be the SAME in every split — drawn from a fixed
    # seed, not from the split's. Redrawing it per split would give train and
    # valid different labelling rules, and validation would sit at chance.
    weights_rng = np.random.default_rng(20240101)
    # Tasks 0 and 1 share a direction; task 2 gets its own.
    shared = weights_rng.standard_normal(N_NUM)
    private = weights_rng.standard_normal((N_TASKS, N_NUM))
    weights = np.stack([
        shared + 0.2 * private[0],
        shared + 0.2 * private[1],
        private[2],
    ])

    signal = np.einsum("ij,ij->i", num, weights[task])
    signal = signal + 0.3 * (cat % per_column).mean(axis=1)
    noise = rng.standard_normal(n_rows) * 0.6
    target = ((signal + noise) > np.median(signal)).astype(np.int64)

    return pa.table({
        "epk_id": pa.array(np.arange(n_rows), pa.int64()),
        "cat_features": pa.array(cat.tolist(), pa.list_(pa.int64())),
        "num_features": pa.array(num.tolist(), pa.list_(pa.float32())),
        "task_name": pa.array([str(t) for t in task], pa.string()),
        "group_id": pa.array(group, pa.int64()),
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
    print(f"  {N_TASKS} tasks, {N_GROUPS} campaign groups")
    print("\nnow launch training with:\n")
    print(
        f"  python -m avatar.train "
        f"--config-dir={os.path.join(HERE, 'configs')} --config-name=mmoe \\\n"
        f"      data_dir={args.out}\n"
    )
    print("  # or the PLE variant:")
    print(
        f"  python -m avatar.train "
        f"--config-dir={os.path.join(HERE, 'configs')} --config-name=ple \\\n"
        f"      data_dir={args.out}"
    )


if __name__ == "__main__":
    main()
