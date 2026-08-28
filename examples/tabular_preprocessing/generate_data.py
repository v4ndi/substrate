"""Generate a synthetic wide tabular dataset for the preprocessing example.

Writes several parquet part-files (so the streaming / multi-file code paths are
exercised) with a mix of categorical and numeric columns, some nulls, and a
target. Layout mimics the aggregate-mart tables used by the tabular transformer
example (:mod:`examples.tabular_hidden_states`).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def build_table(n_rows: int = 200_000, n_num: int = 24, n_cat: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols: dict[str, pa.Array] = {"epk_id": pa.array(np.arange(n_rows), pa.int64())}

    for j in range(n_cat):
        card = int(rng.integers(2, 12))
        # integer category codes incl. a -1 "missing" sentinel, Zipf-ish weights
        values = np.array([-1, *range(card - 1)])
        weights = 1.0 / (1.0 + np.arange(card))
        weights /= weights.sum()
        col = rng.choice(values, size=n_rows, p=weights)
        mask = rng.random(n_rows) < 0.01
        cols[f"cat_{j}"] = pa.array(col, mask=mask)

    for j in range(n_num):
        loc, scale = rng.uniform(-5, 5), rng.uniform(0.5, 50)
        col = rng.normal(loc, scale, n_rows)
        if j % 4 == 0:  # heavy-tailed -> good candidate for signed_log1p
            col = np.exp(col / 10) * rng.choice([-1, 1], n_rows)
        mask = rng.random(n_rows) < 0.02
        cols[f"num_{j}"] = pa.array(col, mask=mask)

    cols["target"] = pa.array(rng.integers(0, 2, n_rows), pa.int8())
    return pa.table(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = build_table(args.rows, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    step = -(-table.num_rows // args.files)
    for i, start in enumerate(range(0, table.num_rows, step)):
        pq.write_table(table.slice(start, step), os.path.join(args.out, f"part-{i:03d}.parquet"))
    print(f"wrote {table.num_rows} rows x {table.num_columns} cols "
          f"to {args.files} files in {args.out}")
    cat_cols = [c for c in table.column_names if c.startswith("cat_")]
    num_cols = [c for c in table.column_names if c.startswith("num_")]
    print("categorical_columns =", cat_cols)
    print("numeric_columns =", num_cols)


if __name__ == "__main__":
    main()
