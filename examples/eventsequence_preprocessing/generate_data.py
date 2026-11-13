"""Generate a synthetic event-log dataset for the sequence preprocessing example.

One row per event: ``epk_id``, a ``timestamps`` column, an ``event_ids`` event-type
column, plus categorical and numeric per-event attributes. Rows are written
shuffled and split across several parquet files. ``EventSequencePreprocessor``
turns this into one row per ``epk_id`` with each attribute collected into a
time-ordered list.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_MCC = ["5411", "5812", "5999", "4814", "6011", "5541", "5912"]
_DIR = ["in", "out"]
_EPOCH_2024 = np.datetime64("2024-01-01T00:00:00", "us")


def build_table(n_users: int = 6_000, seed: int = 0) -> pa.Table:
    """Build a synthetic event log: one row per event, 5-60 events per user."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(5, 60, size=n_users)
    n = int(counts.sum())

    epk_id = np.repeat(np.arange(1, n_users + 1, dtype=np.int32), counts)
    minute_offset = rng.integers(0, 200_000, size=n).astype("timedelta64[m]")
    timestamps = (_EPOCH_2024 + minute_offset).astype("datetime64[us]")

    direction = rng.choice(_DIR, size=n).astype(object)
    direction[rng.random(n) < 0.04] = None

    table = pa.table({
        "epk_id": pa.array(epk_id),
        "timestamps": pa.array(timestamps),
        "event_ids": pa.array(rng.integers(0, 4, size=n).astype(np.int32)),
        "mcc": pa.array(rng.choice(_MCC, size=n)),
        "direction": pa.array(direction),
        "amount": pa.array(np.round(rng.uniform(1, 20_000, size=n), 2)),
    })
    # shuffle rows so the streaming / sort code paths are actually exercised
    perm = rng.permutation(n)
    return table.take(pa.array(perm))


def main() -> None:
    """Write the synthetic event log out as partitioned parquet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--users", type=int, default=6_000)
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = build_table(args.users, args.seed)
    os.makedirs(args.out, exist_ok=True)
    step = -(-table.num_rows // args.files)
    for i, start in enumerate(range(0, table.num_rows, step)):
        pq.write_table(
            table.slice(start, step), os.path.join(args.out, f"part-{i:03d}.parquet")
        )
    print(
        f"wrote {table.num_rows} events for {args.users} users "
        f"to {args.files} files in {args.out}"
    )


if __name__ == "__main__":
    main()
