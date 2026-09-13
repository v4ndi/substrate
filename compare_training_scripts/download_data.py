"""Download the uplift datasets used for the refactor comparison.

Lenta is the main measurement (687k rows, 193 features); Hillstrom is the
smoke dataset (64k rows) used to shake out the wiring before the long runs.
Both come from scikit-uplift and are cached under DATA_ROOT.
"""

import os
import pathlib
import sys

DATA_ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training/data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SKLIFT_DATA_DIR", str(DATA_ROOT / "sklift"))

from sklift.datasets import fetch_hillstrom, fetch_lenta  # noqa: E402


def describe(name, bunch):
    data, target, treatment = bunch.data, bunch.target, bunch.treatment
    print(f"[{name}] rows={len(data)} cols={data.shape[1]}")
    print(f"[{name}] target={target.name} rate={target.mean():.4f} values={sorted(target.unique())[:5]}")
    print(f"[{name}] treatment={treatment.name} counts={treatment.value_counts().to_dict()}")
    print(f"[{name}] dtypes={data.dtypes.value_counts().to_dict()}")
    out = DATA_ROOT / f"{name}_raw.parquet"
    frame = data.copy()
    frame["__target__"] = target.values
    frame["__treatment__"] = treatment.values
    frame.to_parquet(out, index=False)
    print(f"[{name}] saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return frame


if __name__ == "__main__":
    which = sys.argv[1:] or ["hillstrom", "lenta"]
    if "hillstrom" in which:
        describe("hillstrom", fetch_hillstrom(data_home=str(DATA_ROOT / "sklift")))
    if "lenta" in which:
        describe("lenta", fetch_lenta(data_home=str(DATA_ROOT / "sklift")))
    print("done")
