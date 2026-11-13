"""Turn a raw uplift dataset into the parquet layout the pilot config expects.

Output columns follow ``experiments/sbercampaign_pilot/.../td.yaml``:

* ``target_attr_1`` -- binary conversion flag
* ``target_attr_2`` -- campaign group (the "channel" of the pilot config)
* ``target_attr_3`` -- treatment flag stored the way the pilot data stores it,
  ``1 = control``, so the config keeps ``inverse_treatment: True``
* ``cat_features`` / ``num_features`` -- packed by ``TabularPreprocessor``

The preprocessor is fitted on the train split only and applied to both splits.
Both revisions under comparison read exactly these files, so any metric
difference between them cannot come from the data.

Run with cwd set to the worktree whose ``avatar`` should do the preprocessing.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys

# Run the avatar of the *current directory*: this script lives outside any
# checkout, so sys.path[0] would otherwise be its own directory and the import
# would fall through to whatever editable install happens to be registered.
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

import avatar  # noqa: E402
from avatar.preprocessing.local import TabularPreprocessor  # noqa: E402

assert avatar.__file__.startswith(os.getcwd()), (
    f"avatar came from {avatar.__file__}, not from {os.getcwd()}"
)

ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training")
DATA = ROOT / "data"

SPECS = {
    "lenta": {
        "categorical": ["gender", "main_format"],
        "group_column": "main_format",
        "group_map": None,  # taken from the column's own values
        "shards": 16,
    },
    "hillstrom": {
        "categorical": [
            "history_segment",
            "zip_code",
            "channel",
            "mens",
            "womens",
            "newbie",
        ],
        "group_column": "channel",
        "group_map": None,
        "shards": 8,
    },
}

TRAIN_FRACTION = 0.8
SPLIT_SEED = 0


def write_shards(table: pa.Table, out_dir: pathlib.Path, shards: int) -> None:
    """Write one table as several parquet files, so file-level sharding has work."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = table.num_rows
    step = (rows + shards - 1) // shards
    for index in range(shards):
        start = index * step
        if start >= rows:
            break
        pq.write_table(
            table.slice(start, min(step, rows - start)),
            out_dir / f"part-{index:03d}.parquet",
        )


def main(name: str) -> None:
    spec = SPECS[name]
    print(f"avatar from: {avatar.__file__}")
    frame = pd.read_parquet(DATA / f"{name}_raw.parquet")

    # --- targets ---------------------------------------------------------
    target = frame.pop("__target__").astype("int64")
    raw_treatment = frame.pop("__treatment__")
    control_names = {"control", "No E-Mail"}
    is_control = raw_treatment.isin(control_names).astype("int64")

    group_column = spec["group_column"]
    codes, uniques = pd.factorize(frame[group_column], sort=True)
    group = pd.Series(codes, index=frame.index).astype("int64")
    n_real_groups = int(len(uniques))
    # Every record keeps its real group, control included: metrics are reported
    # per group and need both arms inside each one. ``exchange_treatment_group``
    # relabels control to the reserved id ``n_groups - 1`` inside the model, so
    # the config declares one group more than the data uses.
    n_groups = n_real_groups + 1

    frame["target_attr_1"] = target.values
    frame["target_attr_2"] = group.values
    frame["target_attr_3"] = is_control.values

    identity = ["target_attr_1", "target_attr_2", "target_attr_3"]
    cat_cols = list(spec["categorical"])
    num_cols = [c for c in frame.columns if c not in cat_cols and c not in identity]

    print(f"[{name}] rows={len(frame)} numeric={len(num_cols)} categorical={len(cat_cols)}")
    print(f"[{name}] group column={group_column} values={list(uniques)} -> n_groups={n_groups} (last id = control)")
    print(f"[{name}] treated={int((is_control == 0).sum())} control={int(is_control.sum())}")
    print(f"[{name}] conversion: treated={target[is_control == 0].mean():.4f} control={target[is_control == 1].mean():.4f}")

    # --- split -----------------------------------------------------------
    strata = is_control.astype(str) + "_" + target.astype(str) + "_" + group.astype(str)
    rng = np.random.default_rng(SPLIT_SEED)
    is_train = np.zeros(len(frame), dtype=bool)
    for _, index in frame.groupby(strata.values).groups.items():
        positions = frame.index.get_indexer(index)
        picked = rng.random(len(positions)) < TRAIN_FRACTION
        is_train[positions[picked]] = True
    print(f"[{name}] split: train={int(is_train.sum())} valid={int((~is_train).sum())}")

    work = DATA / f"{name}_split"
    if work.exists():
        shutil.rmtree(work)
    (work / "train").mkdir(parents=True)
    (work / "valid").mkdir(parents=True)
    frame[is_train].to_parquet(work / "train" / "part.parquet", index=False)
    frame[~is_train].to_parquet(work / "valid" / "part.parquet", index=False)

    # --- preprocessing ---------------------------------------------------
    preprocessor = TabularPreprocessor(
        categorical_columns=cat_cols,
        numeric_columns=num_cols,
        spec_tokens={"pad": 0},
    )
    preprocessor.fit(str(work / "train"))
    print(f"[{name}] vocab_size={preprocessor.vocab_size} offset_map={preprocessor.offset_map}")

    out_root = DATA / f"{name}_processed"
    for split in ("train", "valid"):
        table = preprocessor.transform(str(work / split), identity_cols=identity)
        keep = [c for c in table.schema.names if c in {*identity, "cat_features", "num_features"}]
        table = table.select(keep)
        write_shards(table, out_root / split, spec["shards"])
        files = sorted((out_root / split).glob("*.parquet"))
        print(f"[{name}] {split}: {table.num_rows} rows -> {len(files)} files, columns={table.schema.names}")

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with (artifacts / f"{name}_preprocessor.yaml").open("w") as handle:
        yaml.safe_dump(preprocessor.dump(), handle)

    dims = {
        "dataset": name,
        "num_numerical_features": len(num_cols),
        "num_categorical_features": len(cat_cols),
        "vocab_size": int(preprocessor.vocab_size),
        "n_groups": n_groups,
        "train_path": str(out_root / "train"),
        "valid_path": str(out_root / "valid"),
        "train_rows": int(is_train.sum()),
        "valid_rows": int((~is_train).sum()),
        "shards": spec["shards"],
    }
    with (artifacts / f"{name}_dims.json").open("w") as handle:
        json.dump(dims, handle, indent=2)
    print(f"[{name}] dims: {dims}")


if __name__ == "__main__":
    main(sys.argv[1])
