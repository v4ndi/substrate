"""Build the four task datasets from open data, in one layout.

Four settings — uplift, response, regression and multi-class classification —
over three public datasets, written so that the four training configs differ
only where the tasks differ.

    python examples/tabular_tasks/prepare_data.py --task all --scale smoke
    python examples/tabular_tasks/prepare_data.py --task regression --scale full

Every task produces the same columns:

======================  ==========================================
``epk_id``              record id, carried into the submit file
``target``              what is predicted
``treatment``           ``1 = control`` — uplift only
``group``               population slice; every metric is per group
``split_type``          ``calib`` / ``test`` inside the valid set
``cat_features``        packed categorical ids, one embedding table
``num_features``        packed standardised numerics
======================  ==========================================

``group`` and ``split_type`` are not free names: the metrics declare exactly
those keys in ``required_inputs``, and a column named anything else is dropped
before it reaches them.

The preprocessor is fitted on train only, and the same object transforms valid
— otherwise the training statistics leak and validation flatters the model.
The three numbers the model config needs are printed at the end and written to
``dims.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, os.getcwd())

from avatar.preprocessing.local import TabularPreprocessor

HERE = pathlib.Path(__file__).resolve().parent

#: Fraction of records used for training; the rest becomes valid, and valid is
#: then halved into the calibration and held-out slices.
TRAIN_FRACTION = 0.8
SPLIT_SEED = 0

#: How many rows a smoke build keeps. Small enough to train through in a couple
#: of minutes on a laptop, big enough that the top five percent of a group is
#: still a whole number of records.
SMOKE_ROWS = 20_000

IDENTITY = ["epk_id", "target", "group", "split_type"]


# ---------------------------------------------------------------------------
# raw data
# ---------------------------------------------------------------------------


def hillstrom(cache: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    """64k customers of a mail campaign, with a control arm.

    The visit flag is the conversion; ``segment`` names which of the two emails
    the customer got, or that they got none, which is the control arm.
    """
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col="visit", data_home=str(cache))
    frame = bunch.data.copy()
    frame["target"] = bunch.target.to_numpy()
    frame["treatment"] = (bunch.treatment == "No E-Mail").astype("int64").to_numpy()
    frame["group"] = pd.factorize(frame["channel"], sort=True)[0].astype("int64")
    return frame, {
        "categorical": [
            "history_segment",
            "zip_code",
            "channel",
            "mens",
            "womens",
            "newbie",
        ],
        "group_source": "channel",
    }


def lenta(cache: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    """687k retail customers, 193 features, an SMS campaign with a control arm."""
    from sklift.datasets import fetch_lenta

    bunch = fetch_lenta(data_home=str(cache))
    frame = bunch.data.copy()
    frame["target"] = bunch.target.to_numpy()
    # sklift calls the treatment column ``group`` — the very name the metrics
    # reserve for the campaign group. Renaming it here is not cosmetic: left
    # alone, everything would assemble and score the wrong thing.
    frame["treatment"] = (bunch.treatment == "control").astype("int64").to_numpy()
    frame["group"] = frame["main_format"].astype("int64")
    return frame, {
        "categorical": ["gender", "main_format"],
        "group_source": "main_format",
    }


def covertype(cache: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    """581k forest plots, seven cover types — the multi-class target.

    The 44 wilderness/soil indicators are binary, so they are declared
    categorical and share the embedding table with everything else; the ten
    measurements stay numeric.
    """
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype(as_frame=True, data_home=str(cache))
    frame = bunch.frame.copy()
    frame["target"] = frame.pop("Cover_Type").astype("int64") - 1  # 1..7 -> 0..6
    areas = [name for name in frame.columns if name.startswith("Wilderness_Area")]
    frame["group"] = frame[areas].to_numpy().argmax(axis=1).astype("int64")
    categorical = [
        name
        for name in frame.columns
        if name not in ("target", "group") and frame[name].nunique() <= 2
    ]
    return frame, {"categorical": categorical, "group_source": "Wilderness_Area"}


# ---------------------------------------------------------------------------
# task specifications
# ---------------------------------------------------------------------------


def regression_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Turn Lenta into a regression set: predict a customer's yearly spend.

    ``spend`` from Hillstrom looks like the obvious choice and is not: 99.1% of
    it is zero, leaving 578 non-zero records out of 64 000. This column is
    dense — 82% non-zero — and heavy-tailed, hence ``log1p``.

    Every sibling column of the target has to go. They are aggregates over the
    same category and the same year, so left in they let the model read the
    answer off its own input.
    """
    source = "sale_sum_12m_g24"
    leaking = [name for name in frame.columns if "g24" in name]
    frame = frame.copy()
    frame["target"] = np.log1p(frame[source].fillna(0.0)).astype("float32")
    return frame.drop(columns=leaking), leaking


TASKS = {
    "uplift": {
        "dataset": hillstrom,
        "task_type": "uplift",
        "keeps_treatment": True,
        "doc": "S-Learner over a mail campaign: P(visit | treated) - P(visit | control)",
    },
    "response": {
        "dataset": hillstrom,
        "task_type": "binary",
        "keeps_treatment": False,
        "doc": "the same population, scored on P(visit) alone",
    },
    "regression": {
        "dataset": lenta,
        "task_type": "regression",
        "keeps_treatment": False,
        "transform": regression_target,
        "doc": "log1p of a customer's yearly spend in one category",
    },
    "multiclass": {
        "dataset": covertype,
        "task_type": "multiclass",
        "keeps_treatment": False,
        "doc": "seven forest cover types from ten measurements and 44 indicators",
    },
}


# ---------------------------------------------------------------------------
# splitting and writing
# ---------------------------------------------------------------------------


def stratum(frame: pd.DataFrame, task_type: str) -> np.ndarray:
    """The key each split is balanced over.

    A rare class or a rare arm must land on both sides of every split, or the
    metric for it is undefined on one of them.
    """
    parts = [frame["group"].astype(str)]
    if "treatment" in frame:
        parts.append(frame["treatment"].astype(str))
    if task_type == "regression":
        # Deciles of the target: a continuous column has no classes to balance.
        parts.append(
            pd.qcut(frame["target"], 10, labels=False, duplicates="drop").astype(str)
        )
    else:
        parts.append(frame["target"].astype(str))
    return parts[0].str.cat(parts[1:], sep="_").to_numpy()


def split_mask(strata: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A stratified boolean mask keeping ``fraction`` of every stratum."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(strata), dtype=bool)
    for value in np.unique(strata):
        positions = np.flatnonzero(strata == value)
        keep[positions[rng.random(len(positions)) < fraction]] = True
    return keep


def write_shards(table: pa.Table, out_dir: pathlib.Path, shards: int) -> int:
    """Write one table as several parquet files, so file sharding has work."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = table.num_rows
    step = (rows + shards - 1) // shards
    written = 0
    for index in range(shards):
        start = index * step
        if start >= rows:
            break
        pq.write_table(
            table.slice(start, min(step, rows - start)),
            out_dir / f"part-{index:03d}.parquet",
        )
        written += 1
    return written


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


def build(task: str, scale: str, out_root: pathlib.Path, cache: pathlib.Path) -> dict:
    """Load, split, preprocess and shard one task's data.

    Returns the dimensions its training config needs.
    """
    spec = TASKS[task]
    print(f"\n=== {task} ({scale}) — {spec['doc']}")

    frame, columns = spec["dataset"](cache)
    if "transform" in spec:
        frame, dropped = spec["transform"](frame)
        print(
            f"[{task}] dropped {len(dropped)} columns that leak the target: {dropped}"
        )

    if not spec["keeps_treatment"]:
        frame = frame.drop(columns=[c for c in ("treatment",) if c in frame])

    if scale == "smoke" and len(frame) > SMOKE_ROWS:
        keep = split_mask(
            stratum(frame, spec["task_type"]), SMOKE_ROWS / len(frame), SPLIT_SEED
        )
        frame = frame[keep].reset_index(drop=True)
        print(f"[{task}] smoke build: sampled down to {len(frame)} rows")

    frame = frame.reset_index(drop=True)
    frame["epk_id"] = np.arange(len(frame), dtype="int64")

    identity = list(IDENTITY)
    if spec["keeps_treatment"]:
        identity.append("treatment")
    cat_cols = [c for c in columns["categorical"] if c in frame.columns]
    num_cols = [c for c in frame.columns if c not in cat_cols and c not in identity]

    print(
        f"[{task}] {len(frame)} rows, {len(num_cols)} numeric, "
        f"{len(cat_cols)} categorical, group from {columns['group_source']} "
        f"({frame['group'].nunique()} values)"
    )
    if spec["task_type"] in ("uplift", "binary", "multiclass"):
        counts = frame["target"].value_counts().sort_index().to_dict()
        print(f"[{task}] target: {counts}")
    else:
        print(
            f"[{task}] target: mean={frame['target'].mean():.3f} "
            f"zeros={float((frame['target'] == 0).mean()):.3f} "
            f"max={frame['target'].max():.3f}"
        )
    if "treatment" in frame:
        treated = frame[frame["treatment"] == 0]["target"].mean()
        control = frame[frame["treatment"] == 1]["target"].mean()
        print(f"[{task}] conversion: treated={treated:.4f} control={control:.4f}")

    # --- split ----------------------------------------------------------
    strata = stratum(frame, spec["task_type"])
    is_train = split_mask(strata, TRAIN_FRACTION, SPLIT_SEED)
    # The held-out half of valid is where calibrated uplift numbers are allowed
    # to come from; the other half is what the calibrators are fitted on.
    is_calib = split_mask(strata, 0.5, SPLIT_SEED + 1)
    frame["split_type"] = np.where(is_calib, "calib", "test")
    frame.loc[is_train, "split_type"] = "calib"  # unused on train, kept uniform
    print(
        f"[{task}] split: train={int(is_train.sum())} valid={int((~is_train).sum())} "
        f"(calib={int((~is_train & is_calib).sum())} "
        f"test={int((~is_train & ~is_calib).sum())})"
    )

    work = out_root / f"{task}_{scale}" / "_raw"
    if work.exists():
        shutil.rmtree(work)
    for name, part in (("train", frame[is_train]), ("valid", frame[~is_train])):
        (work / name).mkdir(parents=True)
        part.to_parquet(work / name / "part.parquet", index=False)

    # --- preprocessing --------------------------------------------------
    preprocessor = TabularPreprocessor(
        categorical_columns=cat_cols,
        numeric_columns=num_cols,
        spec_tokens={"pad": 0},
    )
    preprocessor.fit(str(work / "train"))
    print(f"[{task}] vocab_size={preprocessor.vocab_size}")

    shards = 4 if scale == "smoke" else 16
    out_dir = out_root / f"{task}_{scale}"
    rows = {}
    for name in ("train", "valid"):
        table = preprocessor.transform(str(work / name), identity_cols=identity)
        keep = [*identity, "cat_features", "num_features"]
        table = table.select([c for c in keep if c in table.schema.names])
        files = write_shards(table, out_dir / name, shards)
        rows[name] = table.num_rows
        print(f"[{task}] {name}: {table.num_rows} rows -> {files} files")
    shutil.rmtree(work)

    with (out_dir / "preprocessor.yaml").open("w") as handle:
        yaml.safe_dump(preprocessor.dump(), handle)

    dims = {
        "task": task,
        "scale": scale,
        "vocab_size": int(preprocessor.vocab_size),
        "num_numerical_features": len(num_cols),
        "num_features": len(cat_cols) + len(num_cols),
        "n_groups": int(frame["group"].nunique()),
        "num_classes": int(frame["target"].nunique())
        if spec["task_type"] == "multiclass"
        else 1,
        "train_path": str(out_dir / "train"),
        "valid_path": str(out_dir / "valid"),
        "train_rows": rows["train"],
        "valid_rows": rows["valid"],
    }
    with (out_dir / "dims.json").open("w") as handle:
        json.dump(dims, handle, indent=2)

    print(f"[{task}] config needs:")
    print(f"    embedding.vocab_size:               {dims['vocab_size']}")
    print(f"    embedding.num_numerical_features:   {dims['num_numerical_features']}")
    print(
        f"    aggregation_config.num_features:    {dims['num_features']}"
        " (+1 per extra token: treatment, group)"
    )
    return dims


def main() -> None:
    """Build the tasks named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="all", choices=[*TASKS, "all"], help="which set to build"
    )
    parser.add_argument(
        "--scale",
        default="smoke",
        choices=["smoke", "full"],
        help="smoke samples ~20k rows; full uses everything",
    )
    parser.add_argument(
        "--out", default=str(HERE / "data"), help="where the parquet goes"
    )
    parser.add_argument(
        "--cache",
        default=str(HERE / "data" / "raw"),
        help="download cache for the source datasets",
    )
    args = parser.parse_args()

    out_root = pathlib.Path(args.out)
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SKLIFT_DATA_DIR", str(cache))

    tasks = list(TASKS) if args.task == "all" else [args.task]
    for task in tasks:
        build(task, args.scale, out_root, cache)
    print("\ndone")


if __name__ == "__main__":
    main()
