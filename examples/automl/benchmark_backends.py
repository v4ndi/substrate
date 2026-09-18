"""Train both backend families on the same data and print their metrics side by side.

    .venv/bin/python examples/automl/benchmark_backends.py --rows 20000

**What is being compared.** The same rows, the same split, the same metric --
and two pipelines, not two architectures. The boosting backend sees an external
embedding expanded into one scalar feature per coordinate; TabNN sees it as one
vector and fuses it after pooling. That difference is deliberate (a tree cannot
eat a vector, and a network should not have its embedding torn into columns and
standardised coordinate by coordinate), which makes the comparison honest about
outcomes and dishonest about anything narrower.

The numbers here come from synthetic data with a known signal. They say whether
the plumbing works end to end, not which family wins on real data.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from fmlib.automl import BinaryTask, BinaryTaskConfig

HIDDEN_WIDTH = 16


def write_split(directory: Path, rows: int, seed: int, files: int = 4) -> Path:
    """Write one split with a learnable signal and a hidden-state column."""
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(rows, HIDDEN_WIDTH)).astype(np.float32)
    balance = rng.normal(size=rows)
    tenure = rng.gamma(2.0, 1.0, size=rows)
    segment = rng.choice(["a", "b", "c"], rows)
    logit = (
        1.3 * balance
        - 0.6 * tenure
        + 0.8 * (segment == "a")
        + hidden[:, 0]
        + 0.5 * hidden[:, 1] * balance
    )
    frame = pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 1_000_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "segment": segment,
        "balance": balance,
        "tenure": tenure,
        "seq_hidden_state": [row.tolist() for row in hidden],
        "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
    }).with_columns(pl.col("seq_hidden_state").cast(pl.List(pl.Float32)))

    directory.mkdir(parents=True, exist_ok=True)
    step = max(1, rows // files)
    for index, start in enumerate(range(0, rows, step)):
        frame.slice(start, step).write_parquet(directory / f"part-{index:03d}.parquet")
    return directory


def run(backend: str, engine: str, root: Path, data: dict[str, Path], device: str):
    """Train one family and score the test split with it."""
    config = BinaryTaskConfig(
        env_type="local",
        backend=backend,
        engine=engine,
        device=device,
        target_column="target",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=["segment"],
        numerical_columns=["balance", "tenure"],
        hidden_state_columns=["seq_hidden_state"],
        hyperopt=False,
        model_params=(
            {"iterations": 200, "depth": 5}
            if backend == "boosting"
            else {"max_epochs": 8, "batch_size": 1024, "hidden_size": 64}
        ),
        verbose=False,
        output_dir=root / backend,
    )
    task = BinaryTask(config)

    started = time.perf_counter()
    training = task.train(data["train"], data["valid"])
    trained = time.perf_counter() - started

    prediction = task.predict(data["test"])
    evaluation = task.evaluate(data["test"], prediction)
    return {
        "backend": backend,
        "engine": engine,
        "validation": training.validation_metrics["global"],
        "test_roc_auc": (evaluation.metrics_raw or {}).get("roc_auc"),
        "features": len(training.feature_names),
        "seconds": trained,
    }


def main() -> None:
    """Run both families and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--out", default="outputs/benchmark")
    parser.add_argument(
        "--keep", action="store_true", help="Do not delete --out when finished"
    )
    arguments = parser.parse_args()

    root = Path(arguments.out)
    shutil.rmtree(root, ignore_errors=True)
    data = {
        name: write_split(root / "data" / name, rows, seed)
        for name, rows, seed in (
            ("train", arguments.rows, 1),
            ("valid", arguments.rows // 4, 2),
            ("test", arguments.rows // 4, 3),
        )
    }

    results = [
        run("boosting", "catboost", root, data, arguments.device),
        run("tabnn", "tabular_transformer", root, data, arguments.device),
    ]

    print()
    print(
        f"{'backend':<10} {'engine':<20} {'valid':>9} {'test ROC AUC':>13} {'features':>9} {'seconds':>8}"
    )
    for item in results:
        print(
            f"{item['backend']:<10} {item['engine']:<20} {item['validation']:>9.4f} "
            f"{item['test_roc_auc']:>13.4f} {item['features']:>9d} {item['seconds']:>8.1f}"
        )
    print()
    print(
        "Feature counts differ on purpose: boosting sees the "
        f"{HIDDEN_WIDTH}-dimensional embedding as {HIDDEN_WIDTH} scalar columns, "
        "TabNN as one vector. Two pipelines are being compared, not two "
        "architectures."
    )
    if not arguments.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
