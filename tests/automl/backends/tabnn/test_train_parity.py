"""AutoML must train the *same* run a hand-written fmlib config trains.

``run_trial`` deliberately reimplements nothing: it builds the model, the
loaders, the optimizer, the schedule and the callbacks through the same
``instantiate`` / ``init_*`` layer as ``python -m fmlib.train``. That is an
argument about code, and an argument about code is not evidence about numbers.

This is the evidence. One trial is run through the public AutoML API; the
``run_spec.json`` it leaves behind is handed to the real ``fmlib.train``
entrypoint as a Hydra config, changed in exactly two places -- somewhere else
to write, and the ``mlflow`` names the entrypoint reads to build its own
checkpoint path. Neither touches the model, the data, the optimizer, the
schedule or the seed.

Both runs are then compared where it cannot be argued with: the saved
checkpoint, byte for byte. ``checkpoint.pt`` carries the model, the optimizer
state, the scheduler state, the RNG state and the trainer state, so equal
bytes mean the two paths took the same steps in the same order and stopped in
the same place -- not merely that they landed on a similar score.

Marked ``slow``: it trains twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from automl.backends.tabnn.handwritten import (
    digest,
    handwritten_config,
    only_checkpoint,
    run_handwritten,
)
from fmlib.automl import BinaryTask, BinaryTaskConfig

pytestmark = pytest.mark.slow

PARAMS = {
    "max_epochs": 2,
    "batch_size": 256,
    "hidden_size": 32,
    "num_layers": 2,
    "num_heads": 4,
    "num_workers": 0,
    "evaluations_per_epoch": 1,
    "patience": 2,
}


def _write(directory: Path, rows: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(rows, 4)).astype(np.float32)
    balance = rng.normal(size=rows)
    tenure = rng.gamma(2.0, 1.0, size=rows)
    logit = 1.5 * balance - 0.4 * tenure + hidden[:, 0]
    frame = pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "segment": rng.choice(["a", "b", "c"], rows),
        "balance": balance,
        "tenure": tenure,
        "seq_hidden_state": [row.tolist() for row in hidden],
        "y_binary": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
    }).with_columns(pl.col("seq_hidden_state").cast(pl.List(pl.Float32)))
    directory.mkdir(parents=True, exist_ok=True)
    half = rows // 2
    frame.head(half).write_parquet(directory / "part-0.parquet")
    frame.tail(rows - half).write_parquet(directory / "part-1.parquet")
    return directory


@pytest.fixture
def data(tmp_path) -> dict[str, Path]:
    return {
        "train": _write(tmp_path / "data" / "train", 2000, 1),
        "valid": _write(tmp_path / "data" / "valid", 800, 2),
    }


def _run_automl(output_dir: Path, data: dict[str, Path]) -> tuple[float, Path]:
    """Train one trial through the public API; return its score and its spec."""
    task = BinaryTask(
        BinaryTaskConfig(
            output_dir=output_dir,
            env_type="local",
            device="cpu",
            backend="tabnn",
            engine="tabular_transformer",
            client_id_column="epk_id",
            date_column="report_month",
            group_column=None,
            target_column="y_binary",
            categorical_columns=["segment"],
            numerical_columns=["balance", "tenure"],
            hidden_state_columns=["seq_hidden_state"],
            hyperopt=False,
            random_state=42,
            verbose=False,
            model_params=dict(PARAMS),
        )
    )
    result = task.train(data["train"], data["valid"])
    specs = sorted(output_dir.rglob("run_spec.json"))
    assert len(specs) == 1, f"expected one trial, found {len(specs)}"
    return float(result.validation_metrics["global"]), specs[0]


def test_automl_trains_the_same_run_as_a_handwritten_config(tmp_path, data):
    automl_dir = tmp_path / "automl"
    score, spec_path = _run_automl(automl_dir, data)

    handwritten_dir = tmp_path / "handwritten"
    run_handwritten(handwritten_config(spec_path, handwritten_dir))

    ours = only_checkpoint(automl_dir)
    theirs = only_checkpoint(handwritten_dir)
    # Same step, or the two runs did not even see the same number of batches.
    assert ours.name == theirs.name

    for name in ("model.bin", "checkpoint.pt"):
        assert digest(ours / name) == digest(theirs / name), (
            f"{name} differs between the two paths"
        )

    # Byte equality already implies this; asserted separately so a failure says
    # *what* drifted rather than only that some bytes did.
    left = torch.load(ours / "model.bin", map_location="cpu", weights_only=True)
    right = torch.load(theirs / "model.bin", map_location="cpu", weights_only=True)
    assert list(left) == list(right)
    for key in left:
        assert torch.equal(left[key], right[key]), f"weights differ at {key}"

    state = torch.load(ours / "checkpoint.pt", map_location="cpu", weights_only=False)
    other = torch.load(theirs / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert dict(state["state"]) == dict(other["state"])
    # And the number AutoML reported to the search is the one the loop recorded.
    assert state["state"]["best_metric"] == pytest.approx(score)
