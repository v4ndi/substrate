"""Run a trial's own spec through the real ``fmlib.train`` entrypoint.

Shared by the single-rank and multi-rank parity checks. The point of both is
the same: AutoML claims to reimplement nothing, and the way to test that claim
is to hand the document AutoML produced to the entrypoint a person would have
used, then compare what comes out.

Exactly two things are edited in between -- where to write, and the ``mlflow``
names ``fmlib.train.__main__`` reads to build its own checkpoint path. Nothing
touching the model, the data, the optimizer, the schedule or the seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]


def handwritten_config(spec_path: Path, target: Path) -> Path:
    """Turn a trial's ``run_spec.json`` into a config Hydra can be given."""
    config = json.loads(Path(spec_path).read_text(encoding="utf-8"))["config"]
    if not isinstance(config, dict):
        msg = (
            f"run_spec.json holds a {type(config).__name__}: a rank reading this "
            "file back would get a string instead of a run"
        )
        raise TypeError(msg)
    target.mkdir(parents=True, exist_ok=True)
    for entry in config["callbacks"]:
        if entry["_target_"].endswith("CheckpointCallback"):
            entry["directory"] = str(target / "checkpoints")
    config["mlflow"] = {"experiment_name": "train_parity", "run_name": "debug"}
    path = target / "cfg.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    return path


def cpu_ranks_env() -> dict[str, str]:
    """Force gloo on CPU: these comparisons are about arithmetic, not hardware."""
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join([
            str(REPO_ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep),
    }


def same_threads_env() -> dict[str, str]:
    """The parent's thread count, passed on unchanged.

    Bit-identity between two runs of the same config holds *at the same thread
    count* and not otherwise: CPU reductions in torch are split across threads,
    and floating-point addition is not associative, so eight threads and one
    thread produce answers that differ in the last bits. Small as that is, it
    is enough to move which validation pass early stopping calls best -- the
    first version of this helper pinned the child to one thread and the two
    runs stopped at step 8 and step 16.

    So the single-rank comparison inherits the parent's threading, and the
    multi-rank one pins *both* sides to one thread through `cpu_ranks_env`.
    """
    environment = {
        "PYTHONPATH": os.pathsep.join([
            str(REPO_ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep),
    }
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def run_handwritten(config_path: Path, *, ranks: int = 1) -> None:
    """Run ``fmlib.train`` on that config, under torchrun when ranks > 1."""
    environment = {
        **os.environ,
        **(same_threads_env() if ranks == 1 else cpu_ranks_env()),
    }
    if ranks == 1:
        command = [sys.executable, "-m", "fmlib.train"]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ranks}",
            "-m",
            "fmlib.train",
        ]
    command += [
        f"--config-dir={config_path.parent}",
        f"--config-name={config_path.stem}",
    ]
    completed = subprocess.run(
        command,
        cwd=config_path.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if completed.returncode != 0:
        msg = (
            f"fmlib.train on {ranks} rank(s) exited {completed.returncode}\n"
            f"{completed.stderr[-4000:]}"
        )
        raise AssertionError(msg)


def only_checkpoint(root: Path) -> Path:
    """The single checkpoint directory under ``root``, or a readable failure."""
    steps = sorted(path.parent for path in Path(root).rglob("checkpoint.pt"))
    if len(steps) != 1:
        msg = f"expected one checkpoint under {root}, found {steps}"
        raise AssertionError(msg)
    return steps[0]


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
