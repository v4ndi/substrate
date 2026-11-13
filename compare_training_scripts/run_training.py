"""Run one training, from the worktree the config's side belongs to.

Usage: run_training.py <tag> [hydra overrides...]

The tag names the config in ``configs/`` and decides the worktree: ``*_base_*``
runs on 7c96504 (before the refactor), ``*_new_*`` on
``refactor/data-sharding-hdfs``. The runner checks that ``avatar`` really
resolves inside that worktree before starting — the machine has an editable
install pointing at an unrelated checkout, and a silent fallback to it would
invalidate the whole comparison.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training")
WORKTREES = pathlib.Path("/home/jovyan/rusakov/repos/avatar_fm/.claude/worktrees")
PYTHON = "/usr/bin/python3.11"


def worktree_for(tag: str) -> pathlib.Path:
    if "_base_" in tag:
        return WORKTREES / "cmp-base"
    if "_new_" in tag:
        return WORKTREES / "cmp-new"
    if "_fix_" in tag:
        return WORKTREES / "compare-training"
    raise SystemExit(f"cannot tell which revision {tag!r} belongs to")


def main() -> None:
    tag = sys.argv[1]
    overrides = sys.argv[2:]
    worktree = worktree_for(tag)

    probe = subprocess.run(
        [PYTHON, "-c", "import avatar, sys; print(avatar.__file__)"],
        cwd=worktree,
        capture_output=True,
        text=True,
    )
    resolved = probe.stdout.strip()
    if not resolved.startswith(str(worktree)):
        raise SystemExit(f"avatar resolves to {resolved!r}, not to {worktree}")
    print(f"[{tag}] avatar: {resolved}")

    work = ROOT / "work" / tag
    work.mkdir(parents=True, exist_ok=True)
    # The pre-refactor trainer refuses to start when the checkpoint directory of
    # this experiment/run already exists, so a rerun needs a clean slate.
    shutil.rmtree(work / "best_models", ignore_errors=True)
    log_path = ROOT / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        PYTHON,
        str(ROOT / "scripts" / "entry.py"),
        f"--config-dir={ROOT / 'configs'}",
        f"--config-name={tag}",
        f"hydra.run.dir={work / 'hydra'}",
        *overrides,
    ]
    print(f"[{tag}] cwd={worktree}")
    print(f"[{tag}] cmd={' '.join(argv)}")

    env = dict(os.environ)
    env["HYDRA_FULL_ERROR"] = "1"
    # Same writable inductor cache for both sides (see scripts/entry.py).
    cache = ROOT / "work" / "inductor_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    # The trainer chdir's into root_dir, so a relative sys.path[0] would stop
    # pointing at the checkout; pin the import path to this worktree.
    env["PYTHONPATH"] = str(worktree)

    started = time.time()
    with log_path.open("w") as handle:
        handle.write(f"# cwd={worktree}\n# avatar={resolved}\n# cmd={' '.join(argv)}\n")
        handle.flush()
        code = subprocess.call(
            argv, cwd=worktree, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - started

    record = {
        "tag": tag,
        "revision_dir": str(worktree),
        "avatar": resolved,
        "overrides": overrides,
        "exit_code": code,
        "seconds": round(elapsed, 1),
        "log": str(log_path),
    }
    (ROOT / "artifacts" / f"run_{tag}.json").write_text(json.dumps(record, indent=2))
    print(f"[{tag}] exit={code} in {elapsed:.1f}s -> {log_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()
