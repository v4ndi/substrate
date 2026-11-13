"""Run the whole comparison campaign, one training at a time.

Order matters only for the GPU: there is a single A100 on this host, so the
runs are sequential. Each run's exit code and wall time are recorded by
run_training.py; this script just drives them and stops on the first failure of
a run that the rest depends on.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training")
PYTHON = "/usr/bin/python3.11"

PLAN = [
    # Two seeds per side: the paired difference (same seed, both revisions) is
    # read against the spread a seed change makes inside one revision.
    "hillstrom_base_seed42",
    "hillstrom_new_seed42",
    "hillstrom_base_seed43",
    "hillstrom_new_seed43",
    "hillstrom_base_seed44",
    "hillstrom_new_seed44",
    "hillstrom_base_seed45",
    "hillstrom_new_seed45",
    "lenta_base_seed42",
    "lenta_new_seed42",
    "lenta_base_seed43",
    "lenta_new_seed43",
    "lenta_base_seed44",
    "lenta_new_seed44",
    "lenta_base_seed45",
    "lenta_new_seed45",
    # The weight-averaging path: the refactored branch crashes here, the same
    # branch with the evaluation-device fix does not.
    "hillstrom_base_swa_seed42",
    "hillstrom_new_swa_seed42",
    "hillstrom_fix_swa_seed42",
]


def main() -> None:
    only = sys.argv[1:]
    plan = [tag for tag in PLAN if not only or tag in only or tag.startswith(tuple(only))]
    results = []
    for tag in plan:
        started = time.time()
        code = subprocess.call([PYTHON, str(ROOT / "scripts" / "run_training.py"), tag])
        results.append((tag, code, round(time.time() - started, 1)))
        print(f"=== {tag}: exit={code} in {results[-1][2]}s", flush=True)
    print("\n=== campaign summary ===")
    for tag, code, seconds in results:
        print(f"{tag:26} exit={code} {seconds:>8.1f}s")
    if any(code for _, code, _ in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
