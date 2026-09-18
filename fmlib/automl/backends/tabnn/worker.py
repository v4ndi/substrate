"""Rank-side entrypoint of a trial.

    python -m torch.distributed.run --nproc_per_node=<gpus> \
        -m fmlib.automl.backends.tabnn.worker --spec <run_spec.json>

Every rank runs the same function the in-process runner runs. The spec is the
same document, the assembly is the same assembly, and only rank 0 writes the
result -- which is what makes "the driver reads result.json" the one collection
path, whatever launched the trial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import TrialSpec, run_trial


def main(argv: list[str] | None = None) -> int:
    """Run one trial on this rank.

    Args:
        argv: Command line, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` when the trial completed, ``1`` when it failed. The exit code is
        what a launcher sees; the reason is in ``result.json``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to run_spec.json")
    arguments = parser.parse_args(argv)

    spec = TrialSpec.from_dict(
        json.loads(Path(arguments.spec).read_text(encoding="utf-8"))
    )
    result = run_trial(spec)
    return 0 if result.state == "COMPLETE" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through torchrun
    raise SystemExit(main())
