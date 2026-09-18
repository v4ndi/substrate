"""Launch ``fmlib.train`` for either revision, from the worktree that is cwd.

Two things this does that ``python -m fmlib.train`` does not:

* pins ``sys.path[0]`` to the current directory, because the trainer chdir's
  into ``root_dir`` and there is an editable ``fmlib`` install on this machine
  pointing at an unrelated checkout;
* imports ``torch._dynamo`` first. The pre-refactor ``fmlib/train.py`` sets
  ``TORCHINDUCTOR_CACHE_DIR`` to a hard-coded ``/home/datalab/...`` path at
  import time, and torch creates that directory when ``torch._dynamo`` is first
  imported — which fails on this host. Importing dynamo before the trainer
  resolves the cache directory from the (writable) inherited environment
  instead. Neither revision calls ``torch.compile`` with this config, so
  nothing else touches the inductor cache.

The same entry point is used for both sides, so the launch path is not itself a
difference between them.
"""

from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, os.getcwd())

import torch._dynamo  # noqa: F401,E402  (resolve the inductor cache dir early)

import fmlib  # noqa: E402

assert fmlib.__file__.startswith(os.getcwd()), (
    f"fmlib came from {fmlib.__file__}, not from {os.getcwd()}"
)
print(f"[entry] fmlib={fmlib.__file__}", flush=True)

sys.argv = [sys.argv[0], *sys.argv[1:]]
runpy.run_module("fmlib.train", run_name="__main__", alter_sys=True)
