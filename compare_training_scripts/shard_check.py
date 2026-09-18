"""Check how a dataset splits records across ranks — the heart of the refactor.

Run under torchrun with 2 processes and the gloo backend (this host has one
GPU, so a real two-device DDP run is not available; record sharding is process
logic and does not need the GPU to be exercised).

Each rank builds the dataset exactly as the training config declares it — so
each revision uses its own class — walks one epoch, and records a hash per
record. The merged result answers three questions:

* coverage: how many of the dataset's records were seen at all;
* overlap: how many records both ranks saw (should be none);
* balance: how evenly the two ranks were loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import fmlib  # noqa: E402

assert fmlib.__file__.startswith(os.getcwd()), (
    f"fmlib came from {fmlib.__file__}, not from {os.getcwd()}"
)

config_path, out_dir = sys.argv[1], pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

dist.init_process_group(backend="gloo")
rank, world = dist.get_rank(), dist.get_world_size()

config = OmegaConf.create(yaml.safe_load(pathlib.Path(config_path).read_text()))
dataset = instantiate(config["train_dataloader"]["dataset"])


def record_hash(record) -> str:
    digest = hashlib.blake2b(digest_size=12)
    for key in sorted(record):
        value = record[key]
        if key == "tab_features" and isinstance(value, dict):
            for sub in sorted(value):
                item = value[sub]
                if item is not None:
                    digest.update(np.asarray(item).tobytes())
            continue
        if hasattr(value, "numpy"):
            value = value.numpy()
        digest.update(str(key).encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


hashes = []
for record in dataset:
    hashes.append(record_hash(record))

payload = {
    "rank": rank,
    "world": world,
    "fmlib": fmlib.__file__,
    "n_records": len(hashes),
    "n_unique": len(set(hashes)),
    "hashes": sorted(set(hashes)),
}
(out_dir / f"rank{rank}.json").write_text(json.dumps(payload))
print(f"[rank {rank}/{world}] {len(hashes)} records, {len(set(hashes))} unique", flush=True)

dist.barrier()
dist.destroy_process_group()
