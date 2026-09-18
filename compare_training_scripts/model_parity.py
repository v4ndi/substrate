"""Instantiate the model from a config and dump its parameter inventory.

Run once per revision, from that revision's worktree; the two dumps are then
compared. If the parameter names, shapes, count or the seeded initial values
differ, the two trainings are not training the same model and comparing their
metrics would prove nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
import yaml  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import fmlib  # noqa: E402

assert fmlib.__file__.startswith(os.getcwd()), (
    f"fmlib came from {fmlib.__file__}, not from {os.getcwd()}"
)

config_path, out_path = sys.argv[1], sys.argv[2]
config = OmegaConf.create(yaml.safe_load(pathlib.Path(config_path).read_text()))

torch.manual_seed(config["train"]["seed"])
model = instantiate(config["model"])

params = [(name, list(tensor.shape)) for name, tensor in model.named_parameters()]
total = sum(tensor.numel() for _, tensor in model.named_parameters())

digest = hashlib.sha256()
for name, tensor in sorted(model.state_dict().items()):
    digest.update(name.encode())
    digest.update(tensor.detach().cpu().numpy().tobytes())

# Values only, in declaration order: unlike the digest above this survives a
# pure rename, so it answers "same seed, same initial weights?" on its own.
values = hashlib.sha256()
for _, tensor in model.named_parameters():
    values.update(tensor.detach().cpu().numpy().tobytes())

report = {
    "fmlib": fmlib.__file__,
    "class": type(model).__name__,
    "total_params": total,
    "n_tensors": len(params),
    "params": params,
    "init_sha256": digest.hexdigest(),
    "values_sha256": values.hexdigest(),
}
pathlib.Path(out_path).write_text(json.dumps(report, indent=2))
print(f"{type(model).__name__}: {total} params in {len(params)} tensors")
print(f"init sha256: {report['init_sha256'][:16]}  from {fmlib.__file__}")
