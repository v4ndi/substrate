"""Can a model trained before the refactor be loaded after it?

The refactor renamed 42 tensors (``tabular_backbone.encoder_blocks.N.*`` ->
``tabular_backbone.blocks.N.*``). Nothing in the new checkpoint code remaps
them, so this script tries the load for real and reports what happens.

Run from the *new* worktree, pointing at a checkpoint written by the old one.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
import yaml  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import fmlib  # noqa: E402

assert fmlib.__file__.startswith(os.getcwd())

config_path, checkpoint_path = sys.argv[1], sys.argv[2]
config = OmegaConf.create(yaml.safe_load(pathlib.Path(config_path).read_text()))
model = instantiate(config["model"])

payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state = payload.get("model", payload) if isinstance(payload, dict) else payload
if hasattr(state, "state_dict"):
    state = state.state_dict()

print(f"fmlib:     {fmlib.__file__}")
print(f"checkpoint: {checkpoint_path}")
print(f"tensors in checkpoint: {len(state)}")

try:
    model.load_state_dict(state)
    print("STRICT LOAD: ok")
except RuntimeError as exc:
    text = str(exc)
    missing = text.count("tabular_backbone.blocks")
    unexpected = text.count("tabular_backbone.encoder_blocks")
    print("STRICT LOAD: FAILED")
    print(f"  keys the model wants and the checkpoint lacks : {missing}")
    print(f"  keys in the checkpoint the model does not know: {unexpected}")
    print("  ---")
    print("  " + text.split("\n")[0][:200])

remapped = {k.replace(".encoder_blocks.", ".blocks."): v for k, v in state.items()}
try:
    model.load_state_dict(remapped)
    print("LOAD AFTER RENAMING encoder_blocks -> blocks: ok")
except RuntimeError as exc:
    print(f"LOAD AFTER RENAMING: still failing — {str(exc).splitlines()[0][:200]}")
