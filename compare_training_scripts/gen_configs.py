"""Generate the paired training configs for the two revisions under comparison.

Each side starts from *its own* versioned copy of the pilot config
``experiments/sbercampaign_pilot/configs/pilot/train/td.yaml`` — the refactor
migrated that file, so the pair already differs exactly where the refactor
changed the config surface (``accelerator:`` vs ``distributed:``, the import
paths, the encoder rename) and nowhere else.

On top of that this script applies the same semantic patch to both sides:
the downloaded dataset, its real dimensions, no external hidden states, and a
shared MLflow store. Anything patched here is patched identically, so a metric
difference between the two runs can only come from the code.
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path("/home/jovyan/rusakov/runs/compare_training")
WORKTREES = pathlib.Path("/home/jovyan/rusakov/repos/avatar_fm/.claude/worktrees")
PILOT = "experiments/sbercampaign_pilot/configs/pilot/train/td.yaml"
TRACKING_URI = f"file:{ROOT}/mlruns"

# "fix" is the refactored branch plus the one-line evaluation-device fix, kept
# in this session's own worktree; its config surface is identical to "new".
SIDES = {"base": "cmp-base", "new": "cmp-new", "fix": "compare-training"}

# The uplift metric fits a beta calibrator per group and asserts it does not
# move ROC AUC. An under-trained model produces near-constant scores and trips
# that assertion at the first evaluation, so each dataset gets enough optimiser
# steps per epoch, a warmup shorter than one epoch, and an EMA that starts
# updating within the first epoch. Identical on both sides.
EPOCHS = {"hillstrom": 8, "lenta": 12}
WARMUP = {"hillstrom": 5, "lenta": 50}
BATCH_SIZE = {"hillstrom": 1024, "lenta": 4096}
SWA_MIN_STEPS = {"hillstrom": 16, "lenta": 128}
SWA_USAGE = False
NUM_WORKERS = 4


def build(dataset: str, side: str, seed: int, tag: str, swa: bool = SWA_USAGE) -> pathlib.Path:
    dims = json.loads((ROOT / "artifacts" / f"{dataset}_dims.json").read_text())
    config = yaml.safe_load((WORKTREES / SIDES[side] / PILOT).read_text())

    config["product"] = dataset
    config["root_dir"] = str(ROOT / "work" / tag)

    for loader, path in (
        ("train_dataloader", dims["train_path"]),
        ("valid_dataloader", dims["valid_path"]),
    ):
        node = config[loader]
        node["batch_size"] = BATCH_SIZE[dataset]
        node["num_workers"] = NUM_WORKERS
        node["dataset"]["path"] = path
        # the downloaded data carries no sequence hidden states
        node["dataset"].pop("hidden_state_column", None)

    model = config["model"]
    # SLearnerExp in the pilot config never existed in this package; SLearner is
    # the class both revisions actually export.
    model["_target_"] = "avatar.pipeline.uplift.SLearner"
    model["embedding"]["num_numerical_features"] = dims["num_numerical_features"]
    model["embedding"]["vocab_size"] = dims["vocab_size"]
    # one token per feature, plus the token the group embedding concatenates
    model["aggregation_config"]["num_features"] = (
        dims["num_numerical_features"] + dims["num_categorical_features"] + 1
    )
    model["n_groups"] = dims["n_groups"]
    model["hidden_state_dim"] = None
    model["proj_hiddens_to_dim"] = None

    config["scheduler"]["num_warmup_steps"] = WARMUP[dataset]
    config["swa_model"]["min_num_steps"] = SWA_MIN_STEPS[dataset]
    # Weight averaging is off for the headline comparison: on the refactored
    # side the averaged copy is built before the model reaches the GPU and
    # nothing moves it, so evaluation crashes (see the report). Turning it off
    # on *both* sides keeps the runs comparable; the bug is demonstrated and
    # fixed separately.
    config["swa_model"]["usage"] = swa
    config["train"]["num_epochs"] = EPOCHS[dataset]
    config["train"]["seed"] = seed

    config["mlflow"] = {
        "experiment_name": f"compare_{dataset}",
        "run_name": tag,
        "tracking_uri": TRACKING_URI,
    }

    out_dir = ROOT / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.yaml"
    with out_path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return out_path


def main() -> None:
    dataset = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [42, 43]
    plan = [(dataset, side, seed, f"{dataset}_{side}_seed{seed}", SWA_USAGE)
            for side in ("base", "new")
            for seed in seeds]
    # The weight-averaging path, which only the fixed branch survives on GPU.
    plan += [
        (dataset, "base", 42, f"{dataset}_base_swa_seed42", True),
        (dataset, "new", 42, f"{dataset}_new_swa_seed42", True),
        (dataset, "fix", 42, f"{dataset}_fix_swa_seed42", True),
    ]
    for name, side, seed, tag, swa in plan:
        path = build(name, side, seed, tag, swa=swa)
        (ROOT / "work" / tag).mkdir(parents=True, exist_ok=True)
        print(f"{tag:30} side={side:5} seed={seed} swa={swa}  -> {path.name}")


if __name__ == "__main__":
    main()
