"""Resolution of the run's distributed / AMP / DDP settings, old keys and new."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from fmlib.train.config import DDPConfig, RunConfig, resolve_run_config

LEGACY_WITH_DDP = {
    "accelerator": {
        "_target_": "accelerate.Accelerator",
        "_partial_": True,
        "dataloader_config": {
            "_target_": "accelerate.utils.DataLoaderConfiguration",
            "dispatch_batches": False,
        },
        "kwargs_handlers": [
            {
                "_target_": "accelerate.utils.DistributedDataParallelKwargs",
                "find_unused_parameters": True,
            }
        ],
        "gradient_accumulation_steps": 2,
        "dynamo_plugin": {
            "_target_": "accelerate.utils.TorchDynamoPlugin",
            "backend": "inductor",
        },
    }
}


def test_defaults_when_nothing_is_configured():
    resolved = resolve_run_config(OmegaConf.create({}))
    assert resolved == RunConfig()
    assert resolved.amp == "no"
    assert resolved.gradient_accumulation_steps == 1
    assert resolved.ddp.find_unused_parameters is False


def test_new_keys_are_read():
    config = OmegaConf.create({
        "distributed": {"backend": "gloo", "gradient_accumulation_steps": 4},
        "ddp": {"find_unused_parameters": True, "static_graph": True},
        "amp": "bf16",
        "compile": "inductor",
    })
    resolved = resolve_run_config(config)
    assert resolved.distributed.backend == "gloo"
    assert resolved.gradient_accumulation_steps == 4
    assert resolved.ddp == DDPConfig(find_unused_parameters=True, static_graph=True)
    assert resolved.amp == "bf16"
    assert resolved.compile == "inductor"


def test_legacy_accelerator_block_is_translated():
    with pytest.warns(DeprecationWarning):
        resolved = resolve_run_config(OmegaConf.create(LEGACY_WITH_DDP))
    assert resolved.gradient_accumulation_steps == 2
    assert resolved.ddp.find_unused_parameters is True
    assert resolved.compile == "inductor"
    # dataloader_config was accelerate's own loader wrapping; nothing maps to it.
    assert resolved.amp == "no"


def test_legacy_block_without_handlers_uses_ddp_defaults():
    config = OmegaConf.create({
        "accelerator": {
            "_target_": "accelerate.Accelerator",
            "dataloader_config": {"dispatch_batches": False},
        }
    })
    with pytest.warns(DeprecationWarning):
        resolved = resolve_run_config(config)
    assert resolved.ddp == DDPConfig()
    assert resolved.compile is None


def test_new_keys_win_over_a_legacy_block():
    """A config can be migrated one key at a time."""
    config = OmegaConf.create({**LEGACY_WITH_DDP, "amp": "fp16"})
    resolved = resolve_run_config(config)
    assert resolved.amp == "fp16"
    # Falling back per-key would be worse than ignoring the old block: it would
    # silently mix two sources. The new block is authoritative once present.
    assert resolved.gradient_accumulation_steps == 1


@pytest.mark.parametrize("amp", ["no", "fp16", "bf16"])
def test_amp_choices_are_accepted(amp):
    resolved = resolve_run_config(OmegaConf.create({"amp": amp}))
    assert resolved.amp == amp


def test_unknown_amp_is_rejected():
    with pytest.raises(ValueError, match="amp must be one of"):
        resolve_run_config(OmegaConf.create({"amp": "fp8"}))


def test_amp_none_means_disabled():
    assert resolve_run_config(OmegaConf.create({"amp": None})).amp == "no"


def test_unquoted_no_in_yaml_means_disabled():
    """`amp: no` is what every config writes, and YAML makes it a bool.

    Unquoted ``no``/``off``/``false`` all resolve to False before this code
    ever sees them, so rejecting the bool would reject every config in the
    repository.
    """
    config = OmegaConf.create("amp: no\n")
    assert config.amp is False
    assert resolve_run_config(config).amp == "no"


def test_amp_true_is_rejected():
    """`amp: yes` does not say whether fp16 or bf16 was meant."""
    with pytest.raises(ValueError, match="amp must be one of"):
        resolve_run_config(OmegaConf.create("amp: yes\n"))
