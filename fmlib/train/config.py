"""Resolve the run's distributed / AMP / DDP settings from config.

New configs write three small blocks::

    distributed:
      backend: null            # null -> nccl on GPU, gloo otherwise
      timeout_sec: 36000000
      gradient_accumulation_steps: 1
    amp: no                    # no | fp16 | bf16
    ddp:
      find_unused_parameters: false
    compile: null              # null | inductor | ...

Old configs carry an ``accelerator:`` block instead. It is still read, with a
``DeprecationWarning``, so the rewrite does not have to land in lockstep with
every out-of-repo config.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fmlib.train.dist import DEFAULT_TIMEOUT_SEC

AMP_CHOICES = ("no", "fp16", "bf16")

_DDP_KWARGS_TARGET = "accelerate.utils.DistributedDataParallelKwargs"


@dataclass(frozen=True)
class DistributedConfig:
    """Process-group settings and the accumulation factor.

    ``backend`` of ``None`` picks NCCL on CUDA and gloo otherwise.
    """

    backend: str | None = None
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class DDPConfig:
    """Options passed straight through to ``DistributedDataParallel``."""

    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = True
    static_graph: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Everything the loop needs to know about *how* to run, not *what* to run."""

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    amp: str = "no"
    compile: str | None = None

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.distributed.gradient_accumulation_steps


def _as_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)  # type: ignore[return-value]
    return dict(node)


def _validate_amp(amp: Any) -> str:
    # YAML resolves an unquoted `no` (and `off`, and `false`) to the boolean
    # False, so `amp: no` — which is what every config in the repo writes —
    # arrives here as a bool, not the string "no". Accept it. `True` is not
    # accepted: it does not say whether fp16 or bf16 was meant.
    if amp is None or amp is False:
        return "no"
    amp = str(amp)
    if amp not in AMP_CHOICES:
        raise ValueError(f"amp must be one of {AMP_CHOICES}, got {amp!r}")
    return amp


def _ddp_from_legacy(accelerator: dict[str, Any]) -> dict[str, Any]:
    """Pull DDP options out of ``kwargs_handlers``.

    accelerate hid them in a list of instantiable handler objects; only the
    ``DistributedDataParallelKwargs`` entry carries anything we still need.
    """
    fields = {f.name for f in DDPConfig.__dataclass_fields__.values()}
    resolved: dict[str, Any] = {}
    for handler in accelerator.get("kwargs_handlers") or []:
        if not isinstance(handler, dict):
            continue
        if handler.get("_target_") != _DDP_KWARGS_TARGET:
            continue
        resolved.update({
            key: value
            for key, value in handler.items()
            if key in fields and not key.startswith("_")
        })
    return resolved


def _from_legacy_accelerator(accelerator: dict[str, Any]) -> RunConfig:
    warnings.warn(
        "The 'accelerator:' config block is deprecated; accelerate has been "
        "replaced by plain torch.distributed. Use the 'distributed:', 'amp:', "
        "'ddp:' and 'compile:' keys instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    dynamo = accelerator.get("dynamo_plugin") or {}
    return RunConfig(
        distributed=DistributedConfig(
            gradient_accumulation_steps=int(
                accelerator.get("gradient_accumulation_steps") or 1
            ),
        ),
        ddp=DDPConfig(**_ddp_from_legacy(accelerator)),
        amp=_validate_amp(accelerator.get("mixed_precision")),
        compile=dynamo.get("backend") if isinstance(dynamo, dict) else None,
    )


def resolve_run_config(config: DictConfig | dict[str, Any]) -> RunConfig:
    """Build a :class:`RunConfig` from either the new or the legacy keys.

    New keys win when both are present, so a config can be migrated one key at
    a time.
    """
    keys = set(config.keys())
    has_new = bool(keys & {"distributed", "amp", "ddp", "compile"})
    if not has_new and "accelerator" in keys:
        return _from_legacy_accelerator(_as_dict(config["accelerator"]))

    distributed = _as_dict(config.get("distributed"))
    ddp = _as_dict(config.get("ddp"))
    return RunConfig(
        distributed=DistributedConfig(**distributed),
        ddp=DDPConfig(**ddp),
        amp=_validate_amp(config.get("amp")),
        compile=config.get("compile"),
    )
