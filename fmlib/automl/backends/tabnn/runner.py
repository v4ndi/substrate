"""Where one trial is executed.

The logic of a trial is one thing; where it runs is another. The interface is
asynchronous in shape -- ``submit(spec) -> handle`` and
``collect(handle) -> TrialResult`` -- because an Osiris job does not fit a
synchronous one; a synchronous runner simply does the work inside ``submit``.

``InProcessRunner`` is the only runner that executes anything today. The
subprocess runner (several GPUs on one node) and the Osiris runner (one job per
trial) arrive with the stages that can test them.

A trial returns a **number and a path**, never a model. The winner is
reconstructed from its checkpoint once, after the search. That is what keeps
the coordinator free of torch objects between trials, and it is the only shape
that works for a runner that cannot return a Python object at all.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from omegaconf import DictConfig, ListConfig, OmegaConf

from fmlib.automl.progress import log_progress

__all__ = [
    "RESULT_NAME",
    "SPEC_NAME",
    "InProcessRunner",
    "TorchrunRunner",
    "TrialResult",
    "TrialSpec",
    "run_trial",
]

SPEC_NAME = "run_spec.json"
RESULT_NAME = "result.json"

TrialState = Literal["COMPLETE", "FAIL", "LOST"]


def _plain(value: Any) -> Any:
    """Return ``value`` as plain Python containers, resolving any OmegaConf node."""
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class TrialSpec:
    """Everything one trial needs, in a form that survives a JSON round trip.

    The config *is* the spec: the same document a hand-written run would be
    given, which is what makes a trial reproducible outside AutoML.
    """

    trial_id: str
    config: Mapping[str, Any]
    trial_dir: str
    metric_name: str
    direction: str
    seed: int = 42
    num_gpus: int = 1
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The config arrives as a DictConfig from the assembly, and a DictConfig
        # is not a dict: `dataclasses.asdict` walks straight past it and
        # `json.dumps(default=str)` then writes its *repr* into run_spec.json.
        # The in-process runner never notices -- it holds the object -- but a
        # rank reading the spec back gets a string where the run should be.
        # Normalising here rather than at each call site keeps the guarantee
        # the spec claims: what is written is what is run.
        object.__setattr__(self, "config", _plain(self.config))
        object.__setattr__(self, "params", _plain(self.params))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrialSpec:
        return cls(**dict(payload))

    def write(self) -> Path:
        """Persist the spec next to the trial, before anything is executed."""
        path = Path(self.trial_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / SPEC_NAME
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return target


@dataclass(frozen=True)
class TrialResult:
    """What a trial reports back: a number, a path, and what went wrong."""

    trial_id: str
    state: TrialState
    objective: float | None = None
    checkpoint_dir: str | None = None
    error: str | None = None
    duration: float = 0.0

    @property
    def completed(self) -> bool:
        return self.state == "COMPLETE" and self.objective is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrialResult:
        return cls(**dict(payload))

    def write(self, trial_dir: str | Path) -> Path:
        path = Path(trial_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / RESULT_NAME
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return target

    @classmethod
    def read(cls, trial_dir: str | Path) -> TrialResult | None:
        path = Path(trial_dir) / RESULT_NAME
        if not path.is_file():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def run_trial(spec: TrialSpec) -> TrialResult:
    """Train one trial here, in this process, and report a number and a path.

    Every object is built by the same ``instantiate`` / ``init_*`` layer a
    hand-written ``python -m fmlib.train`` uses. Nothing about the loop, the
    checkpointing, the optimizer or the inference path is reimplemented.

    Args:
        spec: The trial to run.

    Returns:
        ``COMPLETE`` with the best validation value, or ``FAIL`` with the
        traceback. A failing trial must not end a search.
    """
    # Imported here rather than at module import: this is the point where a
    # run genuinely needs torch, and the task layer must be able to load this
    # module's neighbours without it.
    import torch
    from hydra.utils import instantiate

    from fmlib.train.config import resolve_run_config
    from fmlib.train.dist import DistEnv, seed_everything
    from fmlib.train.factory import build_callbacks
    from fmlib.train.loop import Trainer
    from fmlib.training_arguments import TrainingArguments
    from fmlib.utils.init_modules import (
        init_dataloaders,
        init_early_stopping,
        init_metrics,
        init_optimizer,
        init_scheduler,
    )

    started = perf_counter()
    config: DictConfig = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.create(dict(spec.config)), resolve=True)
    )
    spec.write()
    env = None
    is_main = True
    model = trainer = train_dataloader = valid_dataloader = None
    optimizer = scheduler = callbacks = valid_metrics = None
    try:
        run_config = resolve_run_config(config)
        env = DistEnv.from_env(
            backend=run_config.distributed.backend,
            timeout_sec=run_config.distributed.timeout_sec,
        )
        is_main = env.is_main
        train_config = OmegaConf.to_container(config["train"])
        early_stopping = init_early_stopping(train_config)
        training_arguments = TrainingArguments(**train_config)
        seed_everything(training_arguments.seed, rank=env.rank)

        model = instantiate(config["model"])
        train_dataloader, valid_dataloader, _ = init_dataloaders(config)
        _, valid_metrics, _ = init_metrics(config)
        optimizer = init_optimizer(config, model=model)
        scheduler = init_scheduler(
            config,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=run_config.gradient_accumulation_steps,
        )
        callbacks = build_callbacks(
            config,
            model=model,
            train_dataloader=train_dataloader,
            checkpoint_dir=spec.trial_dir,
            early_stopping=early_stopping,
            train_metrics=None,
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            valid_metrics=valid_metrics,
            training_arguments=training_arguments,
            run_config=run_config,
            env=env,
            callbacks=callbacks,
            checkpoint_dir=spec.trial_dir,
            config=config,
        )
        trainer.train()
        # `train()` returns the *test* score, and there is no test loader here:
        # AutoML scores with its own evaluate(). The number that ranks trials is
        # what early stopping recorded on the best validation pass.
        objective = trainer.state.best_metric
        if not is_main:
            # Only rank 0 holds the scores -- early stopping records the best
            # metric there and nowhere else -- so only rank 0 has a number to
            # report, and only rank 0 writes the result. The other ranks
            # finished their work, which is what their state says.
            result = TrialResult(
                trial_id=spec.trial_id,
                state="COMPLETE",
                checkpoint_dir=spec.trial_dir,
                duration=perf_counter() - started,
            )
        else:
            result = TrialResult(
                trial_id=spec.trial_id,
                state="COMPLETE" if objective is not None else "FAIL",
                objective=None if objective is None else float(objective),
                checkpoint_dir=spec.trial_dir,
                error=None
                if objective is not None
                else "no validation metric was recorded",
                duration=perf_counter() - started,
            )
    except Exception as error:
        log_progress(
            "[tabnn trial %s] failed: %s: %s",
            spec.trial_id,
            type(error).__name__,
            error,
        )
        result = TrialResult(
            trial_id=spec.trial_id,
            state="FAIL",
            error=f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            duration=perf_counter() - started,
        )
    finally:
        # Every one of these has to go before the next trial starts. The two
        # datasets each hold a torch shared-memory descriptor for their epoch
        # counter, and the trainer keeps the loaders alive through its
        # callbacks, so dropping only the model leaks two descriptors a trial
        # -- slowly, invisibly, and for the whole length of a search.
        model = trainer = train_dataloader = valid_dataloader = None
        optimizer = scheduler = callbacks = valid_metrics = None
        gc.collect()
        if env is not None:
            env.destroy()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if is_main:
        result.write(spec.trial_dir)
    return result


class InProcessRunner:
    """Run each trial here, one after another.

    The only runner that executes anything today, which is why the leak check
    in S10 matters: if it leaks, a subprocess runner stops being optional.
    """

    def submit(self, spec: TrialSpec) -> str:
        """Run the trial now and return its id as the handle."""
        log_progress("[tabnn trial %s] started in-process", spec.trial_id)
        self._results = getattr(self, "_results", {})
        self._results[spec.trial_id] = run_trial(spec)
        return spec.trial_id

    def collect(self, handle: str) -> TrialResult:
        """Return the result of an already-finished trial."""
        results = getattr(self, "_results", {})
        if handle in results:
            return results[handle]
        msg = f"Unknown trial handle {handle!r}"
        raise KeyError(msg)


class TorchrunRunner:
    """Run each trial in its own ``torch.distributed.run`` group.

    One trial, ``num_gpus`` ranks, one node. The driver never joins the group:
    it writes a spec, waits for the launcher, and reads ``result.json``, which
    is the same collection path an Osiris job uses. A non-zero exit code is a
    failed trial, not an exception in the driver -- a failing trial must not
    end a search.

    Args:
        timeout: Seconds to wait for one trial before giving up on it.
        env: Extra environment for the worker processes.
    """

    def __init__(
        self, *, timeout: float = 24 * 3600.0, env: Mapping[str, str] | None = None
    ):
        self.timeout = timeout
        self.env = dict(env or {})
        self._handles: dict[str, subprocess.Popen] = {}
        self._specs: dict[str, TrialSpec] = {}

    def submit(self, spec: TrialSpec) -> str:
        """Launch the trial and return its id; does not wait."""
        spec.write()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={max(1, int(spec.num_gpus))}",
            "--nnodes=1",
            "--standalone",
            "-m",
            "fmlib.automl.backends.tabnn.worker",
            "--spec",
            str(Path(spec.trial_dir) / SPEC_NAME),
        ]
        log_progress(
            "[tabnn trial %s] launching %d rank(s) under torchrun",
            spec.trial_id,
            max(1, int(spec.num_gpus)),
        )
        self._specs[spec.trial_id] = spec
        self._handles[spec.trial_id] = subprocess.Popen(
            command,
            env={**os.environ, **self.env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return spec.trial_id

    def collect(self, handle: str) -> TrialResult:
        """Wait for a launched trial and read back what rank 0 wrote."""
        process = self._handles.pop(handle)
        spec = self._specs.pop(handle)
        try:
            output, _ = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            return TrialResult(
                trial_id=handle,
                state="LOST",
                error=f"trial exceeded {self.timeout:g}s and was killed\n{output}",
            )
        result = TrialResult.read(spec.trial_dir)
        if result is not None and process.returncode == 0:
            return result
        if result is not None:
            return result
        return TrialResult(
            trial_id=handle,
            state="FAIL",
            error=f"torchrun exited with {process.returncode} and wrote no result\n{output}",
        )
