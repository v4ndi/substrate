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
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

from fmlib.automl.progress import log_progress

__all__ = [
    "RESULT_NAME",
    "SPEC_NAME",
    "InProcessRunner",
    "TrialResult",
    "TrialSpec",
    "run_trial",
]

SPEC_NAME = "run_spec.json"
RESULT_NAME = "result.json"

TrialState = Literal["COMPLETE", "FAIL", "LOST"]


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
    model = trainer = train_dataloader = valid_dataloader = None
    optimizer = scheduler = callbacks = valid_metrics = None
    try:
        run_config = resolve_run_config(config)
        env = DistEnv.from_env(
            backend=run_config.distributed.backend,
            timeout_sec=run_config.distributed.timeout_sec,
        )
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
