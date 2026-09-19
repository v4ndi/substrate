"""An Osiris stand-in that executes what it is handed, and can misbehave on cue.

There is no cluster on this machine and the ``osiris`` package is not even
installed, so no test can reach a real scheduler by accident. That limits where
the evidence comes from, not how much of the remote path is provable: a job is
a JSON spec plus an entrypoint, and ``execute_spec`` is an ordinary function, so
a local stand-in can *run* the job rather than record that it was asked to.

Three levels, in increasing honesty:

* the recording client (``_FakeOsiris`` in ``test_environment``) proves what was
  submitted;
* :class:`LocalScheduler` proves the whole fan-out, because it executes;
* the fault knobs below are the only way P1-P4 are provable at all -- a healthy
  cluster will not produce a vanished job or a mid-submit crash on request.

Everything runs under ``tmp_path``. Nothing here ever writes processed data
outside it: the directories are not deleted by default and the volume is
shared.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from automl.osiris_contract import validate_create_request
from fmlib.automl.run import execute_spec


class SchedulerFault(RuntimeError):
    """What the stub raises when a fault scenario says ``create`` should fail."""


class ContainerNotAvailable(RuntimeError):
    """The transient start-up error a healthy submit produces on the cluster."""


@dataclass
class Job:
    """One submitted job, as the stub remembers it."""

    job_id: str
    name: str
    spec_path: str | None
    request: dict[str, Any]
    state: str = "queued"
    error: str | None = None


@dataclass
class LocalScheduler:
    """Execute submitted specs locally and report their states back.

    Args:
        execute: Run each spec on submit. ``False`` leaves every job ``queued``
            until :meth:`run_pending` is called, which is how a test controls
            the moment a job finishes.
        fail_create_on: 1-based index of a ``create`` call that raises. Used to
            prove that jobs submitted before it are already persisted.
        vanish: Job names that stop appearing in :meth:`list` -- a job the
            scheduler has lost.
        container_unavailable_for: Job names whose first ``list`` reports the
            transient container error instead of a state.
        corrupt_result: Job names whose ``result.json`` is overwritten with
            unreadable content after the job runs.
        check_spec: Also validate the spec file each submission names. On by
            default; off only for a test that submits without one on purpose.
    """

    execute: bool = True
    check_spec: bool = True
    fail_create_on: int | None = None
    vanish: Sequence[str] = ()
    container_unavailable_for: Sequence[str] = ()
    corrupt_result: Sequence[str] = ()
    on_create: Callable[[dict[str, Any]], None] | None = None

    create_calls: list[dict[str, Any]] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    list_calls: int = 0
    _served_transient: set[str] = field(default_factory=set)

    # -- the two methods the runner actually uses --------------------------
    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Accept a submission, and run it unless the test says otherwise.

        Raises:
            SchedulerFault: When this call is the configured failing one. The
                job is *not* recorded, exactly as a rejected submit behaves.
        """
        # Checked before anything is recorded: a submission a real scheduler
        # would refuse must not become a job here either, or the stand-in
        # agrees with a mistake instead of catching it. Every test that submits
        # gets this for free, which is the point -- the one that mattered was
        # not about the request shape at all.
        validate_create_request(kwargs, check_spec=self.check_spec)
        self.create_calls.append(kwargs)
        index = len(self.create_calls)
        if self.fail_create_on is not None and index == self.fail_create_on:
            msg = f"scheduler refused submit #{index}"
            raise SchedulerFault(msg)
        if self.on_create is not None:
            self.on_create(kwargs)
        job = Job(
            job_id=f"job-{index}",
            name=kwargs.get("name", f"job-{index}"),
            spec_path=_spec_of(kwargs),
            request=dict(kwargs),
        )
        self.jobs.append(job)
        if self.execute:
            self._run(job)
        return {"job_id": job.job_id}

    def list(self) -> dict[str, Any]:
        """Report states, honouring whatever the test asked to go wrong."""
        self.list_calls += 1
        rows = []
        for job in self.jobs:
            if job.name in self.vanish:
                continue
            if (
                job.name in self.container_unavailable_for
                and job.name not in self._served_transient
            ):
                self._served_transient.add(job.name)
                rows.append({
                    "job_id": job.job_id,
                    "name": job.name,
                    "state": 'KubernetesError: container "pytorch" is not available',
                })
                continue
            rows.append({
                "job_id": job.job_id,
                "name": job.name,
                "state": job.state,
                "error": job.error,
            })
        return {"jobs": rows}

    # -- driving the stub from a test --------------------------------------
    def run_pending(self) -> list[Job]:
        """Execute every job still queued. Returns the jobs that ran."""
        pending = [job for job in self.jobs if job.state == "queued"]
        for job in pending:
            self._run(job)
        return pending

    def job_named(self, name: str) -> Job:
        """Look one job up by name, for a test that wants to change its state."""
        for job in self.jobs:
            if job.name == name:
                return job
        msg = f"no job named {name!r}; have {[job.name for job in self.jobs]}"
        raise KeyError(msg)

    def _run(self, job: Job) -> None:
        if job.spec_path is None:
            job.state = "failed"
            job.error = "submitted command carried no --spec"
            return
        job.state = "running"
        try:
            execute_spec(job.spec_path)
        except Exception as error:
            job.state = "failed"
            job.error = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
            return
        job.state = "succeeded"
        if job.name in self.corrupt_result:
            _corrupt(job.spec_path)


def _spec_of(request: dict[str, Any]) -> str | None:
    """Pull the spec path out of a submitted command, as a job's entrypoint would."""
    args = list(request.get("args") or ())
    if "--spec" in args:
        index = args.index("--spec")
        if index + 1 < len(args):
            return str(args[index + 1])
    return None


def _corrupt(spec_path: str) -> None:
    """Make a finished job's result unreadable, the way a half-written file is."""
    import json
    from pathlib import Path

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    Path(spec["result_path"]).write_text("{not json", encoding="utf-8")
