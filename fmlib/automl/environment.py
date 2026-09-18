"""Single local/Osiris dispatcher for AutoML actions."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Literal, TypeVar

from fmlib.automl.config.base import BaseTaskConfig
from fmlib.automl.exceptions import (
    ConfigError,
    MissingDependencyError,
    RemoteExecutionError,
)
from fmlib.automl.progress import use_progress_logger

logger = logging.getLogger(__name__)
_ResultT = TypeVar("_ResultT")


@contextmanager
def _cuda_visible_devices(value: str | None) -> Iterator[None]:
    """Scope ``CUDA_VISIBLE_DEVICES`` to one operation.

    ``value=None`` leaves the variable exactly as the caller set it. Otherwise
    the previous value -- including its absence -- is restored on the way out,
    so running AutoML in a notebook does not permanently narrow that kernel to
    one card.

    Args:
        value: Value to expose for the duration, or ``None`` to change nothing.

    Yields:
        ``None``.
    """
    if value is None:
        yield
        return
    missing = object()
    previous: Any = os.environ.get("CUDA_VISIBLE_DEVICES", missing)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    try:
        yield
    finally:
        if previous is missing:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _local_cuda_visibility(config: BaseTaskConfig) -> str | None:
    """Return the ``CUDA_VISIBLE_DEVICES`` value one local run should impose.

    Boosting is pinned to card 0 on purpose: CatBoost with ``task_type='GPU'``
    and no explicit ``devices`` spreads over every visible card. TabNN must not
    be pinned -- its device count comes from ``environment.num_gpus``, and a pin
    would both hide the other cards and make that field a lie.

    Args:
        config: Task configuration of the operation being run.

    Returns:
        ``"0"`` for boosting, ``None`` for any other backend family.
    """
    return "0" if config.backend == "boosting" else None


#: A job in one of these states will never change again.
_TERMINAL_STATES = {"failed", "succeeded", "lost"}
_STATE_ALIASES = {
    "created": "queued",
    "pending": "queued",
    "queued": "queued",
    "running": "running",
    "failed": "failed",
    "error": "failed",
    "killed": "failed",
    "succeeded": "succeeded",
    "success": "succeeded",
    "completed": "succeeded",
}

#: Substrings the scheduler puts in a state when a job is still starting. This
#: is a normal start, not a failure, and the job must never be resubmitted for
#: it. Matched as text because it arrives as an error string, not a state.
_STARTING_MARKERS = ("is not available", "containercreating", "podinitializing")

#: Substrings that mean the scheduler does not know this job. After a grace
#: period that becomes terminal; before it, a listing can simply lag.
_MISSING_MARKERS = ("notfound", "not found", "no such")

#: How long a job may be missing or unrecognised before it is declared lost.
#: Module constants with an environment override, because this is an
#: operational property of the scheduler, not of the training task.
MISSING_GRACE_SECONDS = float(os.environ.get("FMLIB_AUTOML_MISSING_GRACE", 300.0))
UNKNOWN_GRACE_SECONDS = float(os.environ.get("FMLIB_AUTOML_UNKNOWN_GRACE", 900.0))


def classify_scheduler_state(row: Mapping[str, Any] | None) -> str:
    """Map one scheduler listing row onto the states the driver reasons about.

    Args:
        row: The scheduler's entry for a job, or ``None`` when it has none.

    Returns:
        ``submitting``, ``queued``, ``running``, ``succeeded``, ``failed``,
        ``missing`` (the scheduler does not know this job) or ``unknown`` (the
        answer was not recognised at all). ``missing`` and ``unknown`` both get
        a finite deadline from :func:`apply_grace`; neither is silently active
        forever.
    """
    if row is None:
        return "missing"
    raw = str(row.get("state", "unknown"))
    lowered = raw.lower()
    if any(marker in lowered for marker in _STARTING_MARKERS):
        return "submitting"
    if any(marker in lowered for marker in _MISSING_MARKERS):
        return "missing"
    return _STATE_ALIASES.get(lowered, "unknown")


def apply_grace(
    state: str, since: float | None, now: float
) -> tuple[str, float | None]:
    """Give ``missing`` and ``unknown`` a deadline instead of an open end.

    Args:
        state: The state :func:`classify_scheduler_state` produced.
        since: When this job first entered that state, or ``None``.
        now: Current time.

    Returns:
        The state to record, and the timestamp to carry forward. A job that has
        been missing or unrecognised past its grace becomes ``lost``: terminal,
        diagnosable, and never resubmitted automatically -- recreating it is a
        decision for a person.
    """
    if state not in {"missing", "unknown"}:
        return state, None
    grace = MISSING_GRACE_SECONDS if state == "missing" else UNKNOWN_GRACE_SECONDS
    first_seen = now if since is None else float(since)
    if now - first_seen >= grace:
        return "lost", first_seen
    return state, first_seen


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    msg = f"Cannot serialize {type(value).__name__}"
    raise TypeError(msg)


def _load_osiris():
    """Import the DataLab scheduler client only when remote execution is requested."""
    clients = os.path.abspath("/opt/clients")
    if clients not in sys.path:
        sys.path.insert(0, clients)
    try:
        import osiris
    except ImportError as exc:
        msg = "Osiris is not installed; env_type='osiris' execution is unavailable"
        raise MissingDependencyError(msg) from exc
    return osiris


class EnvironmentRunner:
    """Run local actions and manage exact Osiris jobs for remote actions.

    Args:
        osiris_client: Optional scheduler-compatible client, primarily for tests.
            When omitted, Osiris is imported lazily on first remote operation.
    """

    def __init__(self, osiris_client: Any | None = None):
        """Initialize an execution runner.

        Args:
            osiris_client: Optional injected scheduler client.
        """
        self._osiris = osiris_client
        self._last_states: dict[str, str] = {}

    @property
    def osiris(self):
        """Return the injected or lazily imported scheduler client.

        Returns:
            An object exposing the Osiris ``create`` and ``list`` operations.

        Raises:
            MissingDependencyError: If Osiris is required but not installed.
        """
        if self._osiris is None:
            self._osiris = _load_osiris()
        return self._osiris

    def run_local(
        self,
        *,
        config: BaseTaskConfig,
        action: Literal["train", "predict", "calibrate", "evaluate"],
        callback: Callable[[], _ResultT],
    ) -> _ResultT:
        """Run one local action with run-scoped console and file logs.

        For ``backend='boosting'`` the launcher exposes card 0 for the duration
        of the callback and restores the caller's value afterwards; TabNN runs
        with the visibility it was given.

        Args:
            config: Local task configuration.
            action: Operation name used in logs.
            callback: Zero-argument operation to execute.

        Returns:
            The callback result without conversion.

        Raises:
            ConfigError: If the configuration is not local.
        """
        if config.env_type != "local":
            msg = "EnvironmentRunner.run_local requires env_type='local'"
            raise ConfigError(msg)
        run_id = uuid.uuid4().hex
        log_dir = (
            Path(config.environment.log_dir or Path(config.output_dir) / "logs")
            .expanduser()
            .resolve()
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.{action}.log"
        action_started = perf_counter()
        run_logger = logging.getLogger(f"fmlib.automl.local.{run_id}")
        run_logger.setLevel(logging.INFO)
        run_logger.propagate = False
        formatter = logging.Formatter(
            f"%(asctime)s %(levelname)s run_id={run_id} %(message)s"
        )
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ]
        for handler in handlers:
            handler.setFormatter(formatter)
            run_logger.addHandler(handler)
        try:
            run_logger.info(
                "Starting action=%s task_config=%s backend=%s engine=%s env_type=local device=%s",
                action,
                type(config).__name__,
                config.backend,
                config.engine,
                config.resolved_device,
            )
            with (
                _cuda_visible_devices(_local_cuda_visibility(config)),
                use_progress_logger(run_logger),
            ):
                result = callback()
            run_logger.info(
                "Finished action=%s duration_seconds=%.3f",
                action,
                perf_counter() - action_started,
            )
            return result
        except Exception:
            run_logger.exception(
                "AutoML action failed duration_seconds=%.3f",
                perf_counter() - action_started,
            )
            raise
        finally:
            for handler in handlers:
                run_logger.removeHandler(handler)
                handler.close()

    @staticmethod
    def _remote_python(config: BaseTaskConfig) -> str:
        if config.environment.venv_path is None:
            msg = f"environment.venv_path is required for env_type={config.env_type!r}"
            raise ConfigError(msg)
        venv = PurePosixPath(str(config.environment.venv_path))
        if not venv.is_absolute():
            msg = f"environment.venv_path must be absolute: {venv!r}"
            raise ConfigError(msg)
        if os.name == "posix" and not Path(venv).is_dir():
            msg = f"Remote fmlib venv does not exist: {venv}"
            raise ConfigError(msg)
        return str(venv / "bin" / "python")

    @staticmethod
    def _resources(config: BaseTaskConfig) -> tuple[int, int]:
        return (
            config.environment.resolved_num_nodes,
            config.environment.resolved_num_gpus,
        )

    @staticmethod
    def _job_id(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            for key in ("job_id", "job", "id", "name"):
                if response.get(key):
                    return str(response[key])
        for key in ("job_id", "job", "id", "name"):
            value = getattr(response, key, None)
            if value:
                return str(value)
        msg = f"Osiris response does not contain an exact job ID: {response!r}"
        raise RemoteExecutionError(msg)

    def submit(
        self,
        *,
        config: BaseTaskConfig,
        action: Literal["train", "predict", "calibrate", "evaluate"],
        payload: Mapping[str, Any],
        run_dir: Path,
        job_suffix: str | None = None,
    ) -> dict[str, Any]:
        """Persist one run specification and submit it to Osiris.

        Args:
            config: Osiris task configuration.
            action: Operation executed by the remote process.
            payload: JSON-serializable paths and operation-specific values.
            run_dir: Directory the run specification is written to.
            job_suffix: Appended to the scheduler job name, so one job of a
                fan-out can be told from another without reading its spec.

        Returns:
            A persistent handle containing the exact scheduler job ID.

        Raises:
            ConfigError: If local execution is requested.
            RemoteExecutionError: If the scheduler response has no job ID.
        """
        if config.env_type != "osiris":
            msg = "EnvironmentRunner.submit requires env_type='osiris'"
            raise ConfigError(msg)
        python = self._remote_python(config)
        num_nodes, num_gpus = self._resources(config)
        osiris_client = self.osiris
        run_id = uuid.uuid4().hex
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        spec_path = run_dir / "run_spec.json"
        result_path = run_dir / "result.json"
        requested_config = asdict(config)
        remote_config = dict(requested_config)
        remote_config["device"] = "gpu"
        spec = {
            "run_id": run_id,
            "task": config.task_name,
            "action": action,
            "config": remote_config,
            "requested_config": requested_config,
            "payload": dict(payload),
            "result_path": str(result_path),
            "log_path": str(run_dir / "remote.log"),
            "environment_profile": {
                "original_pool": config.environment.pool,
                "effective_pool": config.environment.effective_pool,
                "profile": config.environment.resource_profile,
                "num_gpus": num_gpus,
                "num_nodes": num_nodes,
            },
        }
        spec_path.write_text(
            json.dumps(spec, default=_json_default, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # The part index is in the name so a job orphaned between submit and
        # persistence is findable rather than anonymous.
        job_name = f"fmlib-{action}-{run_id[:8]}"
        if job_suffix:
            job_name = f"{job_name}-{job_suffix}"
        envs = dict(config.environment.env)
        if (
            config.environment.resource_profile == "supercomp"
            and config.backend == "boosting"
        ):
            envs["CUDA_VISIBLE_DEVICES"] = "0"
        kwargs: dict[str, Any] = {
            "name": job_name,
            "image": config.environment.image,
            "restart": False,
            "command": [python, "-m", "fmlib.automl.run"],
            "args": ["--spec", str(spec_path)],
            "envs": envs,
            "num_nodes": num_nodes,
            "num_gpus": num_gpus,
            "type": "pytorchjob",
        }
        if config.environment.effective_pool is not None:
            kwargs["pool"] = config.environment.effective_pool
        response = osiris_client.create(**kwargs)
        job_id = self._job_id(response)
        submit_log = run_dir / "submit.log"
        submit_log.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "job_id": job_id,
                    "job_name": job_name,
                    "request": kwargs,
                    "response": response,
                },
                default=_json_default,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Submitted osiris job %s (%s), run_id=%s original_pool=%r effective_pool=%r profile=%s num_gpus=%d num_nodes=%d",
            job_name,
            job_id,
            run_id,
            config.environment.pool,
            config.environment.effective_pool,
            config.environment.resource_profile,
            num_gpus,
            num_nodes,
        )
        return {
            "run_id": run_id,
            "job_id": job_id,
            "job_name": job_name,
            "action": action,
            "env_type": config.env_type,
            "original_pool": config.environment.pool,
            "effective_pool": config.environment.effective_pool,
            "resource_profile": config.environment.resource_profile,
            "num_gpus": num_gpus,
            "num_nodes": num_nodes,
            "spec_path": str(spec_path),
            "submit_log_path": str(submit_log),
            "remote_log_path": str(run_dir / "remote.log"),
            "result_path": str(result_path),
            "state": "queued",
        }

    def poll_jobs(
        self, jobs: list[Mapping[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Poll exact Osiris job IDs and return aggregate and detailed states."""
        rows = self.osiris.list().get("jobs", [])
        by_id = {
            str(row.get("job_id") or row.get("job") or row.get("id")): row
            for row in rows
        }
        details: list[dict[str, Any]] = []
        now = time.time()
        for job in jobs:
            row = by_id.get(str(job["job_id"]))
            observed = classify_scheduler_state(row)
            state, since = apply_grace(observed, job.get("unseen_since"), now)
            details.append(
                dict(job)
                | {
                    "state": state,
                    "observed_state": observed,
                    "unseen_since": since,
                    "scheduler": row,
                    "submitted_time": None
                    if row is None
                    else row.get("submitted_time") or row.get("created_at"),
                    "updated_time": None
                    if row is None
                    else row.get("updated_time") or row.get("updated_at"),
                }
            )
        states = {item["state"] for item in details}
        if states == {"succeeded"}:
            aggregate = "succeeded"
        elif states and states <= {"failed", "lost"}:
            aggregate = "failed"
        elif states and states <= _TERMINAL_STATES:
            aggregate = "partial_failed"
        elif "running" in states:
            aggregate = "running"
        elif "queued" in states:
            aggregate = "queued"
        elif "submitting" in states:
            aggregate = "submitting"
        else:
            # Only `missing` and `unknown` are left, and both are inside their
            # grace: still worth waiting for, but not forever.
            aggregate = "unknown"
        return aggregate, details

    def failure_logs(self, jobs: list[Mapping[str, Any]]) -> str:
        """Collect scheduler and shared-filesystem diagnostics for failed jobs."""
        chunks: list[str] = []
        for job in jobs:
            chunks.append(
                json.dumps(
                    job.get("scheduler", {}),
                    default=_json_default,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            for field in ("submit_log_path", "remote_log_path", "result_path"):
                path = Path(str(job.get(field, "")))
                if path.is_file():
                    chunks.append(
                        f"===== {path} =====\n{path.read_text(encoding='utf-8', errors='replace')}"
                    )
            for method_name in ("logs", "log"):
                method = getattr(self.osiris, method_name, None)
                if callable(method):
                    try:
                        chunks.append(
                            f"===== osiris.{method_name}({job['job_id']}) =====\n{method(job['job_id'])}"
                        )
                    except Exception as exc:
                        chunks.append(
                            f"osiris.{method_name} failed: {type(exc).__name__}: {exc}"
                        )
        return "\n".join(chunks)
