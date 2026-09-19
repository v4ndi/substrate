"""What a submitted job has to look like, stated once and checked everywhere.

The stand-in scheduler used to accept whatever it was handed. That is the same
hole the stringified spec went through: a stand-in written by the same person
who wrote the producer shares that person's assumptions, so it agrees with
mistakes instead of catching them. A real scheduler does not agree. It rejects
an unknown field, it rejects an environment value that is not a string, and a
container started from a command that is not a command simply fails.

So the contract lives here, apart from both the producer and the stand-in, and
:class:`~automl.scheduler_stub.LocalScheduler` calls it on every submit. Every
test that submits a job therefore checks the shape of the request, whether or
not that is what the test is about.

**What this is not.** The contract is read off ``EnvironmentRunner.submit`` and
off what a container job API needs; it has *not* been checked against a live
Osiris, because there is none on this machine and the client is not installed.
It freezes our understanding and catches drift away from it. Confirming the
understanding itself needs someone with cluster access -- the checklist for
that is in ``docs/decisions/testing_plan.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Fields Osiris is given on every submit.
REQUIRED_FIELDS = frozenset({
    "name",
    "image",
    "restart",
    "command",
    "args",
    "envs",
    "num_nodes",
    "num_gpus",
    "type",
})

#: Fields that may be present. Anything outside the union is refused: a client
#: that quietly drops an unknown keyword turns a typo into a job that runs with
#: the wrong settings and says nothing.
OPTIONAL_FIELDS = frozenset({"pool"})

#: Keys ``execute_spec`` reads out of the spec file before it does any work.
SPEC_FIELDS = frozenset({
    "run_id",
    "task",
    "action",
    "config",
    "payload",
    "result_path",
    "log_path",
})

ACTIONS = frozenset({"train", "predict", "calibrate", "evaluate"})


class ContractViolation(AssertionError):
    """Raised for a submission a real scheduler would refuse."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def _check_string_list(value: Any, field: str) -> None:
    _require(
        isinstance(value, list), f"{field} must be a list, got {type(value).__name__}"
    )
    for index, item in enumerate(value):
        _require(
            isinstance(item, str),
            f"{field}[{index}] must be a string, got {type(item).__name__}: {item!r}",
        )


def validate_create_request(
    request: dict[str, Any], *, check_spec: bool = True
) -> None:
    """Refuse a submission a real scheduler would refuse.

    Args:
        request: The keyword arguments handed to the scheduler client.
        check_spec: Also open the spec file named in ``args`` and check it.
            Off for tests that submit a job whose spec is deliberately absent.

    Raises:
        ContractViolation: With a message naming the field and the value.
    """
    keys = set(request)
    missing = sorted(REQUIRED_FIELDS - keys)
    _require(not missing, f"submission is missing required fields: {missing}")
    unknown = sorted(keys - REQUIRED_FIELDS - OPTIONAL_FIELDS)
    _require(
        not unknown, f"submission carries fields Osiris does not accept: {unknown}"
    )

    for field in ("name", "image", "type"):
        value = request[field]
        _require(
            isinstance(value, str) and value.strip() != "",
            f"{field} must be a non-empty string, got {value!r}",
        )
    _require(
        isinstance(request["restart"], bool),
        f"restart must be a bool, got {type(request['restart']).__name__}",
    )

    _check_string_list(request["command"], "command")
    _check_string_list(request["args"], "args")
    _require(bool(request["command"]), "command must not be empty")

    envs = request["envs"]
    _require(isinstance(envs, dict), f"envs must be a dict, got {type(envs).__name__}")
    for name, value in envs.items():
        _require(isinstance(name, str), f"env name must be a string, got {name!r}")
        # A container environment is strings. An int here reaches the cluster as
        # whatever the client's serializer decides, which is not our decision to
        # make quietly.
        _require(
            isinstance(value, str),
            f"env {name!r} must be a string, got {type(value).__name__}: {value!r}",
        )

    for field in ("num_nodes", "num_gpus"):
        value = request[field]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{field} must be a non-negative int, got {value!r}",
        )
    _require(request["num_nodes"] >= 1, "num_nodes must be at least 1")

    if "pool" in request:
        _require(
            isinstance(request["pool"], str) and request["pool"].strip() != "",
            f"pool must be a non-empty string when present, got {request['pool']!r}",
        )

    _validate_entrypoint(request, check_spec=check_spec)


def _validate_entrypoint(request: dict[str, Any], *, check_spec: bool) -> None:
    """The command has to be a command, and the spec it names has to be a spec."""
    command = request["command"]
    _require(
        command[0].startswith("/"),
        f"command[0] must be an absolute interpreter path, got {command[0]!r}",
    )
    _require(
        command[1:] == ["-m", "fmlib.automl.run"],
        f"the remote entrypoint must be `-m fmlib.automl.run`, got {command[1:]!r}",
    )

    args = request["args"]
    _require(
        len(args) == 2 and args[0] == "--spec",
        f"args must be ['--spec', <path>], got {args!r}",
    )
    if not check_spec:
        return

    spec_path = Path(args[1])
    _require(
        spec_path.is_absolute(), f"the spec path must be absolute, got {args[1]!r}"
    )
    _require(
        spec_path.is_file(),
        f"the spec must exist before the job is submitted: {spec_path}",
    )
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractViolation(
            f"the spec at {spec_path} is not JSON: {error}"
        ) from None
    _require(
        isinstance(spec, dict),
        f"the spec must be a JSON object, got {type(spec).__name__}",
    )

    missing = sorted(SPEC_FIELDS - set(spec))
    _require(not missing, f"the spec is missing fields execute_spec reads: {missing}")
    _require(
        spec["action"] in ACTIONS,
        f"action must be one of {sorted(ACTIONS)}, got {spec['action']!r}",
    )
    # The bug this whole plan started from: a document that survived being
    # written and came back as a description of itself.
    for field in ("config", "payload"):
        _require(
            isinstance(spec[field], dict),
            f"spec[{field!r}] must be an object, got {type(spec[field]).__name__} "
            f"-- a string here means something was serialised by its repr",
        )
    for field in ("result_path", "log_path"):
        _require(
            isinstance(spec[field], str) and Path(spec[field]).is_absolute(),
            f"spec[{field!r}] must be an absolute path, got {spec[field]!r}",
        )


def minimal_request(spec_path: str | Path, *, name: str = "job") -> dict[str, Any]:
    """A submission that satisfies the contract, for tests about the stand-in.

    A test whose subject is the *scheduler* rather than the producer still has
    to hand it something a scheduler would accept. Building that here keeps the
    shape tied to :func:`validate_create_request` instead of drifting from it,
    and makes the two remaining hand-built requests obvious.
    """
    return {
        "name": name,
        "image": "registry.example/fmlib:test",
        "restart": False,
        "command": ["/usr/bin/python", "-m", "fmlib.automl.run"],
        "args": ["--spec", str(spec_path)],
        "envs": {},
        "num_nodes": 1,
        "num_gpus": 0,
        "type": "pytorchjob",
    }
