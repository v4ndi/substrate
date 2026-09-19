"""The contract itself: what it refuses, and that production satisfies it.

Two directions, and both are needed. A validator nobody has seen reject
anything is decoration, so the first half feeds it submissions a real scheduler
would turn down and checks it says which field and why. A validator production
does not satisfy is a test that will be deleted the first time it fails, so the
second half drives the real submit path and checks the request it builds passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automl.osiris_contract import (
    ContractViolation,
    minimal_request,
    validate_create_request,
)
from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.environment import EnvironmentRunner


@pytest.fixture
def valid(tmp_path) -> dict:
    spec = tmp_path / "run_spec.json"
    spec.write_text(
        json.dumps({
            "run_id": "0" * 32,
            "task": "binary",
            "action": "train",
            "config": {"backend": "boosting"},
            "payload": {"train_path": "/data/train"},
            "result_path": str(tmp_path / "result.json"),
            "log_path": str(tmp_path / "remote.log"),
        }),
        encoding="utf-8",
    )
    return minimal_request(spec)


def test_the_contract_accepts_a_well_formed_submission(valid):
    validate_create_request(valid)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(
            lambda r: r.pop("image"), "missing required fields", id="missing-field"
        ),
        pytest.param(
            lambda r: r.update(nodes=2),
            "fields Osiris does not accept",
            id="typo-field",
        ),
        pytest.param(lambda r: r.update(name=""), "name must be", id="empty-name"),
        pytest.param(
            lambda r: r.update(restart="no"), "restart must be a bool", id="restart-str"
        ),
        pytest.param(
            lambda r: r.update(envs={"OMP_NUM_THREADS": 8}),
            "must be a string",
            id="env-int",
        ),
        pytest.param(
            lambda r: r.update(envs={"PATH": Path("/usr/bin")}),
            "must be a string",
            id="env-path",
        ),
        pytest.param(
            lambda r: r.update(num_gpus=-1), "num_gpus must be", id="negative-gpus"
        ),
        pytest.param(lambda r: r.update(num_nodes=0), "at least 1", id="zero-nodes"),
        pytest.param(
            lambda r: r.update(command=["python", "-m", "fmlib.automl.run"]),
            "absolute interpreter path",
            id="relative-python",
        ),
        pytest.param(
            lambda r: r.update(command=["/usr/bin/python", "-m", "avatar.automl.run"]),
            "remote entrypoint",
            id="wrong-entrypoint",
        ),
        pytest.param(
            lambda r: r.update(args=["--spec"]), "args must be", id="truncated-args"
        ),
        pytest.param(lambda r: r.update(pool=""), "pool must be", id="empty-pool"),
    ],
)
def test_the_contract_refuses_what_a_scheduler_would_refuse(valid, mutate, expected):
    mutate(valid)
    with pytest.raises(ContractViolation, match=expected):
        validate_create_request(valid)


@pytest.mark.parametrize(
    ("break_spec", "expected"),
    [
        pytest.param(lambda p: p.unlink(), "must exist before the job", id="absent"),
        pytest.param(lambda p: p.write_text("not json"), "is not JSON", id="not-json"),
        pytest.param(
            lambda p: p.write_text('"a string"'), "must be a JSON object", id="scalar"
        ),
        pytest.param(
            lambda p: p.write_text(json.dumps({"run_id": "x"})),
            "missing fields execute_spec reads",
            id="incomplete",
        ),
    ],
)
def test_the_contract_reads_the_spec_the_job_will_be_given(valid, break_spec, expected):
    break_spec(Path(valid["args"][1]))
    with pytest.raises(ContractViolation, match=expected):
        validate_create_request(valid)


def test_a_config_serialised_by_its_repr_is_refused(valid):
    """The exact shape of the bug this plan started from.

    `TrialSpec.write` stored a DictConfig's repr and every in-process test
    stayed green. A scheduler handed a spec whose config is a string is handed
    a job that cannot start, so the contract says so here rather than in a
    traceback on a rank.
    """
    spec_path = Path(valid["args"][1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["config"] = str(spec["config"])
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ContractViolation, match="serialised by its repr"):
        validate_create_request(valid)


class _Recorder:
    """Records submissions and validates them, which is all Osiris does here."""

    def __init__(self):
        self.requests: list[dict] = []

    def create(self, **kwargs):
        validate_create_request(kwargs)
        self.requests.append(kwargs)
        return {"job_id": f"job-{len(self.requests)}"}

    def list(self):
        return {"jobs": []}


def test_the_real_submit_path_satisfies_the_contract(tmp_path):
    """Not a hand-built request: the one `EnvironmentRunner.submit` produces."""
    venv = tmp_path / "venv"
    venv.mkdir()
    import numpy as np
    import polars as pl

    rows = 200
    rng = np.random.default_rng(0)
    train = tmp_path / "train.parquet"
    pl.DataFrame({
        "epk_id": np.arange(rows),
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "group": rng.choice(["a", "b"], rows),
        "feature": rng.normal(size=rows),
        "target": rng.integers(0, 2, rows),
    }).write_parquet(train)

    recorder = _Recorder()
    task = BinaryTask(
        BinaryTaskConfig(
            env_type="osiris",
            backend="boosting",
            engine="catboost",
            device="gpu",
            target_column="target",
            client_id_column="epk_id",
            group_column="group",
            date_column="report_month",
            categorical_columns=(),
            numerical_columns=["feature"],
            hidden_state_columns=(),
            model_layout="global_and_per_group",
            hyperopt=False,
            model_params={"iterations": 5, "depth": 2, "thread_count": 1},
            verbose=False,
            output_dir=tmp_path / "outputs",
            environment={"venv_path": str(venv), "pool": "common"},
        )
    )
    task._environment_runner = EnvironmentRunner(recorder)
    task.train(train, train)

    # One job per model part, every one of them contract-clean.
    assert len(recorder.requests) == 3
    assert len({request["name"] for request in recorder.requests}) == 3
