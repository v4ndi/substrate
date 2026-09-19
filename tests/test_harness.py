"""The test harness itself: markers, environment gating, deadlock catcher.

Configuration that decides what runs is worth asserting. A marker that is
deselected by default and never skipped on the machines that cannot run it is
indistinguishable from a marker nobody wrote, and the difference only shows up
on the day someone runs the full set.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tests.conftest import environment_skips

REPO_ROOT = Path(__file__).resolve().parents[1]

GATED_MARKERS = ("slow", "gpu", "cluster")


@pytest.fixture(scope="module")
def pytest_config() -> dict:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["tool"]["pytest"]["ini_options"]


def test_every_gated_marker_is_declared(pytest_config):
    """An undeclared marker is an error under --strict-markers; keep it that way."""
    declared = {entry.split(":", 1)[0] for entry in pytest_config["markers"]}
    assert set(GATED_MARKERS) <= declared
    assert "--strict-markers" in pytest_config["addopts"]


def test_the_default_run_deselects_everything_that_needs_setup(pytest_config):
    """The default `pytest` must stay runnable with no extra hardware or access."""
    expression = pytest_config["addopts"]
    for marker in GATED_MARKERS:
        assert f"not {marker}" in expression, f"{marker} is not deselected by default"


def test_a_timeout_is_configured_and_is_a_deadlock_catcher(pytest_config):
    """Generous against the slowest real test, tight against a hang."""
    timeout = pytest_config["timeout"]
    # The slowest test in the repo takes ~26s; a limit below that would turn a
    # slow machine into a red run, and one above ~10 min stops catching hangs.
    assert 120 <= timeout <= 600


def test_environment_skips_explain_themselves():
    """Every skip carries a reason a reader can act on."""
    for marker, reason in environment_skips().items():
        assert marker in GATED_MARKERS
        assert reason and not reason.endswith("."), reason


def test_cluster_is_gated_on_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("FMLIB_OSIRIS_TESTS", raising=False)
    assert "cluster" in environment_skips()

    monkeypatch.setenv("FMLIB_OSIRIS_TESTS", "1")
    assert "cluster" not in environment_skips()


def test_gpu_gating_follows_the_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert environment_skips()["gpu"] == "no CUDA device is available"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert "gpu" not in environment_skips()


@pytest.mark.gpu
def test_a_gpu_marked_test_never_fails_on_a_cpu_machine():
    """Self-check: on a box without a device this body must not be reached.

    It is deselected by default and skipped by ``pytest_collection_modifyitems``
    when selected anyway, so reaching this line means a device is present.
    """
    import torch

    assert torch.cuda.is_available()
