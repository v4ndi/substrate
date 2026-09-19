"""T3: what the wheel actually contains, and whether it runs on its own.

A working copy answers every import from the source tree, so a file that is not
declared as package data is invisible until someone installs the package. That
is not hypothetical here: the tabnn templates and the default boosting search
space were both missing from `[tool.setuptools.package-data]` and were found by
luck rather than by a test.

Two checks, in that order:

* the wheel carries every non-Python file the tree has, so adding a template is
  enough and forgetting to package it is a red test rather than a support
  ticket;
* the package trains and scores when the working copy is not on the path at
  all, which is the only way to be sure nothing resolves through it.

Both are marked ``slow``: building a wheel is not free.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "fmlib"


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed; the wheel cannot be built here")
    return uv


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """Build a wheel from the working tree, leaving the tree as it was found."""
    out_dir = tmp_path_factory.mktemp("wheel")
    # Two local artifacts can make a build look more complete than the
    # configuration is. `build/` is a staging copy that keeps files which have
    # since been deleted. `*.egg-info/SOURCES.txt` is worse: with
    # include_package_data on, setuptools reads it and ships whatever a
    # *previous* build recorded, so a data file whose declaration was removed
    # keeps shipping and this test keeps passing. Both go before and after --
    # verified by removing the tabnn package-data line, which only turns this
    # test red once the egg-info is gone too.
    staging = [REPO_ROOT / "build", *REPO_ROOT.glob("*.egg-info")]

    def clean() -> None:
        for path in staging:
            shutil.rmtree(path, ignore_errors=True)

    clean()
    try:
        completed = subprocess.run(
            [_uv(), "build", "--wheel", "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=280,
            check=False,
        )
        assert completed.returncode == 0, (
            f"uv build exited {completed.returncode}\n{completed.stderr[-4000:]}"
        )
    finally:
        clean()
    wheels = sorted(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def unpacked(wheel, tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("unpacked")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(target)
    return target


def _tree_data_files() -> set[str]:
    """Non-Python files under the package, as posix paths relative to the root."""
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / PACKAGE).rglob("*")
        if path.is_file()
        and path.suffix not in (".py", ".pyc")
        and "__pycache__" not in path.parts
    }


def test_every_data_file_in_the_tree_is_in_the_wheel(wheel):
    """Adding a template must be enough; forgetting to package it must be red."""
    with zipfile.ZipFile(wheel) as archive:
        shipped = set(archive.namelist())
    missing = sorted(_tree_data_files() - shipped)
    assert not missing, (
        "these files exist in the tree but not in the wheel; declare them in "
        f"[tool.setuptools.package-data]: {missing}"
    )


def test_the_wheel_ships_the_package_and_nothing_else(wheel):
    """Tests, docs and experiments are not part of what a user installs."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    strays = sorted(
        name for name in names if not name.startswith((f"{PACKAGE}/", f"{PACKAGE}-"))
    )
    assert not strays, f"the wheel carries files outside the package: {strays}"
    for forbidden in ("tests/", "experiments/", "docs/", "examples/"):
        assert not [name for name in names if name.startswith(forbidden)]


# The working copy is installed editable, which resolves imports through a
# meta_path finder registered from a .pth file. Dropping that finder is what
# makes the import come from the wheel; the assertion right after is what
# proves it did, so this can never quietly test the working copy again.
FROM_THE_WHEEL = """
import json, sys
target = sys.argv[1]
sys.meta_path = [
    finder for finder in sys.meta_path
    if "editable" not in getattr(finder, "__name__", type(finder).__name__).lower()
    and "editable" not in getattr(finder, "__module__", "").lower()
]
sys.path.insert(0, target)

import fmlib
assert fmlib.__file__.startswith(target), f"fmlib came from {fmlib.__file__}"

import numpy as np, polars as pl
from pathlib import Path
from fmlib.automl import BinaryTask, BinaryTaskConfig
from fmlib.automl.config.boosting import default_search_space as boosting_space
from fmlib.automl.backends.tabnn.assembly import load_template

root = Path(sys.argv[2])
rng = np.random.default_rng(0)
rows = 400
feature = rng.normal(size=rows)
frame = pl.DataFrame({
    "epk_id": np.arange(rows),
    "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
    "feature": feature,
    "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-2 * feature))).astype(np.int8),
})
train = root / "train.parquet"
frame.write_parquet(train)

task = BinaryTask(BinaryTaskConfig(
    output_dir=root / "out", env_type="local", device="cpu", backend="boosting",
    engine="catboost", target_column="target", client_id_column="epk_id",
    group_column=None, date_column="report_month", categorical_columns=(),
    numerical_columns=["feature"], hidden_state_columns=(), hyperopt=False,
    verbose=False, random_state=42,
    model_params={"iterations": 20, "depth": 3, "thread_count": 1},
))
task.train(train, train)
prediction = task.predict(train)

print(json.dumps({
    "origin": fmlib.__file__,
    "rows": prediction.scores.height,
    "finite": bool(np.isfinite(prediction.scores["score"].to_numpy()).all()),
    # The two kinds of packaged data file, each read through the loader
    # production uses, not through importlib or a path the test invented.
    "boosting_space": sorted(boosting_space("catboost", n_trials=10)),
    "tabnn_templates": sorted(
        task for task in ("binary", "multiclass", "regression", "response", "uplift")
        if load_template(task)
    ),
}))
"""


def test_the_wheel_trains_and_predicts_without_the_working_copy(unpacked, tmp_path):
    """Nothing may resolve through the source tree -- including data files."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", FROM_THE_WHEEL, str(unpacked), str(tmp_path)],
        cwd=os.sep,
        env=environment,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
    )
    assert completed.returncode == 0, (
        f"child exited {completed.returncode}\n{completed.stderr[-4000:]}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["origin"].startswith(str(unpacked))
    assert payload["rows"] == 400
    assert payload["finite"] is True
    assert payload["boosting_space"], "the packaged default search space is empty"
    assert payload["tabnn_templates"] == [
        "binary",
        "multiclass",
        "regression",
        "response",
        "uplift",
    ], "a tabnn template is missing from the wheel or failed to parse"
