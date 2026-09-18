"""No module of the AutoML layer imports torch, except the TabNN backend.

Acceptance criterion 16, second half. This is what keeps the boosting path
light: the task layer, the config, the metrics registry, the artifact store and
the boosting adapters are all pure polars/numpy, and the one package that needs
a GPU stack is loaded only when a run actually asks for ``backend='tabnn'``.

It is a static check over the source rather than a look at ``sys.modules``,
because ``fmlib/__init__.py`` imports ``fmlib.losses``, so the *process* has
torch loaded the moment anything under ``fmlib`` is imported. The property that
can hold, and the one that matters, is about the source.
"""

from __future__ import annotations

import ast
import pathlib

AUTOML = pathlib.Path(__file__).resolve().parents[2] / "fmlib" / "automl"

#: The one package allowed to import torch, because its job is to run inside
#: the training loop.
ALLOWED = ("backends/tabnn",)


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_only_the_tabnn_backend_imports_torch():
    offenders = []
    for path in sorted(AUTOML.rglob("*.py")):
        relative = path.relative_to(AUTOML.parent.parent).as_posix()
        if any(part in relative for part in ALLOWED):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        torchy = sorted(
            name
            for name in _imported_names(tree)
            if name == "torch" or name.startswith("torch.")
        )
        if torchy:
            offenders.append(f"{relative}: {torchy}")
    assert not offenders, "\n".join([
        "These AutoML modules import torch; only the TabNN backend may:",
        *offenders,
    ])


def test_the_scan_would_notice_if_it_stopped_finding_anything():
    """Guard against the scan silently walking an empty tree."""
    modules = [path for path in AUTOML.rglob("*.py") if "__pycache__" not in str(path)]
    assert len(modules) > 30, f"only {len(modules)} AutoML modules found"


def test_the_tabnn_backend_is_where_torch_lives():
    """The exemption is real, not hypothetical: state it by naming a module."""
    metric = AUTOML / "backends" / "tabnn" / "metric.py"
    tree = ast.parse(metric.read_text(encoding="utf-8"))
    assert "torch" in _imported_names(tree)
