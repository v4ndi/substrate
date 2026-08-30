"""Scanning helpers shared by the documentation checks.

The three ``test_doc_*`` modules all need the same two things: the list of
markdown/YAML files that count as *documentation* (so, not the decision records
and not ``experiments/``), and a way to turn a dotted path into an object.
Collected here so the exclusion rules are stated once.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import re
import warnings
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Never scanned: VCS/venv noise, plus two directories that are out of scope on
#: purpose — ``experiments/`` is excluded from ruff and pre-commit as well, and
#: ``docs/decisions/`` holds design records that describe *past* states of the
#: code and are expected to name things that no longer exist.
EXCLUDED_DIRS = frozenset({
    ".git",
    ".venv",
    ".claude",
    "__pycache__",
    "experiments",
    "node_modules",
})
EXCLUDED_RELATIVE_DIRS = ("docs/decisions",)

#: Roadmap, not documentation: it describes work that has not happened yet.
EXCLUDED_FILES = frozenset({"TODO.md"})

#: ``avatar`` followed by at least one dotted segment. ``avatar_fm`` does not
#: match (the character after ``avatar`` must be a dot), and a sentence-final
#: period is not consumed (every dot must be followed by an identifier char).
DOTTED_PATH = re.compile(r"\bavatar(?:\.[A-Za-z_][A-Za-z0-9_]*)+")

#: Inline code spans: `like this`. Fenced blocks are handled separately.
INLINE_CODE = re.compile(r"`([^`\n]+)`")

#: Fenced blocks, with the info string (``bash``, ``yaml``, …) captured.
FENCED_BLOCK = re.compile(r"^```([^\n`]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _is_excluded(path: pathlib.Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    if set(relative.parts) & EXCLUDED_DIRS:
        return True
    if relative.name in EXCLUDED_FILES:
        return True
    posix = relative.as_posix()
    return any(posix.startswith(f"{d}/") for d in EXCLUDED_RELATIVE_DIRS)


def iter_files(suffix: str) -> list[pathlib.Path]:
    """Every in-scope file with ``suffix``, sorted, repo-root relative order."""
    return [p for p in sorted(REPO_ROOT.rglob(f"*{suffix}")) if not _is_excluded(p)]


def markdown_files() -> list[pathlib.Path]:
    return iter_files(".md")


def yaml_files() -> list[pathlib.Path]:
    return iter_files(".yaml") + iter_files(".yml")


def relative(path: pathlib.Path) -> str:
    """Repo-relative POSIX path, for readable test ids and failure messages."""
    return path.relative_to(REPO_ROOT).as_posix()


class Unverifiable(Exception):
    """The name may well be fine; an optional dependency stopped us checking.

    Raised when a module *exists* but importing it failed — ``pyspark`` or
    ``catboost`` missing, typically. Callers skip rather than fail, because a
    machine without the extra installed must not turn documentation checks red.
    """


def resolve_dotted(dotted: str) -> Any:
    """Resolve ``a.b.C.attr``, trying the longest importable module prefix first.

    Args:
        dotted: a fully qualified dotted path, e.g. ``avatar.data.TabularDataset``.

    Returns:
        The referenced object.

    Raises:
        AttributeError: the module imported but does not have that attribute.
        ModuleNotFoundError: no importable module prefix at all.
        Unverifiable: a module prefix exists but could not be imported.
    """
    parts = dotted.split(".")
    with warnings.catch_warnings():
        # Deprecation shims (avatar.nn.tabular.ste, avatar.train_utils) warn on
        # import. Resolving a name is not using it, so the warning is noise here.
        warnings.simplefilter("ignore")
        for split in range(len(parts), 0, -1):
            module_name = ".".join(parts[:split])
            try:
                spec_exists = importlib.util.find_spec(module_name) is not None
            except (ImportError, ValueError):
                # A parent package that itself fails to import; keep shortening.
                spec_exists = False
            if not spec_exists:
                continue
            try:
                obj = importlib.import_module(module_name)
            except ImportError as error:  # optional extra missing
                raise Unverifiable(f"{module_name}: {error}") from error
            for attribute in parts[split:]:
                obj = getattr(obj, attribute)  # AttributeError propagates
            return obj
    raise ModuleNotFoundError(f"no importable module prefix in {dotted!r}")


def dotted_paths(text: str) -> list[str]:
    """Every ``avatar.*`` dotted path in ``text``, in order, de-duplicated."""
    seen: dict[str, None] = {}
    for match in DOTTED_PATH.findall(text):
        seen.setdefault(match, None)
    return list(seen)
