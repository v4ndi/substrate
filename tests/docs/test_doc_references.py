"""Every ``fmlib.*`` name mentioned in the documentation must still exist.

This is the check that keeps the rest of the documentation honest. Renaming a
class is a one-line change in the code and an invisible break in six markdown
files; without this test nothing notices until a reader does.

Out of scope by design (see ``docscan.EXCLUDED_*``): ``docs/decisions/``, which
describes past states of the code on purpose, and ``experiments/``.
"""

from __future__ import annotations

import pytest
from docscan import Unverifiable, dotted_paths, markdown_files, relative, resolve_dotted

MARKDOWN = markdown_files()


@pytest.mark.parametrize("path", MARKDOWN, ids=relative)
def test_fmlib_references_resolve(path):
    broken = []
    for dotted in dotted_paths(path.read_text()):
        try:
            resolve_dotted(dotted)
        except Unverifiable:
            continue  # optional extra not installed; not a documentation bug
        except (AttributeError, ImportError) as error:
            broken.append(f"{dotted} -> {type(error).__name__}: {error}")
    assert not broken, "\n".join([
        f"{relative(path)} references names that no longer exist:",
        *broken,
    ])


def test_the_scan_actually_finds_references():
    """Guard against the regex silently matching nothing and the suite passing."""
    total = sum(len(dotted_paths(p.read_text())) for p in MARKDOWN)
    assert total > 20, f"only {total} fmlib.* references found — is the scan broken?"
