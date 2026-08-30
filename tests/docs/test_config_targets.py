"""Every ``_target_`` in every shipped YAML config must still be instantiable.

Configs are the real public API — 500-odd ``_target_`` uses against a few dozen
classes — and they are not exercised by any other test: a config only fails when
someone launches a run with it, which can be weeks after the rename that broke
it. Resolving the target is cheap and catches exactly that class of break.

Only the name is resolved, not instantiated: constructing a pipeline needs data
and a GPU, and a missing class is the failure worth catching here.
"""

from __future__ import annotations

import pytest
import yaml
from docscan import Unverifiable, relative, resolve_dotted, yaml_files

YAML_FILES = yaml_files()


def iter_targets(node):
    """Yield every ``_target_`` string in a parsed YAML document, depth first."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_" and isinstance(value, str):
                yield value
            else:
                yield from iter_targets(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_targets(item)


@pytest.mark.parametrize("path", YAML_FILES, ids=relative)
def test_config_targets_resolve(path):
    try:
        document = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        pytest.fail(f"{relative(path)} is not valid YAML: {error}")

    broken = []
    for target in iter_targets(document):
        if not target.startswith("avatar."):
            continue  # torch / transformers targets are the framework's problem
        try:
            resolve_dotted(target)
        except Unverifiable:
            continue
        except (AttributeError, ImportError) as error:
            broken.append(f"{target} -> {type(error).__name__}: {error}")
    assert not broken, "\n".join([
        f"{relative(path)} names targets that no longer exist:",
        *broken,
    ])


def test_the_scan_actually_finds_targets():
    """Guard against the walk silently yielding nothing and the suite passing."""
    total = 0
    for path in YAML_FILES:
        try:
            document = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        total += sum(1 for t in iter_targets(document) if t.startswith("avatar."))
    assert total > 100, f"only {total} avatar targets found — is the walk broken?"
