"""Files the documentation tells you to open or run must actually exist.

Three scans, each deliberately narrow — a path check that guesses is a check
that gets switched off:

1. **Inline code** (`` `examples/basics/README.md` ``) that looks like a
   repo-relative path: it must resolve, next to the document or from the root.
2. **Shell fences**, but only a ``./script`` in *command position* (first token
   of a line). "Run this" is a promise; an argument like
   ``--artifact-root ./mlruns`` names a directory created at runtime and is not.
3. **Markdown link and image targets** that are not URLs — this is what catches
   a moved diagram.

Anything ambiguous — globs, absolute paths, URIs, shell variables — is skipped
rather than guessed at.
"""

from __future__ import annotations

import re

import pytest
from docscan import FENCED_BLOCK, INLINE_CODE, REPO_ROOT, markdown_files, relative

MARKDOWN = markdown_files()

#: Suffixes that make a slash-free token (``install.sh``) worth checking.
REPO_SUFFIXES = frozenset({
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".sh",
    ".ipynb",
    ".txt",
    ".toml",
    ".json",
    ".cfg",
    ".png",
    ".jpg",
})

#: Any of these means the token is not a plain repo path: globs, placeholders,
#: interpolations, URI schemes, shell syntax.
AMBIGUOUS = set("*?<>{}$|()[]\"'=,:!&;")

SHELL_FENCES = frozenset({"bash", "sh", "shell", "console", "zsh"})

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _normalise(token: str) -> str:
    token = token.strip().strip("`")
    token = token.removeprefix("./")
    return token.rstrip("/").rstrip(".,;")


def looks_like_repo_path(token: str) -> bool:
    """True for tokens we are confident name a file or directory in this repo."""
    if not token or any(c.isspace() for c in token):
        return False
    if token[0] in "/~-.#" or AMBIGUOUS & set(token):
        return False
    if "://" in token:
        return False
    has_suffix = any(token.endswith(s) for s in REPO_SUFFIXES)
    return "/" in token or has_suffix


def _check(path, candidates):
    """Candidates that resolve neither next to the document nor from the root."""
    return [
        token
        for token in candidates
        if not (path.parent / token).exists() and not (REPO_ROOT / token).exists()
    ]


@pytest.mark.parametrize("path", MARKDOWN, ids=relative)
def test_inline_code_paths_exist(path):
    text = path.read_text()
    candidates = {
        normalised
        for span in INLINE_CODE.findall(text)
        if looks_like_repo_path(normalised := _normalise(span))
    }
    missing = _check(path, sorted(candidates))
    assert not missing, "\n".join([
        f"{relative(path)} mentions paths that do not exist:",
        *missing,
    ])


@pytest.mark.parametrize("path", MARKDOWN, ids=relative)
def test_scripts_told_to_run_exist(path):
    candidates = set()
    for info, body in FENCED_BLOCK.findall(path.read_text()):
        if info.strip().lower() not in SHELL_FENCES:
            continue
        # A fence is a transcript: an earlier `cd` moves where a later `./x`
        # resolves from. Without this, "cd examples/foo" then "./run.sh" would
        # be reported missing, and the check would get switched off.
        prefix = ""
        for line in body.splitlines():
            first, _, rest = line.strip().partition(" ")
            if first == "cd":
                target = _normalise(rest.split(" ")[0])
                if looks_like_repo_path(target):
                    prefix = f"{prefix}{target}/"
                continue
            if first.startswith("./") and looks_like_repo_path(_normalise(first)):
                candidates.add(f"{prefix}{_normalise(first)}")
    missing = _check(path, sorted(candidates))
    assert not missing, "\n".join([
        f"{relative(path)} tells the reader to run scripts that do not exist:",
        *missing,
    ])


@pytest.mark.parametrize("path", MARKDOWN, ids=relative)
def test_link_and_image_targets_exist(path):
    candidates = set()
    for target in MARKDOWN_LINK.findall(path.read_text()):
        target = target.split("#")[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        candidates.add(_normalise(target))
    missing = _check(path, sorted(c for c in candidates if c))
    assert not missing, "\n".join([
        f"{relative(path)} links to files that do not exist:",
        *missing,
    ])
