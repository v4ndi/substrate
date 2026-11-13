"""Filesystem resolution for parquet datasets.

Every path accepted by a dataset is either a plain local path (``/mnt/data``,
``file:///mnt/data``) or a URI understood by :mod:`pyarrow.fs` — in practice
``hdfs://namenode:8020/user/...``. :func:`resolve_filesystem` turns a path into
a ``(filesystem, path)`` pair; everything downstream (discovery, parquet reads,
file stats) goes through that filesystem object, so the same dataset code runs
against local disk and HDFS without branching.
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import Iterable, Mapping

import pyarrow.fs as pafs

__all__ = [
    "discover_parquet_files",
    "file_size",
    "file_stat",
    "is_local_filesystem",
    "resolve_filesystem",
    "resolve_filesystems",
]

# Sidecar files that live next to parquet parts and must never be read as data.
_IGNORED_PREFIXES = (".", "_")


def _strip_file_scheme(path: str) -> str:
    if path.startswith("file://"):
        path = path[len("file://") :]
    return path


def _build_filesystem(options: Mapping) -> pafs.FileSystem:
    """Build a filesystem from an explicit config mapping.

    ``{"type": "hdfs", "host": "namenode", "port": 8020, "user": "team"}`` ->
    :class:`pyarrow.fs.HadoopFileSystem`. ``type`` defaults to ``hdfs`` because
    local paths never need an options block.
    """
    options = dict(options)
    fs_type = str(options.pop("type", "hdfs")).lower()
    if fs_type in ("local", "file"):
        return pafs.LocalFileSystem(**options)
    if fs_type in ("hdfs", "viewfs"):
        return pafs.HadoopFileSystem(**options)
    raise ValueError(
        f"Unsupported filesystem type {fs_type!r}; pass a pyarrow.fs.FileSystem "
        "instance for anything other than 'local' or 'hdfs'"
    )


def resolve_filesystem(
    path: str,
    filesystem: pafs.FileSystem | Mapping | None = None,
) -> tuple[pafs.FileSystem, str]:
    """Return ``(filesystem, path)`` for ``path``.

    Args:
        path: local path, ``file://`` URI, or a URI with a scheme pyarrow can
            resolve (``hdfs://host:port/dir``).
        filesystem: an explicit :class:`pyarrow.fs.FileSystem`, or a mapping of
            constructor options (see :func:`_build_filesystem`). When given, the
            scheme and authority of ``path`` are ignored and only its path
            component is used.

    Returns:
        The filesystem and the path *within* it — never a URI, so it can be
        handed straight to ``pyarrow`` APIs that take ``filesystem=``.
    """
    if filesystem is not None:
        if isinstance(filesystem, Mapping):
            filesystem = _build_filesystem(filesystem)
        if not isinstance(filesystem, pafs.FileSystem):
            raise TypeError(
                "filesystem must be a pyarrow.fs.FileSystem or a mapping of "
                f"options, got {type(filesystem).__name__}"
            )
        if "://" in path:
            # Keep only the path component; the caller's filesystem wins.
            _, _, remainder = path.partition("://")
            _, slash, tail = remainder.partition("/")
            path = f"{slash}{tail}"
        return filesystem, path

    path = _strip_file_scheme(path)
    if "://" not in path:
        return pafs.LocalFileSystem(), os.path.abspath(path)
    return pafs.FileSystem.from_uri(path)


def resolve_filesystems(
    paths: str | Iterable[str],
    filesystem: pafs.FileSystem | Mapping | None = None,
) -> tuple[pafs.FileSystem, list[str]]:
    """Resolve one or many paths that must share a single filesystem."""
    if isinstance(paths, str):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("At least one dataset path is required")

    resolved_fs, first = resolve_filesystem(paths[0], filesystem)
    resolved = [first]
    for path in paths[1:]:
        other_fs, other_path = resolve_filesystem(path, filesystem)
        if not resolved_fs.equals(other_fs):
            raise ValueError(
                "All dataset paths must live on the same filesystem; "
                f"{paths[0]!r} and {path!r} do not"
            )
        resolved.append(other_path)
    return resolved_fs, resolved


def is_local_filesystem(filesystem: pafs.FileSystem) -> bool:
    """Whether ``filesystem`` is backed by the local POSIX filesystem."""
    if isinstance(filesystem, pafs.SubTreeFileSystem):
        return is_local_filesystem(filesystem.base_fs)
    return isinstance(filesystem, pafs.LocalFileSystem)


def discover_parquet_files(
    filesystem: pafs.FileSystem, paths: str | Iterable[str]
) -> list[str]:
    """Recursively list ``*.parquet`` files under ``paths``.

    Directories are walked recursively; a path that is itself a parquet file is
    taken as-is. Hidden and underscore-prefixed entries (``_SUCCESS``,
    ``.crc``, ``_temporary``) are skipped, matching ``glob``'s behaviour on the
    previous local-only implementation. The result is sorted so that shard
    ownership is derived from a canonical, reproducible file order.
    """
    if isinstance(paths, str):
        paths = [paths]

    files: list[str] = []
    for path in paths:
        info = filesystem.get_file_info(path)
        if info.type == pafs.FileType.NotFound:
            raise AssertionError(f"Directory {path} doesn't exist")
        if info.type == pafs.FileType.File:
            files.append(info.path)
            continue

        selector = pafs.FileSelector(path, recursive=True, allow_not_found=False)
        for entry in filesystem.get_file_info(selector):
            if entry.type != pafs.FileType.File:
                continue
            if not entry.base_name.endswith(".parquet"):
                continue
            relative = posixpath.relpath(entry.path, info.path)
            if any(
                part.startswith(_IGNORED_PREFIXES)
                for part in relative.split(posixpath.sep)
            ):
                continue
            files.append(entry.path)

    return sorted(files)


def file_stat(filesystem: pafs.FileSystem, path: str) -> pafs.FileInfo:
    """Return the :class:`pyarrow.fs.FileInfo` for a single file."""
    info = filesystem.get_file_info(path)
    if info.type == pafs.FileType.NotFound:
        raise FileNotFoundError(path)
    return info


def file_size(filesystem: pafs.FileSystem, path: str) -> int:
    """Size of ``path`` in bytes."""
    return int(file_stat(filesystem, path).size)
