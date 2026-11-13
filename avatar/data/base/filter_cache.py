"""Persistent index of the rows a filter accepts, shared by every rank.

Sharding needs to know how many records each file contributes *after* filtering
before it can hand out ownership. Recomputing that on every rank at every start
means reading the filter columns of the whole corpus each run. This cache stores
the accepted physical row indexes per file once, keyed by a hash of the file
identities and the filter configuration, and validates them on load.

The cache always lives on a POSIX filesystem — it relies on ``os.replace`` for
atomic publication and on file visibility for cross-rank coordination when the
process group is not up yet. Neither holds on HDFS, so a dataset reading from
HDFS must be given an explicit ``cache_dir`` on shared POSIX storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow.fs as pafs
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from avatar.data.base.fs import file_stat, is_local_filesystem
from avatar.data.base.parquet import parquet_num_rows, read_parquet_columns

__all__ = ["PersistentFilterCache", "SourceFilter"]

FILTER_CACHE_VERSION = 1


@runtime_checkable
class SourceFilter(Protocol):
    """The filter contract the cache can index.

    Implemented by
    :class:`avatar.data.sampler.MultiTaskColumnsFilterSampler`.
    """

    filters: Mapping
    task_name_column: str

    @property
    def filter_columns(self) -> set[str]: ...

    def build_mask(self, data) -> np.ndarray: ...


class PersistentFilterCache:
    """Builds, validates and reads the per-file accepted-row indexes.

    Args:
        files: Canonical, sorted list of parquet files.
        source_filter: The filter whose decisions are being cached.
        filesystem: Filesystem the parquet files live on.
        cache_dir: Parent directory for the cache. Must be POSIX and visible to
            every node. ``None`` derives it from the (local) data directory.
        scan_workers: Threads per participating rank during a cache build.
        scan_ranks: How many global ranks take part in a build. ``None`` uses
            all of them.
        timeout_sec: Ceiling on shared-filesystem coordination waits.
    """

    def __init__(
        self,
        files: Sequence[str],
        source_filter: SourceFilter,
        filesystem: pafs.FileSystem | None = None,
        cache_dir: str | None = None,
        scan_workers: int = 4,
        scan_ranks: int | None = None,
        timeout_sec: float = 3600.0,
    ):
        self.files = list(files)
        self.source_filter = source_filter
        self.filesystem = filesystem
        self.cache_dir = cache_dir
        self.scan_workers = scan_workers
        self.scan_ranks = scan_ranks
        self.timeout_sec = float(timeout_sec)

        if self.cache_dir is None and not self._source_is_local():
            raise ValueError(
                "filter_cache_dir is required when the dataset does not read from "
                "a local filesystem: the cache relies on POSIX atomic renames and "
                "must live on storage shared by every node"
            )

        self._file_to_index = {file: index for index, file in enumerate(self.files)}
        self._cache_path: str | None = None
        self._manifest_path: str | None = None
        self._index_paths: list[str] = []
        self._manifest: dict | None = None
        self._hit = False
        self._effective_scan_ranks = 0

    def _source_is_local(self) -> bool:
        return self.filesystem is None or is_local_filesystem(self.filesystem)

    # ------------------------------------------------------------------ #
    # Public surface                                                      #
    # ------------------------------------------------------------------ #
    @property
    def hit(self) -> bool:
        """Whether the last :meth:`prepare` reused an existing cache."""
        return self._hit

    @property
    def effective_scan_ranks(self) -> int:
        """Ranks that participated in the last build (0 on a cache hit)."""
        return self._effective_scan_ranks

    def index_path(self, file_index: int) -> str:
        """Path of the ``.npy`` holding accepted row indexes for one file."""
        return self._index_paths[file_index]

    def load_indexes(self, file: str) -> np.ndarray:
        """Memory-mapped accepted physical row indexes for ``file``."""
        return np.load(
            self.index_path(self._file_to_index[file]),
            mmap_mode="r",
            allow_pickle=False,
        )

    def count_valid_records(self, file: str) -> int:
        """Accepted-row count for ``file``, validated against the parquet footer."""
        index = self._file_to_index[file]
        return self._validate_index_file(
            self._index_paths[index],
            physical_rows=parquet_num_rows(file, self.filesystem),
        )

    def prepare(self, world_size: int, rank: int) -> np.ndarray:
        """Return per-file accepted-row counts, building the cache if needed."""
        payload = self._cache_payload()
        cache_key = self._initialize_paths(payload)
        os.makedirs(self._cache_path, exist_ok=True)

        distributed_ready = (
            world_size > 1 and dist.is_available() and dist.is_initialized()
        )
        manifest, invalid_reason = self._load_manifest(payload, cache_key)
        all_ranks_hit = manifest is not None
        if distributed_ready:
            hit_tensor = torch.tensor(
                int(all_ranks_hit),
                dtype=torch.int64,
                device=self._reduction_device(),
            )
            dist.all_reduce(hit_tensor, op=dist.ReduceOp.MIN)
            all_ranks_hit = bool(hit_tensor.item())
            if not all_ranks_hit:
                manifest = None
                invalid_reason = "at least one distributed rank reported a cache miss"

        if all_ranks_hit:
            self._hit = True
            self._manifest = manifest
            return np.asarray(
                [entry["valid_rows"] for entry in manifest["files"]], dtype=np.int64
            )

        self._invalidate(invalid_reason, world_size, rank, distributed_ready)
        counts = self._build(payload, world_size, rank, distributed_ready)

        if rank == 0:
            self._write_manifest(payload, cache_key, counts)
        if distributed_ready:
            dist.barrier()
        else:
            self._wait_for(
                lambda: self._load_manifest(payload, cache_key)[0] is not None,
                "completed filter manifest",
            )

        manifest, error = self._load_manifest(payload, cache_key)
        if manifest is None:
            raise RuntimeError(f"Completed filter manifest is invalid: {error}")
        self._manifest = manifest
        return np.asarray(
            [entry["valid_rows"] for entry in manifest["files"]], dtype=np.int64
        )

    # ------------------------------------------------------------------ #
    # Cache identity                                                      #
    # ------------------------------------------------------------------ #
    @classmethod
    def _json_value(cls, value):
        """Convert filter configuration values into stable JSON data."""
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, set | frozenset):
            normalized = [cls._json_value(item) for item in value]
            return sorted(normalized, key=repr)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [cls._json_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return [cls._json_value(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            if isinstance(value, np.datetime64 | np.timedelta64):
                return {"numpy_type": type(value).__name__, "value": str(value)}
            return value.item()
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return {"python_type": type(value).__qualname__, "value": repr(value)}

    def _cache_payload(self) -> dict:
        files = []
        for file in self.files:
            info = file_stat(self.filesystem, file) if self.filesystem else None
            if info is not None:
                size, mtime_ns = int(info.size), int(info.mtime_ns)
                path = info.path
            else:
                stat = os.stat(file)
                size, mtime_ns = stat.st_size, stat.st_mtime_ns
                path = os.path.abspath(file)
            files.append({
                "path": path,
                "size": size,
                "mtime_ns": mtime_ns,
                "physical_rows": parquet_num_rows(file, self.filesystem),
            })
        return {
            "version": FILTER_CACHE_VERSION,
            "files": files,
            "task_name_column": self.source_filter.task_name_column,
            "filters": self._json_value(self.source_filter.filters),
        }

    def _initialize_paths(self, payload: dict) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        cache_key = hashlib.sha256(encoded).hexdigest()
        if self.cache_dir is None:
            common_path = os.path.commonpath([
                os.path.abspath(file) for file in self.files
            ])
            if not os.path.isdir(common_path):
                common_path = os.path.dirname(common_path)
            cache_root = os.path.join(common_path, ".avatar_filter_cache")
        else:
            cache_root = os.path.abspath(self.cache_dir)

        self._cache_path = os.path.join(cache_root, cache_key)
        self._manifest_path = os.path.join(self._cache_path, "manifest.json")
        self._index_paths = [
            os.path.join(self._cache_path, f"{index:08d}.npy")
            for index in range(len(self.files))
        ]
        return cache_key

    # ------------------------------------------------------------------ #
    # Atomic IO + validation                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _atomic_save_numpy(path: str, values: np.ndarray) -> None:
        temporary_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            with open(temporary_path, "wb") as output:
                np.save(output, values, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _atomic_save_json(path: str, value: dict) -> None:
        temporary_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as output:
                json.dump(value, output, sort_keys=True, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _validate_index_file(
        self, path: str, physical_rows: int, valid_rows: int | None = None
    ) -> int:
        try:
            indexes = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot load {path}: {error}") from error
        if indexes.ndim != 1 or indexes.dtype != np.int64:
            raise ValueError(f"{path} must contain a one-dimensional int64 array")
        if valid_rows is not None and len(indexes) != valid_rows:
            raise ValueError(
                f"{path} contains {len(indexes)} rows, expected {valid_rows}"
            )
        if len(indexes) and (indexes[0] < 0 or indexes[-1] >= physical_rows):
            raise ValueError(f"{path} contains out-of-range physical row indexes")
        return len(indexes)

    def _load_manifest(
        self, payload: dict, cache_key: str
    ) -> tuple[dict | None, str | None]:
        if not os.path.exists(self._manifest_path):
            return None, "manifest is missing"
        try:
            with open(self._manifest_path, encoding="utf-8") as source:
                manifest = json.load(source)
            if manifest.get("version") != FILTER_CACHE_VERSION:
                raise ValueError("cache format version changed")
            if manifest.get("cache_key") != cache_key:
                raise ValueError("cache key does not match")
            entries = manifest.get("files")
            if not isinstance(entries, list) or len(entries) != len(self.files):
                raise ValueError("manifest file list does not match the dataset")
            for index, (entry, identity) in enumerate(
                zip(entries, payload["files"], strict=False)
            ):
                valid_rows = entry.get("valid_rows")
                if not isinstance(valid_rows, int) or valid_rows < 0:
                    raise ValueError(f"file {index} has an invalid valid_rows value")
                expected = {
                    "file_index": index,
                    **identity,
                    "index_file": os.path.basename(self._index_paths[index]),
                }
                for key, value in expected.items():
                    if entry.get(key) != value:
                        raise ValueError(f"file {index} metadata changed: {key}")
                self._validate_index_file(
                    self._index_paths[index],
                    physical_rows=identity["physical_rows"],
                    valid_rows=valid_rows,
                )
        except (
            AttributeError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            return None, str(error)
        return manifest, None

    def _write_manifest(
        self, payload: dict, cache_key: str, counts: np.ndarray
    ) -> None:
        entries = []
        for index, identity in enumerate(payload["files"]):
            entries.append({
                "file_index": index,
                **identity,
                "valid_rows": int(counts[index]),
                "index_file": os.path.basename(self._index_paths[index]),
            })
        self._atomic_save_json(
            self._manifest_path,
            {
                "version": FILTER_CACHE_VERSION,
                "cache_key": cache_key,
                "task_name_column": self.source_filter.task_name_column,
                "filters": self._json_value(self.source_filter.filters),
                "files": entries,
            },
        )

    def _wait_for(self, condition, description: str) -> None:
        deadline = time.monotonic() + self.timeout_sec
        while not condition():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {description} in shared filter cache "
                    f"{self._cache_path}"
                )
            time.sleep(0.2)

    # ------------------------------------------------------------------ #
    # Build                                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reduction_device() -> torch.device:
        if dist.get_backend() == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @staticmethod
    def _owned_file_indices(file_count: int, scan_ranks: int, rank: int) -> list[int]:
        """Return the deterministic cache-build assignment for one rank."""
        if rank >= scan_ranks:
            return []
        return list(range(rank, file_count, scan_ranks))

    def _build_index(self, file_index: int, physical_rows: int) -> int:
        file = self.files[file_index]
        table = read_parquet_columns(
            file, sorted(self.source_filter.filter_columns), self.filesystem
        )
        if table.num_rows != physical_rows:
            raise RuntimeError(
                f"{file} metadata reports {physical_rows} rows, "
                f"but the filter scan read {table.num_rows}"
            )
        mask = self.source_filter.build_mask(table)
        if mask.ndim != 1 or len(mask) != physical_rows:
            raise RuntimeError(
                f"Filter mask for {file} has invalid shape {mask.shape}; "
                f"expected ({physical_rows},)"
            )
        valid_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        self._atomic_save_numpy(self._index_paths[file_index], valid_indices)
        return len(valid_indices)

    def _invalidate(
        self,
        invalid_reason: str | None,
        world_size: int,
        rank: int,
        distributed_ready: bool,
    ) -> None:
        if invalid_reason == "manifest is missing":
            return
        warnings.warn(
            f"Rebuilding invalid filter cache {self._cache_path}: {invalid_reason}",
            stacklevel=2,
        )
        if rank == 0 and os.path.exists(self._manifest_path):
            invalid_path = f"{self._manifest_path}.invalid.{time.time_ns()}"
            try:
                os.replace(self._manifest_path, invalid_path)
            except FileNotFoundError:
                pass
        if distributed_ready:
            dist.barrier()
        elif world_size > 1 and rank != 0:
            self._wait_for(
                lambda: not os.path.exists(self._manifest_path),
                "invalid manifest removal",
            )

    def _build(
        self, payload: dict, world_size: int, rank: int, distributed_ready: bool
    ) -> np.ndarray:
        scan_ranks = min(
            self.scan_ranks if self.scan_ranks is not None else world_size, world_size
        )
        self._effective_scan_ranks = scan_ranks
        owned_indices = self._owned_file_indices(len(self.files), scan_ranks, rank)
        if rank == 0:
            warnings.warn(
                f"[filter_cache] cache miss: {scan_ranks}/{world_size} global "
                f"ranks will scan {len(self.files)} files with up to "
                f"{self.scan_workers} threads per participating rank",
                stacklevel=2,
            )

        local_counts = np.zeros(len(self.files), dtype=np.int64)
        if owned_indices:
            with ThreadPoolExecutor(
                max_workers=min(self.scan_workers, len(owned_indices))
            ) as executor:
                futures = {
                    executor.submit(
                        self._build_index,
                        index,
                        payload["files"][index]["physical_rows"],
                    ): index
                    for index in owned_indices
                }
                completed = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Building persistent filter indexes",
                    disable=(rank != 0),
                )
                for future in completed:
                    local_counts[futures[future]] = future.result()

        if world_size > 1 and not distributed_ready and rank == 0:
            warnings.warn(
                "[filter_cache] torch.distributed is not initialized during dataset "
                "construction; ranks are coordinating through the cache directory. "
                "filter_cache_dir must be on storage shared by every node.",
                stacklevel=2,
            )
        if distributed_ready:
            counts_tensor = torch.as_tensor(
                local_counts, device=self._reduction_device()
            )
            dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
            counts = counts_tensor.cpu().numpy().astype(np.int64, copy=False)
            dist.barrier()
        else:
            self._wait_for(
                lambda: all(os.path.exists(path) for path in self._index_paths),
                "all per-file filter indexes",
            )
            counts = np.asarray(
                [
                    self._validate_index_file(
                        path, physical_rows=payload["files"][index]["physical_rows"]
                    )
                    for index, path in enumerate(self._index_paths)
                ],
                dtype=np.int64,
            )
        return counts
