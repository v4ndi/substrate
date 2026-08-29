import hashlib
import json
import os
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

from avatar.data.parquet import read_parquet_file
from avatar.data.sampler import BaseSampler
from avatar.data.sampler.filter_sampler import MultiTaskColumnsFilterSampler

from .parquet_dataset import IterDataset


class TabularDataset(IterDataset):
    """Tabular dataset

    Args:
        path: str - path to directory with parquet files. don't pass path to parquet file directly
        read_columns: Optional[list[str] - columns to read
        shuffle_files: bool - shuffle order of reading files or not
        shuffle_pq: bool - shuffle order of reading rows from parquet file or not
        hidden_state_columns: Optional[Union[str, List[str]]] - column name or names of hidden_state in parquet schema
        hidden_state_column: Optional[str] - column name of hidden_state in parquet schema
        sampler: structure with __iter__() and set_dataset_iterator methods
                    use this iterator to sample data

    Methods:
        check_tabular_features(): check parquet schema for presence of tabular features
        process_tabular(record, hidden_state_column): process tabular features to torch.tensor
        process(record): process each record in parquet file
        lazy_process: bool - if True, process_tabular() is called lazily only in collate_fn.
            It works faster for multi-gpu training.
    """

    def __init__(
        self,
        path: str | list,
        read_columns: list[str] | None = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        hidden_state_column: str | None = None,
        hidden_state_columns: str | list[str] | None = None,
        lazy_process: bool = False,
        sampler: BaseSampler = None,
        **kwargs,
    ):
        assert not (
            hidden_state_column is not None and hidden_state_columns is not None
        ), "Only one of hidden_state_column and hidden_state_columns can be specified"
        super().__init__(
            path=path,
            read_columns=read_columns,
            shuffle_files=shuffle_files,
            shuffle_pq=shuffle_pq,
            sampler=sampler,
            **kwargs,
        )
        if hidden_state_column is not None:
            hidden_state_columns = [hidden_state_column]
        elif hidden_state_columns is not None and isinstance(hidden_state_columns, str):
            hidden_state_columns = [hidden_state_columns]

        self.hidden_state_columns = hidden_state_columns

        self.lazy_process = lazy_process
        assert self.check_tabular_features(), (
            "Not found tabular features in parquet schema or hidden_state_column."
        )

    def check_tabular_features(self) -> bool:
        """Check parquet schema for presence of tabular features"""
        schema = pq.read_schema(self.files[0])
        if (
            "cat_features" in schema.names
            or "num_features" in schema.names
            or all([
                hidden_state_col in schema.names
                for hidden_state_col in self.hidden_state_columns
            ])
        ):
            return True
        else:
            return False

    @staticmethod
    def process_tabular(record, hidden_state_columns: list[str] | None = None):
        """Process tabular features to torch.tensor"""
        tabular_features = {
            "cat_features": None,
            "num_features": None,
        }

        if "cat_features" in record:
            tabular_features["cat_features"] = torch.from_numpy(
                np.array(record["cat_features"])
            ).long()
            del record["cat_features"]

        if "num_features" in record:
            tabular_features["num_features"] = torch.from_numpy(
                np.array(record["num_features"])
            ).float()
            del record["num_features"]

        if hidden_state_columns is not None:
            tabular_features["_hidden_states"] = {}
            for hidden_state_col in hidden_state_columns:
                tabular_features["_hidden_states"][hidden_state_col] = torch.from_numpy(
                    np.array(record[hidden_state_col])
                ).float()
                del record[hidden_state_col]

        record["tab_features"] = tabular_features

        return record

    def process(self, record: dict):
        if not self.lazy_process:
            record = TabularDataset.process_tabular(
                record=record, hidden_state_columns=self.hidden_state_columns
            )
            return record
        else:
            return partial(
                TabularDataset.process_tabular,
                record=record,
                hidden_state_columns=self.hidden_state_columns,
            )


class ShardTabularDataset(IterDataset):
    """Shard Tabular dataset

    Args:
        path: str - path to directory with parquet files. don't pass path to parquet file directly
        read_columns: Optional[list[str] - columns to read
        shuffle_files: bool - shuffle order of reading files or not
        shuffle_pq: bool - shuffle order of reading rows from parquet file or not
        hidden_state_columns: Optional[Union[str, List[str]]] - column name or names of hidden_state in parquet schema
        hidden_state_column: Optional[str] - column name of hidden_state in parquet schema
        sampler: structure with __iter__() and set_dataset_iterator methods
                    use this iterator to sample data
        shard_by_rank: split the filtered global record stream across DDP ranks
        drop_tail: drop the global remainder so every rank has equal cardinality
        rotate_tail: rotate logical rank ownership by the remainder each epoch
        scan_workers: number of threads used to count valid records per file
        scan_ranks: number of global ranks that build a missing filter cache
        filter_cache: persist valid physical row indexes for filtered sharding
        filter_cache_dir: shared parent directory for persistent filter indexes
        filter_cache_timeout_sec: maximum shared-filesystem coordination wait

    Methods:
        check_tabular_features(): check parquet schema for presence of tabular features
        process_tabular(record, hidden_state_column): process tabular features to torch.tensor
        process(record): process each record in parquet file
        lazy_process: bool - if True, process_tabular() is called lazily only in collate_fn.
            It works faster for multi-gpu training.
    """

    _FILTER_CACHE_VERSION = 1
    _SOURCE_ROWS = 0
    _PARQUET_BYTES = 1
    _FILTER_NANOSECONDS = 2

    def __init__(
        self,
        path: str | list,
        read_columns: list[str] | None = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        hidden_state_column: str | None = None,
        hidden_state_columns: str | list[str] | None = None,
        lazy_process: bool = False,
        sampler: BaseSampler = None,
        shard_by_rank: bool = False,
        drop_tail: bool = True,
        rotate_tail: bool = False,
        scan_workers: int = 4,
        scan_ranks: int | None = None,
        filter_cache: bool = True,
        filter_cache_dir: str | None = None,
        filter_cache_timeout_sec: float = 3600.0,
        **kwargs,
    ):
        assert not (
            hidden_state_column is not None and hidden_state_columns is not None
        ), "Only one of hidden_state_column and hidden_state_columns can be specified"
        self.shard_by_rank = shard_by_rank
        self.drop_tail = drop_tail
        self.rotate_tail = rotate_tail
        self.scan_workers = scan_workers
        self.scan_ranks = scan_ranks
        self.filter_cache = filter_cache
        self.filter_cache_dir = filter_cache_dir
        self.filter_cache_timeout_sec = float(filter_cache_timeout_sec)
        if self.scan_workers <= 0:
            raise ValueError("scan_workers must be greater than zero")
        if self.scan_ranks is not None and self.scan_ranks <= 0:
            raise ValueError("scan_ranks must be greater than zero or None")
        if self.filter_cache_timeout_sec <= 0:
            raise ValueError("filter_cache_timeout_sec must be greater than zero")
        self._shuffle_files = shuffle_files

        # Sharded ownership is calculated from a canonical file order. File
        # shuffling is applied to the already-owned segments during iteration.
        super().__init__(
            path=path,
            read_columns=read_columns,
            shuffle_files=shuffle_files if not shard_by_rank else False,
            shuffle_pq=shuffle_pq,
            sampler=sampler,
            **kwargs,
        )
        if self.shard_by_rank:
            self.files.sort()

        self._source_filter = (
            sampler if isinstance(sampler, MultiTaskColumnsFilterSampler) else None
        )
        self._use_filter_cache = bool(
            self.shard_by_rank and self._source_filter is not None and filter_cache
        )
        self._source_only_columns: set[str] = set()
        self._source_read_columns = self.read_columns
        if (
            self._source_filter is not None
            and not self._use_filter_cache
            and self.read_columns is not None
        ):
            self._source_only_columns = self._source_filter.filter_columns - set(
                self.read_columns
            )
            self._source_read_columns = [
                *self.read_columns,
                *sorted(self._source_only_columns),
            ]
        self._file_counts: np.ndarray | None = None
        self._file_to_index = {file: index for index, file in enumerate(self.files)}
        self._filter_cache_path: str | None = None
        self._filter_manifest_path: str | None = None
        self._filter_index_paths: list[str] = []
        self._filter_manifest: dict | None = None
        self._filter_cache_hit = False
        self._epoch_state = torch.zeros(1, dtype=torch.int64)
        if self.rotate_tail:
            # Make set_epoch() visible to persistent DataLoader workers.
            self._epoch_state.share_memory_()
        self._scan_duration_sec = 0.0
        self._epoch_metric_counters: torch.Tensor | None = None
        self.total_parquet_bytes = sum(os.path.getsize(file) for file in self.files)
        self._world_size, self._rank = self._resolve_dist_info()
        self._effective_scan_ranks = min(
            self.scan_ranks if self.scan_ranks is not None else self._world_size,
            self._world_size,
        )
        if self.shard_by_rank and self._world_size == 1:
            warnings.warn(
                "[shard_by_rank] world_size resolved to 1 at dataset construction "
                "time. If this is a distributed run, initialize the process group "
                "or launch with WORLD_SIZE/RANK before building the dataset; otherwise "
                "every rank will read the full dataset.",
                stacklevel=2,
            )
        if hidden_state_column is not None:
            hidden_state_columns = [hidden_state_column]
        elif hidden_state_columns is not None and isinstance(hidden_state_columns, str):
            hidden_state_columns = [hidden_state_columns]

        self.hidden_state_columns = hidden_state_columns

        self.lazy_process = lazy_process
        assert self.check_tabular_features(), (
            "Not found tabular features in parquet schema or hidden_state_column."
        )
        if self.shard_by_rank:
            self._scan_files()

    @staticmethod
    def _resolve_dist_info() -> tuple[int, int]:
        """Return ``(world_size, rank)`` from torch or launcher variables."""
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))

    def _get_environment_info(self) -> tuple[int, int, int, int]:
        """Return DataLoader worker and distributed rank information."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers, worker_id = worker_info.num_workers, worker_info.id

        world_size, rank = self._world_size, self._rank
        current_world_size, current_rank = self._resolve_dist_info()
        if (world_size, rank) == (1, 0) or current_world_size > 1:
            world_size, rank = current_world_size, current_rank
        return num_workers, worker_id, world_size, rank

    @classmethod
    def _cache_json_value(cls, value):
        """Convert filter configuration values into stable JSON data."""
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, Mapping):
            return {
                str(key): cls._cache_json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, set | frozenset):
            normalized = [cls._cache_json_value(item) for item in value]
            return sorted(normalized, key=repr)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [cls._cache_json_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return [cls._cache_json_value(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            if isinstance(value, np.datetime64 | np.timedelta64):
                return {"numpy_type": type(value).__name__, "value": str(value)}
            return value.item()
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return {"python_type": type(value).__qualname__, "value": repr(value)}

    def _filter_cache_payload(self) -> dict:
        files = []
        for file in self.files:
            stat = os.stat(file)
            files.append({
                "path": os.path.abspath(file),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "physical_rows": pq.ParquetFile(file).metadata.num_rows,
            })
        return {
            "version": self._FILTER_CACHE_VERSION,
            "files": files,
            "task_name_column": self._source_filter.task_name_column,
            "filters": self._cache_json_value(self._source_filter.filters),
        }

    def _initialize_filter_cache_paths(self, payload: dict) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        cache_key = hashlib.sha256(encoded).hexdigest()
        if self.filter_cache_dir is None:
            common_path = os.path.commonpath([
                os.path.abspath(file) for file in self.files
            ])
            if not os.path.isdir(common_path):
                common_path = os.path.dirname(common_path)
            cache_root = os.path.join(common_path, ".avatar_filter_cache")
        else:
            cache_root = os.path.abspath(self.filter_cache_dir)

        self._filter_cache_path = os.path.join(cache_root, cache_key)
        self._filter_manifest_path = os.path.join(
            self._filter_cache_path, "manifest.json"
        )
        self._filter_index_paths = [
            os.path.join(self._filter_cache_path, f"{index:08d}.npy")
            for index in range(len(self.files))
        ]
        return cache_key

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

    @staticmethod
    def _owned_filter_file_indices(
        file_count: int, scan_ranks: int, rank: int
    ) -> list[int]:
        """Return the deterministic cache-build assignment for one rank."""
        if rank >= scan_ranks:
            return []
        return list(range(rank, file_count, scan_ranks))

    def _load_filter_manifest(
        self, payload: dict, cache_key: str
    ) -> tuple[dict | None, str | None]:
        if not os.path.exists(self._filter_manifest_path):
            return None, "manifest is missing"
        try:
            with open(self._filter_manifest_path, encoding="utf-8") as source:
                manifest = json.load(source)
            if manifest.get("version") != self._FILTER_CACHE_VERSION:
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
                    "index_file": os.path.basename(self._filter_index_paths[index]),
                }
                for key, value in expected.items():
                    if entry.get(key) != value:
                        raise ValueError(f"file {index} metadata changed: {key}")
                self._validate_index_file(
                    self._filter_index_paths[index],
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

    def _read_filter_table(self, file: str):
        """Read only columns required to build a persistent filter index."""
        return ds.dataset(file, format="parquet", partitioning="hive").to_table(
            use_threads=False,
            columns=sorted(self._source_filter.filter_columns),
        )

    def _build_filter_index(self, file_index: int, physical_rows: int) -> int:
        table = self._read_filter_table(self.files[file_index])
        if table.num_rows != physical_rows:
            raise RuntimeError(
                f"{self.files[file_index]} metadata reports {physical_rows} rows, "
                f"but the filter scan read {table.num_rows}"
            )
        mask = self._source_filter.build_mask(table)
        if mask.ndim != 1 or len(mask) != physical_rows:
            raise RuntimeError(
                f"Filter mask for {self.files[file_index]} has invalid shape "
                f"{mask.shape}; expected ({physical_rows},)"
            )
        valid_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        self._atomic_save_numpy(self._filter_index_paths[file_index], valid_indices)
        return len(valid_indices)

    def _wait_for_cache_condition(self, condition, description: str) -> None:
        deadline = time.monotonic() + self.filter_cache_timeout_sec
        while not condition():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {description} in shared filter cache "
                    f"{self._filter_cache_path}"
                )
            time.sleep(0.2)

    def _write_filter_manifest(
        self, payload: dict, cache_key: str, counts: np.ndarray
    ) -> None:
        entries = []
        for index, identity in enumerate(payload["files"]):
            entries.append({
                "file_index": index,
                **identity,
                "valid_rows": int(counts[index]),
                "index_file": os.path.basename(self._filter_index_paths[index]),
            })
        self._atomic_save_json(
            self._filter_manifest_path,
            {
                "version": self._FILTER_CACHE_VERSION,
                "cache_key": cache_key,
                "task_name_column": self._source_filter.task_name_column,
                "filters": self._cache_json_value(self._source_filter.filters),
                "files": entries,
            },
        )

    @staticmethod
    def _distributed_reduction_device() -> torch.device:
        if dist.get_backend() == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _prepare_filter_cache(self) -> np.ndarray:
        payload = self._filter_cache_payload()
        cache_key = self._initialize_filter_cache_paths(payload)
        os.makedirs(self._filter_cache_path, exist_ok=True)

        world_size, rank = self._world_size, self._rank
        distributed_ready = (
            world_size > 1 and dist.is_available() and dist.is_initialized()
        )
        manifest, invalid_reason = self._load_filter_manifest(payload, cache_key)
        all_ranks_hit = manifest is not None
        if distributed_ready:
            hit_tensor = torch.tensor(
                int(all_ranks_hit),
                dtype=torch.int64,
                device=self._distributed_reduction_device(),
            )
            dist.all_reduce(hit_tensor, op=dist.ReduceOp.MIN)
            all_ranks_hit = bool(hit_tensor.item())
            if not all_ranks_hit:
                manifest = None
                invalid_reason = "at least one distributed rank reported a cache miss"

        if all_ranks_hit:
            self._filter_cache_hit = True
            self._filter_manifest = manifest
            return np.asarray(
                [entry["valid_rows"] for entry in manifest["files"]],
                dtype=np.int64,
            )

        if invalid_reason != "manifest is missing":
            warnings.warn(
                f"Rebuilding invalid filter cache {self._filter_cache_path}: "
                f"{invalid_reason}",
                stacklevel=2,
            )
            if rank == 0 and os.path.exists(self._filter_manifest_path):
                invalid_path = f"{self._filter_manifest_path}.invalid.{time.time_ns()}"
                try:
                    os.replace(self._filter_manifest_path, invalid_path)
                except FileNotFoundError:
                    pass
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            elif world_size > 1 and rank != 0:
                self._wait_for_cache_condition(
                    lambda: not os.path.exists(self._filter_manifest_path),
                    "invalid manifest removal",
                )

        scan_ranks = min(
            self.scan_ranks if self.scan_ranks is not None else world_size,
            world_size,
        )
        self._effective_scan_ranks = scan_ranks
        owned_indices = self._owned_filter_file_indices(
            len(self.files), scan_ranks, rank
        )
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
                        self._build_filter_index,
                        index,
                        payload["files"][index]["physical_rows"],
                    ): index
                    for index in owned_indices
                }
                completed = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Building persistent tabular filter indexes",
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
                local_counts, device=self._distributed_reduction_device()
            )
            dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
            counts = counts_tensor.cpu().numpy().astype(np.int64, copy=False)
            dist.barrier()
        else:
            self._wait_for_cache_condition(
                lambda: all(os.path.exists(path) for path in self._filter_index_paths),
                "all per-file filter indexes",
            )
            counts = np.asarray(
                [
                    self._validate_index_file(
                        path,
                        physical_rows=payload["files"][index]["physical_rows"],
                    )
                    for index, path in enumerate(self._filter_index_paths)
                ],
                dtype=np.int64,
            )

        if rank == 0:
            self._write_filter_manifest(payload, cache_key, counts)
        if distributed_ready:
            dist.barrier()
        else:
            self._wait_for_cache_condition(
                lambda: self._load_filter_manifest(payload, cache_key)[0] is not None,
                "completed filter manifest",
            )

        manifest, error = self._load_filter_manifest(payload, cache_key)
        if manifest is None:
            raise RuntimeError(f"Completed filter manifest is invalid: {error}")
        self._filter_manifest = manifest
        return np.asarray(
            [entry["valid_rows"] for entry in manifest["files"]], dtype=np.int64
        )

    def _count_valid_records(self, file: str) -> int:
        """Count rows emitted by ``file`` before rank/worker slicing."""
        if self._use_filter_cache and self._filter_index_paths:
            index = self._file_to_index[file]
            return self._validate_index_file(
                self._filter_index_paths[index],
                physical_rows=pq.ParquetFile(file).metadata.num_rows,
            )
        if self._source_filter is None:
            return pq.ParquetFile(file).metadata.num_rows

        filter_columns = sorted(self._source_filter.filter_columns)
        return sum(
            self._source_filter.accepts(record)
            for record in read_parquet_file(
                file=file,
                columns=filter_columns,
                shuffle=False,
            )
        )

    def _scan_files(self) -> None:
        """Cache valid row counts used to construct the global prefix sum."""
        scan_start = time.perf_counter()
        if self._use_filter_cache:
            counts = self._prepare_filter_cache()
        else:
            counts = np.empty(len(self.files), dtype=np.int64)
            with ThreadPoolExecutor(max_workers=self.scan_workers) as executor:
                future_to_index = {
                    executor.submit(self._count_valid_records, file): index
                    for index, file in enumerate(self.files)
                }
                completed = tqdm(
                    as_completed(future_to_index),
                    total=len(self.files),
                    desc="Scanning parquet files, preparing sharded tabular dataset",
                )
                for future in completed:
                    counts[future_to_index[future]] = future.result()
        self._file_counts = counts
        self._scan_duration_sec = time.perf_counter() - scan_start

        total = int(self._file_counts.sum())
        world_size, _ = self._resolve_dist_info()
        per_rank = total // world_size
        tail = total % world_size
        warnings.warn(
            "\n"
            f"[shard_by_rank] files: {len(self.files)}, records per file: "
            f"min {int(self._file_counts.min())}, "
            f"mean {self._file_counts.mean():.1f}, "
            f"std {self._file_counts.std():.1f}, "
            f"max {int(self._file_counts.max())}\n"
            f"[shard_by_rank] records: total {total}, per rank {per_rank}, "
            f"tail {tail} "
            f"({'dropped' if self.drop_tail else 'to the last rank'}), "
            f"epoch rotation {'enabled' if self.rotate_tail else 'disabled'}"
            + (
                (
                    ", filter cache hit"
                    if self._filter_cache_hit
                    else (
                        f", filter cache built using "
                        f"{self._effective_scan_ranks}/{world_size} scan ranks"
                    )
                )
                if self._use_filter_cache
                else ""
            ),
            stacklevel=2,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the logical shard rotation epoch for this dataset and its workers."""
        if epoch < 0:
            raise ValueError("epoch must be greater than or equal to zero")
        self._epoch_state[0] = int(epoch)

    @staticmethod
    def _split_cyclic_range(
        start: int, length: int, total: int
    ) -> list[tuple[int, int]]:
        """Split a cyclic logical range into at most two ordinary ranges."""
        if length <= 0:
            return []
        if total <= 0:
            raise ValueError("total must be positive for a non-empty cyclic range")
        if length > total:
            raise ValueError("cyclic range length cannot exceed total")

        start %= total
        stop = start + length
        if stop <= total:
            return [(start, stop)]
        return [(start, total), (0, stop - total)]

    def _epoch_offset(self, total: int, world_size: int) -> int:
        """Return a deterministic rotation in the global valid-record stream."""
        tail = total % world_size
        if not self.rotate_tail or tail == 0 or total == 0:
            return 0
        return (int(self._epoch_state[0].item()) * tail) % total

    def _range_to_segments(
        self, start: int, stop: int, cum: np.ndarray
    ) -> list[tuple[str, int, int]]:
        """Map one non-cyclic global logical range to filtered file slices."""
        first = max(int(np.searchsorted(cum, start, side="right")) - 1, 0)
        last = min(
            int(np.searchsorted(cum, stop, side="left")),
            len(self._file_counts),
        )
        segments = []
        for index in range(first, last):
            lo = max(start, int(cum[index])) - int(cum[index])
            hi = min(stop, int(cum[index + 1])) - int(cum[index])
            if hi > lo:
                segments.append((self.files[index], lo, hi))
        return segments

    def configure_epoch_metrics(self, num_workers: int) -> None:
        """Allocate per-worker shared counters for optional benchmark metrics."""
        worker_slots = max(int(num_workers), 1)
        self._epoch_metric_counters = torch.zeros(
            (worker_slots, 3), dtype=torch.int64
        ).share_memory_()

    def reset_epoch_metrics(self) -> None:
        """Reset counters after the previous DataLoader iterator has completed."""
        if self._epoch_metric_counters is not None:
            self._epoch_metric_counters.zero_()

    def get_epoch_metrics(self) -> dict[str, float]:
        """Return counters accumulated by this rank's DataLoader workers."""
        if self._epoch_metric_counters is None:
            return {
                "source_rows_scanned": 0.0,
                "parquet_bytes_opened": 0.0,
                "filter_sec": 0.0,
            }
        totals = self._epoch_metric_counters.sum(dim=0).tolist()
        max_worker_filter_nanoseconds = (
            self._epoch_metric_counters[:, self._FILTER_NANOSECONDS].max().item()
        )
        return {
            "source_rows_scanned": float(totals[self._SOURCE_ROWS]),
            "parquet_bytes_opened": float(totals[self._PARQUET_BYTES]),
            "filter_sec": float(max_worker_filter_nanoseconds) / 1e9,
        }

    def _record_file_metrics(
        self, source_rows: int, parquet_bytes: int, filter_nanoseconds: int
    ) -> None:
        if self._epoch_metric_counters is None:
            return
        worker_info = torch.utils.data.get_worker_info()
        worker_slot = worker_info.id if worker_info is not None else 0
        if worker_slot >= len(self._epoch_metric_counters):
            raise RuntimeError(
                "TabularDataset epoch metrics were configured for fewer workers "
                "than the DataLoader uses"
            )
        counters = self._epoch_metric_counters[worker_slot]
        counters[self._SOURCE_ROWS] += source_rows
        counters[self._PARQUET_BYTES] += parquet_bytes
        counters[self._FILTER_NANOSECONDS] += filter_nanoseconds

    def __len__(self) -> int:
        if not self.shard_by_rank:
            return super().__len__()
        if self._file_counts is None:
            raise RuntimeError("Sharded file counts have not been initialized")

        _, _, world_size, rank = self._get_environment_info()
        total = int(self._file_counts.sum())
        tail = (
            total % world_size if not self.drop_tail and rank == world_size - 1 else 0
        )
        return total // world_size + tail

    def _cumsum_segments(self) -> list[tuple[str, int, int]]:
        """Return filtered-row ``(file, lo, hi)`` segments for this worker."""
        if self._file_counts is None:
            raise RuntimeError(
                "Rank sharding requires a valid-record scan; set shard_by_rank=True"
            )

        num_workers, worker_id, world_size, rank = self._get_environment_info()
        cum = np.concatenate([[0], np.cumsum(self._file_counts)])
        total = int(cum[-1])
        per_rank = total // world_size
        rank_count = per_rank
        if not self.drop_tail and rank == world_size - 1:
            rank_count += total % world_size
        base, remainder = divmod(rank_count, num_workers)
        worker_relative_start = worker_id * base + min(worker_id, remainder)
        worker_count = base + int(worker_id < remainder)

        if total == 0 or worker_count == 0:
            return []
        worker_start = (
            self._epoch_offset(total, world_size)
            + rank * per_rank
            + worker_relative_start
        ) % total

        segments = []
        for start, stop in self._split_cyclic_range(
            start=worker_start,
            length=worker_count,
            total=total,
        ):
            segments.extend(self._range_to_segments(start, stop, cum))
        return segments

    def _read_owned_records(self, file: str, lo: int, hi: int) -> list[dict]:
        """Select an owned logical slice using cached or row-wise membership."""
        owned_records = []
        source_rows = 0
        filter_nanoseconds = 0
        collect_metrics = self._epoch_metric_counters is not None
        if self._use_filter_cache:
            file_index = self._file_to_index[file]
            valid_indices = np.load(
                self._filter_index_paths[file_index], mmap_mode="r", allow_pickle=False
            )
            if len(valid_indices) < hi:
                raise RuntimeError(
                    f"[shard_by_rank] cached index for {file} contains "
                    f"{len(valid_indices)} valid records, but ownership requires {hi}"
                )
            physical_indices = valid_indices[lo:hi]
            next_owned = 0
            for physical_index, record in enumerate(
                read_parquet_file(
                    file=file,
                    columns=self._source_read_columns,
                    shuffle=False,
                )
            ):
                source_rows += 1
                if (
                    next_owned < len(physical_indices)
                    and physical_index == physical_indices[next_owned]
                ):
                    owned_records.append(record)
                    next_owned += 1
            if next_owned != len(physical_indices):
                raise RuntimeError(
                    f"[shard_by_rank] {file} ended before cached physical row "
                    f"{physical_indices[next_owned]} could be read"
                )
            valid_index = len(valid_indices)
        else:
            valid_index = 0
            for record in read_parquet_file(
                file=file,
                columns=self._source_read_columns,
                shuffle=False,
            ):
                source_rows += 1
                if self._source_filter is not None:
                    if collect_metrics:
                        filter_start = time.perf_counter_ns()
                        accepted = self._source_filter.accepts(record)
                        filter_nanoseconds += time.perf_counter_ns() - filter_start
                    else:
                        accepted = self._source_filter.accepts(record)
                    if not accepted:
                        continue
                if lo <= valid_index < hi:
                    for column in self._source_only_columns:
                        del record[column]
                    owned_records.append(record)
                valid_index += 1

        self._record_file_metrics(
            source_rows=source_rows,
            parquet_bytes=os.path.getsize(file),
            filter_nanoseconds=filter_nanoseconds,
        )

        if valid_index < hi:
            raise RuntimeError(
                f"[shard_by_rank] {file} yielded {valid_index} valid records during "
                f"iteration, but the scan counted at least {hi}; iteration filtering "
                "diverged from _count_valid_records()."
            )
        return owned_records

    def _iter_sharded_impl(self) -> Iterator[dict]:
        segments = self._cumsum_segments()
        if self._shuffle_files and len(segments) > 1:
            np.random.shuffle(segments)

        for file, lo, hi in segments:
            records = self._read_owned_records(file=file, lo=lo, hi=hi)
            if self.shuffle_pq and len(records) > 1:
                np.random.shuffle(records)
            for record in records:
                processed_record = self.process(record)
                if processed_record is not None:
                    yield processed_record

    def __iter__(self) -> Iterator[dict]:
        if not self.shard_by_rank:
            return super().__iter__()

        iterator = self._iter_sharded_impl()
        # MultiTaskColumnsFilterSampler already defined source membership and
        # must not be executed a second time. Other samplers retain the legacy
        # behavior of wrapping processed records.
        if self.sampler is None or self.sampler is self._source_filter:
            return iterator
        self.sampler.set_dataset_iterator(iterator)
        return iter(self.sampler)

    def check_tabular_features(self) -> bool:
        """Check parquet schema for presence of tabular features"""
        schema = pq.read_schema(self.files[0])
        if (
            "cat_features" in schema.names
            or "num_features" in schema.names
            or all([
                hidden_state_col in schema.names
                for hidden_state_col in self.hidden_state_columns
            ])
        ):
            return True
        else:
            return False

    @staticmethod
    def process_tabular(record, hidden_state_columns: list[str] | None = None):
        """Process tabular features to torch.tensor"""
        tabular_features = {
            "cat_features": None,
            "num_features": None,
        }

        if "cat_features" in record:
            tabular_features["cat_features"] = torch.from_numpy(
                np.array(record["cat_features"])
            ).long()
            del record["cat_features"]

        if "num_features" in record:
            tabular_features["num_features"] = torch.from_numpy(
                np.array(record["num_features"])
            ).float()
            del record["num_features"]

        if hidden_state_columns is not None:
            tabular_features["_hidden_states"] = {}
            for hidden_state_col in hidden_state_columns:
                tabular_features["_hidden_states"][hidden_state_col] = torch.from_numpy(
                    np.array(record[hidden_state_col])
                ).float()
                del record[hidden_state_col]

        record["tab_features"] = tabular_features

        return record

    def process(self, record: dict):
        if not self.lazy_process:
            record = TabularDataset.process_tabular(
                record=record, hidden_state_columns=self.hidden_state_columns
            )
            return record
        else:
            return partial(
                TabularDataset.process_tabular,
                record=record,
                hidden_state_columns=self.hidden_state_columns,
            )
