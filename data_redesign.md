# Design: `avatar/data` restructure — `base` / `sequential` / `tabular`, one sharding engine, HDFS

Status: **draft / proposal** (2026-08-29). Companion to `train_redesign.md` — land this
first, before the `train.py` / `inference.py` rewrite.

Three asks:

1. Split the flat module into `avatar/data/{base,sequential,tabular}` (mirrors the
   `avatar/nn/embedding` and `avatar/nn/sequential` refactors).
2. Keep **only** sharded datasets — one dataset class per modality that always
   distributes the record stream correctly across processes. `ShardEventSequenceDataset`
   is currently incorrect; the sharding logic must be unified and moved to `base`,
   then reused by both the tabular and sequence datasets.
3. Add HDFS read support to both datasets.

---

## 0. Current state

```
avatar/data/
├── __init__.py                     re-exports EventSequenceBatch, TabularBatch, dataset, parquet
├── parquet.py                      read_parquet_file / parquet_num_rows  (LOCAL FS ONLY)
├── event_seq_batch.py              EventSequenceBatch
├── tabular_batch.py                TabularBatch, UpliftTabularBatch, move_to_device
├── dataset/
│   ├── parquet_dataset.py          BaseIterDataset, IterDataset      (glob(), worker split)
│   ├── tabular_dataset.py          TabularDataset, ShardTabularDataset  (1046 LOC)
│   ├── sequence_dataset.py         EventSequenceDataset, ShardEventSequenceDataset
│   └── collate_fn/                 Base / EventSequence / FixedHorizon / Tabular(+4)
└── sampler/                        BaseSampler + 8 samplers (5 dead)
```

### What actually gets used (config sweep)

| class | `_target_` in configs | instantiated in `avatar/` |
|---|---|---|
| `TabularDataset` | **86** (~55 files) | never (Hydra only) |
| `EventSequenceDataset` | 7 (4 files) | never |
| `ShardTabularDataset` | **0** | **0** — defined, re-exported, never referenced anywhere |
| `ShardEventSequenceDataset` | **0** (1 mention in a `docs/` md) | **0** |
| `TabularCollateFn` / `UpliftCollateFn` | 41 / 45 | — |
| `EventSequenceCollateFn` | 7 | — |
| `ColumnFilterSampler` | 6 | — |
| `MultiTaskColumnsFilterSampler` | 0 | used in an `isinstance` branch of tabular sharding |
| `SupervisedCollateFn`, `MultiTask*CollateFn`, `FixedHorizonCollateFn`, `ColesCollateFn`, `AccumulateSampler`, `ColumnsFilterSampler`, `StreamingBalanced*Sampler`, `ScheduleConversionSampler` | 0 | 0 |

**Consequence:** the "correct" sharded path (`ShardTabularDataset`, `shard_by_rank=True`)
is fully built but **no config uses it** — every training run today uses the
non-sharded `TabularDataset` / `EventSequenceDataset` and relies on
`accelerator.prepare(dataloader)` to split the stream. `train.py:294` already has the
branch that *would* self-shard if a dataset exposed `shard_by_rank=True`. The ask
formalises this: make the single per-modality class always shard, delete the fork.

### No external Python consumers

Outside `avatar/data/` and `tests/`, the only imports of this package are the batch
containers (`EventSequenceBatch`, `TabularBatch`) in `avatar/nn/**` and
`avatar/pipeline/**`. Every dataset / collate / sampler is reached through a Hydra
`_target_` string. So the blast radius is **configs + tests**, not source imports.

---

## 1. Why `ShardEventSequenceDataset` is incorrect

`ShardTabularDataset` shards the **global record stream** via a prefix-sum over
per-file valid-record counts. `ShardEventSequenceDataset` shards **files**, and the
implementation has at least six defects:

1. **File-granular split, not record-granular.**
   `files_per_worker()` (`sequence_dataset.py:494`) does
   `per_process = len(files) // world_size`, then
   `self.files = self.files[: per_process * world_size]` — the remainder files are
   **silently dropped** (10 files / 3 ranks → 1 file, ~1/10 of the data, lost every
   run). Each rank then gets a contiguous block of whole files.

2. **`_max_records_per_file` is the global minimum.**
   `_analyze_parquet_directory()` returns `stats_df["valid_rows"].min()` as the
   per-file record budget (`sequence_dataset.py:470`). Every file is then truncated
   to that minimum during iteration (`process()` decrements `_max_records_per_worker`
   and returns `None` past the quota). Files above the minimum lose their tail rows.
   Only survives because example data is near-uniform (`mean 86.9, std 0.36`).

3. **No equal-cardinality guarantee → DDP deadlock.**
   With non-uniform files, rank 0 can own more valid records than rank 1. An
   `IterableDataset` under DDP must yield the **same number of batches on every
   rank** or the collective `all_reduce` in `backward()` hangs when the shorter rank
   stops. Nothing here enforces it; the quota band-aid uses the wrong (min) count.

4. **Scan predicate ≠ iterate predicate.**
   The scan counts `list_value_length(sequence_columns[0]) < min_length`
   (`sequence_dataset.py:416`). The iterator filters on
   `len(record[sequence_columns[-1]])` (`_filter_by_length`, `:112`) **and** drops
   rows whose post-modality length falls below `min_length` (`process()`, `:268`).
   First vs last column, and modality filtering is invisible to the scan → the
   budget is wrong → ranks diverge.

5. **`selected_event_ids` unsupported.** Constructor raises `NotImplementedError`
   (`:354`). Distributed training simply cannot do event-modality filtering.

6. **No epoch decorrelation.** No `set_epoch()` (the `TODO` at `:284` never
   happened), no tail rotation, no per-epoch file reshuffle. `train.py:373` calls
   `set_epoch` behind a `hasattr` guard that silently skips this class — so every
   epoch every rank sees the identical records in the identical order.

`test_shard_iter_seq_dataset.py` "passes" only because it **mocks
`torch.distributed`** and its fixtures use uniform synthetic files, so defects 2–4
never trigger.

### What `ShardTabularDataset` already does right (the engine to extract)

- Threaded scan → `counts[i]` = valid records in file `i`, using the **same
  predicate** the iterator applies.
- Global prefix sum `cum`; `total`, `per_rank = total // world_size`,
  `tail = total % world_size`.
- Per-`(rank, worker)` contiguous logical slice of `[0, total)`, mapped to
  `(file, lo, hi)` filtered-row segments via `np.searchsorted`
  (`_cumsum_segments`, `_range_to_segments`, `_split_cyclic_range`).
- `drop_tail=True` → every rank yields **exactly** `per_rank` → no DDP hang.
  `drop_tail=False` → last rank keeps the remainder (eval).
- `set_epoch()` + `rotate_tail` → deterministic rotation of the global stream each
  epoch (`_epoch_offset`).
- Runtime **divergence guard**: raises if the iteration's valid-record count for a
  file disagrees with the scan (`_read_owned_records`, `:954`).
- Persistent **filter-index cache** (`_prepare_filter_cache`): sha-keyed
  `manifest.json` + per-file `NNNNNNNN.npy` of valid physical row indexes, with
  cross-rank coordination via `dist.all_reduce` / `dist.barrier` or, when the
  process group isn't up yet, a shared-filesystem poll.
- Own `(world_size, rank)` resolution from `torch.distributed` or `WORLD_SIZE` /
  `RANK` env (`_resolve_dist_info`).
- Optional per-worker epoch metrics (`configure_epoch_metrics` /
  `get_epoch_metrics` / `reset_epoch_metrics`): `source_rows_scanned`,
  `parquet_bytes_opened`, `filter_sec`.

The engine is sound; it is just welded into one 1046-line file and duplicates
`check_tabular_features` / `process_tabular` from `TabularDataset`.

---

## 2. Target layout

```
avatar/data/
├── __init__.py                  flat re-export of the whole public API
├── base/
│   ├── __init__.py
│   ├── fs.py                    resolve_filesystem(uri) -> (pa.fs.FileSystem, path); file discovery
│   ├── parquet.py              read_parquet_file / parquet_num_rows / scan helpers — fs-aware
│   ├── iterable.py             BaseParquetDataset: discovery + worker split + process() hook
│   ├── shard.py                ShardPlanner: the global record-level sharding engine
│   ├── filter_cache.py         PersistentFilterCache (extracted verbatim from tabular)
│   └── batch.py                move_to_device + shared batch mixins
├── sequential/
│   ├── __init__.py
│   ├── dataset.py              EventSequenceDataset  (always sharded)
│   ├── collate.py              EventSequenceCollateFn, FixedHorizonCollateFn, ColesCollateFn
│   └── batch.py                EventSequenceBatch
├── tabular/
│   ├── __init__.py
│   ├── dataset.py              TabularDataset  (always sharded)
│   ├── collate.py              TabularCollateFn, SupervisedCollateFn, UpliftCollateFn, MultiTask*
│   └── batch.py                TabularBatch, UpliftTabularBatch
└── sampler/
    ├── __init__.py             BaseSampler, ColumnFilterSampler, MultiTaskColumnsFilterSampler
    └── ...                     (5 dead samplers deleted — see §6)
```

`avatar/data/__init__.py` re-exports every public name so
`from avatar.data import TabularBatch, EventSequenceBatch` and the existing test
imports keep working. `avatar/data/dataset/__init__.py` and
`avatar/data/dataset/collate_fn/__init__.py` become **deprecation shims** (re-export
from the new paths + `DeprecationWarning`) for one release — the ~90 Hydra
`_target_: avatar.data.dataset.*` strings resolve unchanged while external configs
migrate. Same pattern as `avatar/nn/sequence/__init__.py`.

---

## 3. Unified sharding — `base/shard.py`

### 3.1 `ShardPlanner`

A standalone object (no `nn`/dataset coupling) that owns the "who reads which
records" math. Constructed once per dataset, re-queried per epoch / per worker.

```python
@dataclass
class ShardPlanner:
    file_counts: np.ndarray          # valid records per file, from the scan
    world_size: int
    rank: int
    drop_tail: bool = True           # True for train (equal ranks), False for eval
    rotate_tail: bool = False

    def rank_record_count(self, *, num_workers: int = 1) -> int:
        """Records this rank yields this epoch — the value __len__ must return."""

    def worker_segments(
        self, *, worker_id: int, num_workers: int, epoch: int
    ) -> list[tuple[int, int, int]]:
        """(file_index, lo, hi) filtered-row slices this (rank, worker) owns."""
```

Internals are lifted from `ShardTabularDataset`:
`_split_cyclic_range`, `_range_to_segments`, `_epoch_offset`, the
`divmod(rank_count, num_workers)` worker split. **Step 2 of the plan asserts the
segments are byte-identical to today's tabular output** on a fixed synthetic corpus.

### 3.2 The parity contract (mandatory)

The scan count for a file **must equal** the number of records the iterator yields
from that file before rank/worker slicing. Every dataset provides two methods that
apply the *same* predicate:

```python
class BaseShardedParquetDataset(BaseParquetDataset):
    def count_valid_records(self, file: str) -> int: ...          # scan phase
    def iter_file_records(self, file: str) -> Iterator[dict]: ... # iterate phase
```

`BaseShardedParquetDataset.__iter__` wraps `iter_file_records`, counts what passes,
and raises the divergence guard (ported from `tabular_dataset.py:954`) if a file
yields fewer valid rows than the scan promised. This is what makes DDP safe.

### 3.3 `world_size == 1`

Still shards across **DataLoader workers** (so `num_workers` splits the stream
without overlap — today `BaseIterDataset.files_per_worker` does a whole-file split
that starves workers when `num_workers > n_files`). The full-file scan still runs;
for the filter-cache path it is amortised across epochs and ranks. Escape hatch:
`shard=False` skips the scan and falls back to the legacy per-file worker split for
tiny / debug datasets.

### 3.4 `(world_size, rank)` source

Datasets take an optional `dist_env: DistEnv | None` (the dataclass from
`train_redesign.md` §1.2). When `None`, resolve from `torch.distributed` then
`WORLD_SIZE` / `RANK` env — keeps the dataset usable from a notebook or a bare
script. The trainer passes its `DistEnv` explicitly so there is one source of truth.
`set_epoch(epoch)` is called from the trainer loop (or an `on_epoch_begin`
callback); **both** datasets implement it after this refactor.

### 3.5 Collapsing `Shard*` into one class

- `TabularDataset` **becomes** today's `ShardTabularDataset` logic, with
  `shard_by_rank` renamed `shard` and defaulting to `True`. The duplicated
  `check_tabular_features` / `process_tabular` collapse to one copy.
- `EventSequenceDataset` is rebuilt on `BaseShardedParquetDataset`. Its
  record-processing (`_filter_by_length`, `_modalities_mask`, `_process_record`,
  `_prepare_record`) is **preserved as-is** — only the sharding is swapped from the
  broken file-split to `ShardPlanner`. `selected_event_ids` is restored: the scan
  counts post-modality-filter rows (read `sequence_columns[-1]` + `event_ids_column`
  during the scan, apply the modality mask, then the length check).
- `ShardTabularDataset` / `ShardEventSequenceDataset` stay as thin
  `DeprecationWarning` subclasses (`= TabularDataset` / `= EventSequenceDataset`)
  for one release.

### 3.6 `random_slicing` determinism

Long-sequence random slicing (`_filter_by_length`, `sequence_dataset.py:116`) uses
the global `random` module → not reproducible, correlated across workers. Seed it
per `(rank, worker_id, epoch)` from a `torch.Generator`. (It does not affect the
scan/iterate count — a row with `seq_len >= min_length` passes regardless of where
it is sliced — so it is safe to change independently.)

---

## 4. HDFS read support — `base/fs.py`

`pyarrow` 20 is installed (`pyproject` pins `>=14`); `pyarrow.fs.HadoopFileSystem`
and `pyarrow.fs.FileSystem.from_uri` are available. No new dependency.

### 4.1 Filesystem resolution

```python
def resolve_filesystem(
    path: str, options: dict | None = None
) -> tuple[pa.fs.FileSystem, str]:
    """
    'hdfs://namenode:8020/user/.../data'  -> (HadoopFileSystem, '/user/.../data')
    '/mnt/data' or 'file:///mnt/data'     -> (LocalFileSystem, '/mnt/data')
    """
    if "://" not in path or path.startswith("file://"):
        return pa.fs.LocalFileSystem(), path.removeprefix("file://")
    return pa.fs.FileSystem.from_uri(path)   # dispatches on scheme
```

Optional `filesystem:` block in config for explicit Hadoop params
(`host`, `port`, `user`, `replication`, `kerb_ticket`) when the URI alone is not
enough; otherwise URI + environment is sufficient.

### 4.2 What becomes fs-aware

| today (local-only) | after |
|---|---|
| `glob(os.path.join(path, "**/*.parquet"))` in `BaseIterDataset.collate_files` | `fs.get_file_info(FileSelector(base, recursive=True))`, keep `.parquet`, sort |
| `ds.dataset(file, format="parquet")` in `read_parquet_file` | `ds.dataset(path, filesystem=fs, format="parquet")` |
| `pq.ParquetFile(file)` / `pq.read_table(file, memory_map=True)` | `pq.ParquetFile(fs.open_input_file(path))` (no `memory_map` over HDFS) |
| `os.path.getsize(file)`, `os.stat(file).st_mtime_ns`, `os.path.abspath` | `fs.get_file_info(path)` → `.size`, `.mtime_ns`; normalise path via fs |
| `parquet_num_rows(path)` | same, `fs`-parametrised |

`read_parquet_file` and `parquet_num_rows` gain an `fs` parameter (default
`LocalFileSystem` → existing call sites unchanged). `BaseParquetDataset` resolves
the fs once in `__init__` and threads it everywhere.

### 4.3 Filter cache over HDFS

The persistent filter cache writes `.npy` / `.json` with `os.replace` (atomic
rename) + `os.fsync`, and coordinates ranks by polling for file existence. HDFS
rename/visibility semantics do not give the same guarantees. Rule:

- **When the data source is HDFS, `filter_cache_dir` must point at a shared POSIX
  mount** (NFS / Lustre visible to every node). Assert this in `__init__`; if
  unset, `warn` + disable the cache (fall back to the plain threaded scan every
  run).
- The cache never lives on HDFS.

### 4.4 Docs / workflow change

`examples/basics/README.md` and `experiments/sbercampaign_pilot/README.md` currently
instruct `hdfs dfs -get hdfs://.../data <local>` before running. With HDFS read
support the configs can point `path:` straight at `hdfs://...` and skip the copy.
Update both READMEs; keep the `-get` path documented as the offline-friendly option.

### 4.5 Caveats

- `HadoopFileSystem` spins a JVM (libhdfs) **per worker process** — `num_workers: 8`
  means 8 JVMs. Memory budget; may need `num_workers` down or `persistent_workers`
  so the connection is reused across epochs. Document required env
  (`HADOOP_HOME` / `CLASSPATH` / `ARROW_LIBHDFS_DIR` / `LD_LIBRARY_PATH`).
- `read_parquet_file(shuffle=True)` loads the **whole file** into memory to permute
  rows — already true locally, worse over HDFS. Row-group-level streaming +
  shuffle-buffer is a follow-up, out of scope here.
- Connection setup latency per worker on first read; the scan phase (many small
  metadata reads) is the most HDFS-sensitive part — the filter cache mitigates it
  after the first run.

---

## 5. API sketch

```python
class BaseParquetDataset(IterableDataset):
    def __init__(
        self,
        path: str | list[str],
        *,
        filesystem: dict | pa.fs.FileSystem | None = None,
        read_columns: list[str] | None = None,
        shuffle_files: bool = True,
        shuffle_pq: bool = True,
    ): ...
    def process(self, record: dict) -> dict | None: ...   # modality hook, unchanged contract

class BaseShardedParquetDataset(BaseParquetDataset):
    def __init__(
        self, *args,
        shard: bool = True,
        drop_tail: bool = True,
        rotate_tail: bool = False,
        dist_env: "DistEnv | None" = None,
        scan_workers: int = 4,
        filter_cache: bool = True,
        filter_cache_dir: str | None = None,
        **kwargs,
    ): ...
    def count_valid_records(self, file: str) -> int: ...       # scan predicate  (abstract)
    def iter_file_records(self, file: str) -> Iterator[dict]: ... # same predicate (abstract)
    def set_epoch(self, epoch: int) -> None: ...
    def __len__(self) -> int: ...        # ShardPlanner.rank_record_count()
    def __iter__(self) -> Iterator[dict]: ...   # segments -> read -> process -> divergence guard
    # opt-in perf counters (unchanged names)
    def configure_epoch_metrics(self, num_workers: int) -> None: ...
    def get_epoch_metrics(self) -> dict[str, float]: ...
    def reset_epoch_metrics(self) -> None: ...
```

`TabularDataset(BaseShardedParquetDataset)` and
`EventSequenceDataset(BaseShardedParquetDataset)` implement the two predicate
methods and keep their existing `process()` / collate contracts. Batch containers
(`TabularBatch`, `EventSequenceBatch`) are unchanged.

---

## 6. Blast radius

- **~55 configs** `_target_: avatar.data.dataset.TabularDataset` + **4** for
  `EventSequenceDataset`: resolve unchanged via the flat re-export + the
  `avatar.data.dataset` shim. **Behaviour change:** the dataset now shards by
  default. On 1 GPU with `drop_tail=True` the output is identical except the global
  remainder (`total % 1 == 0` → nothing dropped); on N GPUs each rank now yields
  exactly `total // N` instead of accelerate's dispatch split. Document; provide
  `shard=False` for exact legacy behaviour.
- **collate `_target_`** `avatar.data.dataset.collate_fn.*` (86 refs): shim module
  re-exporting from `avatar.data.{sequential,tabular}.collate`.
- **Samplers** — delete the 5 dead + 1 unexported
  (`AccumulateSampler`, `ColumnsFilterSampler`, `StreamingBalancedSampler`,
  `StreamingBalancedUnderSampler`, `ScheduleConversionSampler`,
  `DatePrioritazeSampler`; 0 refs anywhere). Keep `BaseSampler`,
  `ColumnFilterSampler` (6 configs), `MultiTaskColumnsFilterSampler` (wired into the
  filter cache).
- **`train.py`** — the `if getattr(dataset, "shard_by_rank", False)` branches
  (`:153, 257, 294, 703`) and the `hasattr(dataset, "set_epoch")` /
  `hasattr(dataset, "configure_epoch_metrics")` guards collapse to unconditional
  calls once every dataset is a `BaseShardedParquetDataset`. Ties into
  `train_redesign.md` §0 (which then loses the "no change to `avatar/data`" claim —
  this doc supersedes that line).
- **`pyproject.toml`** — no new hard dep. Optional documented env for HDFS; a
  `hdfs` extra is not needed (pyarrow already carries the client).
- **Tests** — reorganise `tests/data/` → `tests/data/{base,sequential,tabular}/`.
  The 5 sharding-correctness tests in `test_shard_iter_seq_dataset.py` become the
  `ShardPlanner` spec and are **un-mocked**: add a real
  `torchrun --nproc_per_node=2` test on tiny synthetic data (union of ranks == full
  set, no key collisions, equal batch count) — same posture as the `train_redesign`
  smoke test. Add the first-ever `ShardTabularDataset` sharding test in the process.
- **`docs/`** — update the two READMEs (§4.4); fix the stale
  `ShardEventSequenceDataset` mention in `docs/best_practices/gradient_checkpointing.md`.

---

## 7. Step plan

1. **`base/fs.py` + fs-threading.** `resolve_filesystem`, fs-aware discovery in
   `collate_files`, `fs=` params on `read_parquet_file` / `parquet_num_rows`. Local
   path byte-identical. Tests with a `LocalFileSystem` round-trip.
2. **Extract the engine.** `ShardPlanner` → `base/shard.py`, `PersistentFilterCache`
   → `base/filter_cache.py`. Re-point `ShardTabularDataset` at them. Assert the
   `(file, lo, hi)` segments and `__len__` are identical to the pre-refactor output
   on a fixed corpus (golden test).
3. **Rebuild `EventSequenceDataset` on the engine.** Fix defects 1–6; restore
   `selected_event_ids`; port + un-mock the 5 sharding tests; add the divergence
   guard.
4. **Collapse `Shard*` → one class per modality** + `DeprecationWarning` aliases.
   `set_epoch` / epoch-metrics on both.
5. **Move files** into `base/ sequential/ tabular/`; write the flat `__init__.py`
   re-export + `avatar.data.dataset` / `...collate_fn` shims.
6. **Prune dead samplers.**
7. **Migrate in-repo configs** (mechanical `_target_` rewrites where we want the new
   paths; the shims mean this is not urgent). Update the two READMEs.
8. **HDFS smoke** against a real cluster path — manual / CI-gated (needs a live
   namenode + libhdfs).

Land 1→4 behind the golden + smoke tests before 5. HDFS (1) and the sharding
rebuild (2–4) are independent and can proceed in parallel.

---

## 8. Risks

- **Filter cache over HDFS** — atomic-rename / visibility semantics differ; mandated
  POSIX cache dir (§4.3). If a site has no shared POSIX mount, they lose the cache
  and eat a full scan every run.
- **Default behaviour change for single-GPU tabular** — `shard=True` adds a
  startup file scan (threaded; amortised by the filter cache). Mitigated by
  `shard=False`. The per-rank `drop_tail` drops up to `world_size - 1` records
  globally — zero on 1 GPU, negligible on N.
- **Scan/iterate parity for sequences** — modality filtering + the last-vs-first
  column bug must be replicated *exactly* in the scan, or the divergence guard
  fires (fail-loud, which is the point) or — worse, if the guard is wrong — DDP
  hangs. Step 3 is the delicate one; the golden test must cover
  `selected_event_ids` + `min_length` interaction.
- **`__len__` accuracy** — `init_scheduler` uses `len(train_dataloader)` to size the
  LR schedule (`init_modules.py:129`). `ShardPlanner.rank_record_count()` must be
  exact per rank or the schedule shape changes. Same class of risk flagged in
  `train_redesign.md` §5.
- **libhdfs / JVM per worker** — memory and startup cost; may force `num_workers`
  down. Needs a real-cluster benchmark (TODO.md item 6 wants the epoch-metrics
  benchmark anyway).
- **No real distributed test today** — everything is mocked. The `torchrun` test in
  step 3 is a prerequisite, not a follow-up.
- **Scope** — three independent changes (layout, sharding engine, HDFS). Land in the
  step order; keep `ShardTabularDataset` byte-parity as the guard rail through 2–4.
