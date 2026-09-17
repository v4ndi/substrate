# Design: neural-network backend for `avatar.automl`

Status: **proposed**, open questions resolved 2026-09-16 (§11), review answers
and the `obligatory.md` audit items applied 2026-09-17 (§12), cross-checked
against an independent design the same day (§13). Stage 3 of the combine-avatar-automl task, on top of
[automl_migration.md](automl_migration.md) (stages 1–2, done).

Written against `master`, re-grounded on `refactor/data-sharding-hdfs`
(2026-09-16). That branch rewrote three of the four things this design leans
on — `avatar/train.py` became the `avatar/train/` package on plain
`torch.distributed` with callbacks, `avatar/data/` split into
`base` / `sequential` / `tabular` with one sharding engine and HDFS, and the
four tabular pipelines collapsed into `SupervisedLearner` + `SLearner`. Every
change was in this design's favour; §6 and N4 are written against the new API,
and the three `train.py` library-mode defects the first draft had to fix are
already fixed upstream.

Goal: `backend="tabnn"` trains a tabular neural network with hyperparameter
search through the same `BinaryTask/.../UpliftTask` API that `backend="boosting"`
uses today — same config object, same `train/predict/evaluate/save/load`, same
artifact layout, same metrics and reports.

Two hard constraints, in this order:

1. **The boosting pipeline does not change behaviour.** Every refactor step is
   gated on the parity harness reproducing byte-identical boosting results
   (§9, S0).
2. **Maximum reuse.** The NN backend reuses the AutoML task/data/metric/
   artifact layers unchanged, and the torch modules that already exist in
   `avatar.nn` / `avatar.pipeline`. New code is confined to one package.

**Target environment** (§12, F1–F3): one A100 80 GB; splits of 3–100 M rows and
~200 features; 128–256 GB RAM on the box; one AutoML run may take a night or
two. Every default below is calibrated for that, and the multi-GPU paths are
specified but dormant until a multi-card node exists.

---

## 1. Where the seam already is

The config layer was written with this in mind and already reserves the family:

```python
# avatar/automl/config/base.py
backend: Literal["boosting", "tabnn"]
...
if self.backend == "tabnn" and self.engine != "ste":       raise UnsupportedBackendError
if self.backend == "tabnn" and self.hyperopt:              raise ConfigError("Optuna hyperopt is available only for boosting")
if self.backend == "tabnn" and self.device == "cpu":       raise ConfigError("TabNN does not support device='cpu'")
```

All three guards go (N6, N8, N9). The training call chain for a supervised task
is:

```
Task.train
  -> TrainingCoordinator.execute            tasks/training.py    (generic: sources, plan, per-part loop)
     -> SupervisedBoostingTask._fit_one     tasks/supervised.py  (generic: prepare_data + target + metric)
        -> fit_boosting_model               backends/boosting/hyperopt.py
           -> backend.prepare_fit_data / fit_prepared / predict_prepared_score
```

Everything above `fit_boosting_model` is already backend-agnostic: it works on a
normalized polars frame, a `FeatureSchema`, a target `np.ndarray`, and an
`objective_metric(target, scores) -> float` callable. `fit_boosting_model`
itself is generic too — the Optuna loop, the "keep the best fitted model"
bookkeeping and the progress logging contain nothing boosting-specific except:

| boosting-specific in `fit_boosting_model` | generalization |
|---|---|
| `backend_class(engine=…, params=…, random_state=…, device=…, verbose=…)` | same signature for every family |
| `can_prequantize(space)` (CatBoost pools) | rename to `can_reuse_prepared(space)`; default `False` |
| `resolve_default_search_space(engine, …)` | per-family lookup |
| the `BoostingFitResult` name | `FitResult` |

Persistence is already delegated: `ArtifactRepository` calls
`item.backend.save(dir)` / `backend_class.load(dir)` and only reads
`backend.json` for the `engine` field, so an NN backend that writes the same
metadata file needs **no** artifact-layer change.

## 2. What the NN backend must provide

```python
class ModelBackend(ABC):                     # today: BoostingBackend
    engine: str; params: Mapping[str, Any]; random_state: int
    device: str; verbose: bool | int

    def predict_score(self, frame: pl.DataFrame, schema: FeatureSchema) -> np.ndarray
    def feature_importance(self, schema: FeatureSchema) -> pl.DataFrame | None
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> Self
    def set_runtime_device(self, device: str) -> None
    def for_execution(self, device: str) -> Self

class TrainableBackend(ModelBackend):        # what the search loop needs
    consumes: ClassVar[Literal["frames", "sources"]]
    def prepare_fit_data(self, part: PartDescriptor, *, reuse: bool) -> Prepared
    def fit_prepared(self, prepared: Prepared) -> None
    def predict_prepared_score(self, features: Any) -> np.ndarray
    def can_reuse_prepared(self, search_space) -> bool
```

`consumes` is what §6 turns on: `"frames"` (boosting, today's behaviour, two
materialized polars frames) or `"sources"` (tabnn: `ParquetSource` + a group
predicate, nothing materialized). For tabnn, `Prepared` holds **paths to an
encoded parquet cache**, not tensors — and unlike CatBoost quantization that
cache is *always* reusable across trials, so `can_reuse_prepared` returns `True`
unless the search space touches the encoding itself.

## 3. Locked decisions

- **N1 — one new package, `avatar/automl/backends/tabnn/`,** mirroring
  `backends/boosting/`: `interface.py` (shared runtime), `base.py` (assembly,
  the `Trainer` call, persistence), `binary.py` / `regression.py` /
  `multiclass.py` / `uplift.py` (task adapters), `spaces.py` (default space).
  `backends/boosting/**` keeps its files; only the three shared pieces move up.
- **N2 — backend-neutral names at the task layer, renamed outright.**
  `BaseBoostingTask -> BaseTask`, `SupervisedBoostingTask -> SupervisedTask`,
  `fit_boosting_model -> fit_model` (`backends/search.py`),
  `BoostingFitResult -> FitResult`. **No deprecation aliases** (§12, A2): the
  old names are semi-public but their users are in this repo and in cluster
  code we control, and a rename that leaves both spellings alive gets frozen in
  that state. S1 updates the call sites in the same commit.
- **N3 — no registry module.** Backend choice is a dict inside
  `Task._backend_class_for(config.backend)` (§12, A3). Two families do not
  justify a registry with its own module and import graph; a third family is a
  new line in the dict.
- **N4 — training goes through `avatar.train.Trainer`; AutoML owns no
  training logic.** `Trainer` (`avatar/train/loop.py`) takes model, optimizer,
  scheduler, dataloaders, metrics, `TrainingArguments`, `RunConfig`, a callback
  list and a checkpoint directory. Everything cross-cutting is a
  `TrainerCallback`, so the backend passes exactly three —
  `EarlyStoppingCallback`, `CheckpointCallback` and `MLflowCallback` (N15) —
  and gets nothing it did not ask for: no profiler, no progress bar, no
  throughput logging. What it does get is DDP, AMP, gradient accumulation,
  clipping, checkpoint rotation and resume. Writing a second training loop is
  not on the list of options.
- **N4a — no library-mode fixes are needed any more.** The first draft had to
  patch three defects in `avatar/train.py` (no return value, an unguarded
  `accelerator.trackers[0]`, an unconditionally indexed `logging_info`). The
  `train.py` rewrite removed all three: `Trainer.train()` returns
  `dict | None`, the objective is `trainer.state.best_metric` — written by
  `EarlyStoppingCallback` at every evaluation — and tracking is opt-in through
  the callback list. AutoML calls the library as it is.
- **N4b — one trial is one `Trainer(...).train()`, behind a trial-runner
  interface with three implementations** (§12, B2/B2a/B2b):

  | runner | when | how the objective comes back |
  |---|---|---|
  | in-process | `num_gpus <= 1` or CPU — **today's only real case** | `trainer.state.best_metric` |
  | `torch.distributed.run` subprocess | `num_gpus > 1` on one node | worker writes `result.json`, driver reads it |
  | Osiris job | `env_type="osiris"` with trial fan-out (N14) | job writes `result.json`, `status()` collects it |

  The three share one interface (`run_trial(spec) -> TrialResult`) and one spec
  format — the same `run_spec.json` protocol `avatar.automl.run` already uses
  for remote operations. There is no fourth option: running *the whole AutoML
  process* under `torchrun` is rejected, because launching from a notebook
  makes it impossible and because every rank would then execute the polars,
  reporting and artifact code.
- **N4c — ranking stays comparable across backends** (§12, B3). `Trainer` ranks
  by `avatar.metrics` objects, AutoML by its own registry
  (`resolve_metric(...).compute(MetricInput)`). **One** `ScalarMetric` subclass
  in `backends/tabnn/metric.py` bridges them *by composition*: it holds the
  resolved AutoML metric, accumulates targets and scores in `update()`, and
  `compute()` returns `{name: value}` for the configured `optimization_metric`.
  The AutoML metric classes are not touched and inherit nothing —
  `Metric` is a structural `Protocol`, `METRIC_REGISTRY` holds instances, and
  `avatar/automl/**` contains no `import torch` today, which is exactly the
  property that keeps the boosting path torch-free. The adapter sets
  `needs_full_population = True` (ROC-AUC and Qini are not sums) and names its
  `required_inputs` / `required_outputs` so a distributed run gathers those two
  tensors instead of whole batches. `EarlyStopping(main_metric=<that name>,
  strategy=direction)` then drives on exactly the number AutoML reports.
- **N4d — multi-GPU is a launch decision, not a backend one.** With
  `torch.distributed` there is no in-process launcher, so `world_size > 1`
  requires an external launcher; N4b puts that behind the subprocess runner.
  On the current hardware (one A100) the in-process runner is the only one that
  executes, and the subprocess runner is implemented when a multi-card node
  appears — the interface exists from the start so that arrival is a new class,
  not a refactor. Inside an Osiris job the rule is the same: one trial per job,
  and if the job holds several cards the trial uses them through
  `torch.distributed.run` within that job (§12, B2b).
- **N5 — the modules are reused verbatim:**
  `avatar.nn.embedding.TabularEmbedding` → embedding,
  `avatar.nn.tabular.TabularTransformer` → encoder,
  `avatar.pipeline.tabular.SupervisedLearner` → the whole stack (embedding,
  encoder, pooling, late-fused external embeddings, head, loss),
  `avatar.pipeline.tabular.SLearner` → uplift,
  `avatar.data.tabular.TabularBatch` → the batch contract. Binary, regression
  and multiclass are `SupervisedLearner` with a different `num_classes` /
  `task_type` pair, and the loss (`avatar.losses.ClassificationLoss`) is
  injected rather than chosen inside the model — exactly the shape a per-task
  backend adapter wants.
- **N6 — the tabnn engine is `engine="tabular_transformer"`** (§12, A4).
  `"ste"` is *rejected* with an `UnsupportedBackendError` naming the
  replacement: it was only ever a reserved value, no tabnn artifact exists, and
  the class it referred to was already renamed `STEv2 -> TabularTransformer` in
  [tabular_refactor.md](tabular_refactor.md), its deprecation shim since
  deleted. The engine string is persisted in `backend.json` and checked on
  load, so it names the class it builds. Adding an engine later (`mlp`,
  `ft_transformer`) is a dict entry plus a default search space.
- **N7 — encoding reuses `avatar.preprocessing.local.TabularPreprocessor`**
  (§12, C3). It fits **streaming, from the parquet source** — not from a
  materialized frame — and gives exactly what `TabularEmbedding` expects: a
  *shared* categorical vocabulary with per-column offsets (`offset_map`,
  `vocab_size`, a reserved `unk` id 0 that unseen values map to) and
  standardized numericals (`mean_std`, optional `signed_log1p`). This keeps
  AutoML's encoding identical to the encoding used by hand-written avatar
  training runs, and its Spark twin produces the same artifact. `dump()` is
  JSON-safe and goes into `backend.json`; it is **also written as a standalone
  file** next to the model (§8) so it can be restored with
  `TabularPreprocessor.load()` without parsing the AutoML artifact.
- **N8 — CPU is supported, with a warning** (§12, D1). The current
  `tabnn + device="cpu"` ban goes: the unit suite has no GPU, so a CPU path is
  the only way the backend is testable at all, and the boosting backend already
  supports both devices. `env_type="osiris"` keeps requiring `device="gpu"`.
  `device="cpu"` logs one explicit warning naming the training-row count,
  because CPU × `n_trials` × epochs is how a one-hour run silently becomes a
  one-day run.
- **N9 — hyperopt is enabled for tabnn, but the epoch budget is not searched**
  (§12, D2). The `"Optuna hyperopt is available only for boosting"` guard is
  removed once `fit_model` is family-agnostic. `max_epochs` and `patience` are
  **fixed defaults**, not search dimensions: with early stopping live, a larger
  epoch cap does not buy metric, it buys wall-clock, so that axis trades search
  budget for nothing. This mirrors boosting, where the iteration count is
  settled by early stopping inside one fit rather than by Optuna.
  **On large splits the unit is steps, not epochs:** at 100 M rows one epoch is
  hours and `patience=5` means "stop after a week", so the backend sets
  `TrainingArguments.steps_before_evaluation` (already supported —
  `avatar/training_arguments.py:46`, used at `avatar/train/loop.py:315`) and
  counts patience in evaluations. Default: evaluate every
  `ceil(rows / batch_size / 8)` steps, capped at one epoch.
- **N10 — task coverage lands in two waves** (§12, E1). Wave 1: `binary`,
  `response`, `regression`, `multiclass` — one `SupervisedLearner` each,
  differing only in `num_classes` / `task_type` and the injected loss. Wave 2:
  `uplift` via `SLearner` only — the boosting backend's S/T/X metalearner
  search is out of scope for v1, and `UpliftTaskConfig` gains no new field (an
  unsupported metalearner combination raises `UnsupportedBackendError`).
- **N11 — model-part layouts are unchanged, but partitioned lazily** (§12, C5).
  `global`, `per_group` and `global_and_per_group` are decided by
  `TrainingCoordinator` above the backend, so they work for free. Under
  `consumes="sources"` the per-group split becomes a scan predicate
  (`scan().filter(group == value)`) instead of `DataFrame.partition_by`
  (`tasks/training.py:143`), which is what makes per-group training possible at
  all when one split does not fit in RAM. The cache is written per part, so the
  predicate runs once and trials read their own slice directly.
- **N12 — out-of-core by construction: encode once to a packed parquet cache**
  (§12, C1/C2/G6). Real datasets already exceed RAM under the boosting backend,
  so the NN backend does not inherit AutoML's in-memory assumption — and **there
  is no in-memory fast path**, not even for small splits: two data paths in one
  backend cost more than one. `prepare_fit_data`:

  1. fits `TabularPreprocessor` in one streaming pass (`MeanStdAccumulator` /
     `ValueCountAccumulator` are already batched at `batch_rows=250_000`);
  2. `transform(..., output="packed", identity_cols=[target, treatment, group,
     date, *hidden_state_columns])` writes the `cat_features` / `num_features`
     list columns that `avatar.data.tabular.TabularDataset` expects.

  Every trial streams that cache through `TabularDataset` +
  `TabularCollateFn`; peak memory is one batch. The cache is the streaming
  analogue of CatBoost's shared quantized pool: written once, reused by every
  trial. Three properties are decisions, not mechanics:

  - **The cache is kept, not deleted** (§12, C2). It is what
    `calibrate(valid_path)` — a separate call after `train()` — re-scores, what
    the report and `evaluate` pass over again, what a resumed or extended search
    reuses, and what `global_and_per_group` reads twice in one `predict`.
    Deletion is an explicit operation: `cache_policy: "keep" | "delete"`
    (default `keep`) and `task.clear_cache()`.
  - **The cache directory is content-keyed**, not `<part>`-named: the key is the
    source manifest (`DataPreparation.source_manifest` already returns
    `path`/`size`/`modified_ns` per file, `tasks/preparation.py:53`) plus the
    part name plus the preprocessor config. Without the key, "keep" means a
    re-run on different data under the same `output_dir` silently reuses a
    stale cache. This is a condition of the keep policy, not a nicety.
  - **`cache_dir` is configurable**, default `<output_dir>/cache/`. Artifacts
    live in `<output_dir>/artifacts/`, so the cache is a sibling and `save()`
    never picks it up. **On Osiris the cache goes to NFS, not HDFS** (§12, G6):
    HDFS is read natively by `TabularDataset`, but the Kerberos ticket expires,
    and a job that streams batches for hours will hit an expired token
    mid-training. Source parquet is read from HDFS once, during encoding, while
    the ticket is fresh; after that every job reads NFS only.

  The cache is written as **many files, not one**. `TabularPreprocessor.transform`
  currently opens a single `pq.ParquetWriter(output_path, schema)`
  (`avatar/preprocessing/local/pipeline/tabular_pipe.py:156`); tabnn needs it
  split by row count or target file size. Sharding correctness does not depend
  on this — `ShardPlanner` splits by record, not by file
  (`avatar/data/base/iterable.py`) — but read parallelism and shuffle
  granularity do, and on NFS a single file is a contention point for K jobs.

  **Encoding runs in a process pool over parquet shards, not in one process.**
  Encoding blocks every trial, so a serial pass is a card standing idle: at
  100 M rows x ~200 features a single-process `NumCatPipeline.fit`
  (`avatar/preprocessing/local/pipeline/base_pipe.py:70-105`, one sequential
  loop over `iter_record_batches`) is measured in hours before the first step
  of the first trial. The shape is `shard -> worker -> partial state -> merge`
  for the fit and `shard -> worker -> part-*.parquet` for the transform, which
  is also how the cache ends up as many files. No distributed framework: a
  process pool over the accumulators that already exist. What has to be added
  is small and the existing code is unusually friendly to it:

  - `MeanStdAccumulator.update` already combines a batch's `(n, mean, m2)` into
    the running state with Chan's parallel formula
    (`avatar/preprocessing/base/accumulators.py:49-68`), so `merge(other)` is
    those same three lines with a second state in place of a batch.
  - `ValueCountAccumulator` holds `{column: {value: count}}`, so merging is
    summing dicts.
  - **The vocabulary does not depend on shard order.** Ids are assigned in
    `finalize(order="sorted")` — the default (`label_encoder.py:41`) — and the
    count-descending orders break ties by value
    (`accumulators.py:136-176`), so a parallel fit reproduces the serial
    artifact exactly. `order="first_seen"` is the one order-dependent mode and
    is rejected when the parallel path is used.
  - The unit of parallelism is `NumCatPipeline.fit`, not the individual
    accumulator: both accumulators are updated in one pass over the same
    batches, and splitting them would read the source twice.

  This is the one idea taken from the independent design (§13).
- **N13 — hidden states reach the model as vectors; there is no flatten path**
  (§12, C4). `_expand_hidden_states` (`avatar/automl/data/schema.py:162`, called
  at line 306) turns a 256-dim embedding into 256 scalar columns and folds them
  into `numerical` / `feature_order`. That is the only thing a booster can eat
  and the wrong thing for a network: the embedding then dominates attention by
  sheer column count, each coordinate gets standardized separately (destroying
  the geometry the source model produced), and each gets its own input
  projection. tabnn therefore uses **late fusion only**:
  `SupervisedLearner(hidden_state_dim=…, normalize_hidden_states={name: width},
  proj_hiddens_to_dim=…)` (`avatar/pipeline/tabular/supervised.py:154-175`,
  width accounted at line 233), with `TabularDataset(hidden_state_columns=[…])`
  reading the list column natively (`avatar/data/tabular/dataset.py:61`). Early
  fusion (`hidden_state_aggregator`, the embedding as an extra token) is not in
  wave 1 — it touches the encoder — and stays a candidate search axis later.

  Two consequences: schema construction must learn **not** to expand (a
  backend-family flag; `TabularSchema` carries `hidden_states: {name: width}` as
  its own field, and the widths are already computed from the parquet schema
  without reading rows), and the cache carries the list column as an
  `identity_col` — the preprocessor must not standardize it, because the
  embedding is normalized as a whole vector by `LayerNorm` inside the model.
  Feature parity with boosting is therefore gone by design: boosting sees 256
  scalar columns, tabnn sees one vector. Metric comparison stays honest (same
  data, same metric) but compares *pipelines*, not only architectures, and the
  benchmark in S7 says so.
- **N14 — on Osiris, the fan-out goes one level deeper: a job per trial**
  (§12, G1–G5, G7). AutoML already fans out one job per model part
  (`tasks/operations.py`, `operation["jobs"]`, `status()`,
  `_finalize_remote_train`); trial-level fan-out reuses that machinery. What
  changes around it:

  - **Optuna moves to the driver** and the job becomes "train one parameter set,
    return a metric". `search_strategy: "tpe" | "random" | "grid" |
    "explicit"` decides how the driver samples; the default is `tpe` for the
    local sequential path (the driver is alive by construction) and **`random`
    for Osiris fan-out**.
  - **`grid` is the Optuna-free mode** (§13). It expands a fully categorical
    `search_space` into its Cartesian product, and when `n_trials` is smaller
    than the product it picks that many combinations deterministically from
    `random_state`. Nothing is sampled and nothing is persisted: the trial list
    is a pure function of the config, so resuming after a dead driver needs no
    state at all. It is an option, not the default, because it cannot express a
    continuous axis — `lr` log-uniform matters more for a network than for a
    booster — and because the product explodes on the ten axes of §7 (three
    values each is ~59 000 combinations). Use it for a small explicit grid,
    where it is strictly better than sampling.
  - **With `random`/`explicit` there are no waves.** All K parameter sets are
    sampled before the first submit, so the driver's only job is to submit and
    later collect. This matters because the driver is a notebook container that
    can die: jobs cannot submit jobs, so there is no in-cluster driver, and any
    design that needs the notebook alive between waves is fragile. Adaptivity
    costs a live driver; giving it up buys "the driver need not survive".
    The quality loss is small at this budget — with a wave of width K every
    trial in the wave is sampled from the same posterior, so 50 trials at K=20
    is two or three adaptive updates, and random search is a strong baseline on
    six axes at that budget. Waves stay documented as an opt-in; the persistence
    below already supports them.
  - **We drop adaptivity, not Optuna.** `RandomSampler` + `ask()` keeps one
    `search_space` format for both families, one trial history for the report,
    and makes TPE a one-line sampler change later.
  - **Search state is persisted next to the jobs.** In-memory storage would lose
    the trial↔job mapping on a driver restart — the job ids themselves survive,
    because `update_operation` writes `operation.json` before waiting
    (`avatar/automl/lifecycle.py:183`, `tasks/operations.py:549`). So each trial
    is recorded with `trial_id`, parameters, state and metric; **parameters are
    written before submit**, metrics after polling; on resume the study is
    rebuilt with `add_trial`. The sampler is seeded and trials are re-added in
    order, or a resumed run diverges from an uninterrupted one.
  - **Resource fields.** `num_gpus` means **cards per trial** (per job) — the
    `EnvironmentConfig` docstring that calls them "GPUs for the operation" is
    corrected. `trials_per_job` is not introduced: one trial per job is the
    point. `max_parallel_jobs` is an optional cap; when it is set, submission
    happens in chunks and must be **resumable** — `status()` finds trials in
    state `planned` without a `job_id` and submits them. The deterministic
    per-trial `run_dir` plus `mkdir(exist_ok=False)`
    (`avatar/automl/environment.py:238`) is the write-ahead marker, and the job
    name must carry `trial_id` (today it is `f"fmlib-{action}-{run_id[:8]}"`
    with a fresh uuid, `environment.py:265`), so a crash between submit and
    recording the id yields a findable orphan rather than a duplicate job.
  - **A failed trial is `tell(state=FAIL)` and the run continues.** Under
    `random` that costs one point out of K; the operation succeeds if any trial
    produced a model, and failure counts and reasons land in `operation.json`
    and the report. **The local sequential path needs the same tolerance:**
    `study.optimize` is called without `catch=`
    (`backends/boosting/hyperopt.py:336`), so today the first failing trial
    aborts the whole search. That is tolerable for boosting, where a trial
    rarely fails, and wrong for tabnn, where OOM is an ordinary outcome of a
    sampled `batch_size` x `hidden_size` pair. `fit_model` therefore catches
    per-trial exceptions **for tabnn only** and records them as failed trials;
    the boosting call keeps today's fail-fast behaviour, because changing it
    would change boosting results on a failing trial.
  - **Boosting keeps its execution path.** Trial fan-out is tabnn-only: a
    boosting trial takes minutes, the per-job overhead is not worth it, and the
    first constraint of this design is not to disturb that path. The shared
    trial selector makes enabling it later a configuration change.
  - **Reproducibility is scoped.** Byte-identical parity is claimed and tested
    only for the local sequential CPU path. On the cluster, GPU arithmetic is
    non-deterministic, the set of completed trials can differ (failures), and a
    0.7231-vs-0.7230 margin can flip which parameters reach the artifact — so
    the assertion is metric equality within tolerance, and `best_params`
    equality is not asserted. The *search* is reproducible: seeded `random`
    gives the same parameter sets run to run.
- **N15 — every trial logs to MLflow** (§12, B6). `MLflowCallback`
  (`avatar/train/callbacks/mlflow.py`) is passed with the trial's parameters as
  MLflow params and the AutoML metric as the tracked metric: one MLflow run per
  trial, named `<run_id>/<part>/trial-<trial_id>`, so the Osiris fan-out and the
  local loop produce the same tree. Tracking URI and experiment come from the
  config; when no URI is configured the callback is not constructed and nothing
  is logged. AutoML's own `progress.py` reporting stays as it is — MLflow is in
  addition to it, not instead.
- **N16 — four reliability defects of the remote path are fixed before trial
  fan-out is built on top of it.** They come from an independent audit
  (`obligatory.md`) and were re-verified against this branch on 2026-09-17.
  Trial fan-out replaces one or two jobs per operation with K, which moves each
  of them from "rare annoyance" to "loses a night of training":

  - **An operation must not be terminal before it is finalized.** `status()`
    writes `state=aggregate` — i.e. `succeeded` — at
    `tasks/operations.py:381`, *then* calls `_finalize_remote_train` (:388),
    and the polling loop skips operations already in
    `succeeded`/`failed`/`partial_failed` (:375). A crash in between leaves an
    operation that is terminal and unfinished, and no later `status()` will
    touch it again. Fix: a non-terminal `finalizing` state written before
    finalization, `succeeded` only after it, and finalization itself
    idempotent so a resumed driver can repeat it. This is the same failure the
    persisted search state of N14 exists for, one step later in time — for
    tabnn, "finalization" *is* "collect K trial results and pick the best".
  - **Job ids are persisted per submit, not per batch.** `_start_remote_train`
    appends to a local `jobs` list and calls `update_operation` once after the
    loop (`tasks/operations.py:178-208`), so a failure on submit *k* loses the
    ids of jobs 1..k-1 while they keep running and keep holding cards. Fix:
    `update_operation(..., jobs=jobs)` after **every** successful `submit`.
    N14's write-ahead `run_dir` and `trial_id`-bearing job name then cover the
    remaining window — a crash *between* `submit` and the write leaves a
    findable orphan instead of a silent duplicate.
  - **`unknown` needs a finite policy.** A job missing from `osiris.list()`
    polls as `unknown` (`environment.py:345`), the aggregate becomes `unknown`
    (:372), and `status(wait=True)` counts that as active forever
    (`tasks/operations.py:428`). With K trial jobs the chance of meeting it is
    K times higher, and the cost is a run that hangs all night *and* withholds
    the results that did arrive. Fix: `unknown` is tolerated for
    `unknown_job_grace_seconds` (default 900) and then becomes a terminal
    `lost` with diagnostics; a `lost` trial is `tell(state=FAIL)` like any
    other failure and the search finishes with the trials it has.
  - **`CUDA_VISIBLE_DEVICES` stops leaking, and stops being `"0"` for tabnn.**
    `run_local` sets it for every backend and never restores it
    (`environment.py:142`; the `finally` at :165 only closes log handlers), so
    one AutoML call permanently narrows the notebook kernel to one card. Two
    separate corrections, and only the first applies to boosting:
    **(i)** the variable is set through a context manager that restores the
    previous value in `finally` — inside the callback nothing changes, so the
    parity gate stays green; **(ii)** the pinning applies **only to
    `backend="boosting"`**, exactly as `submit` already does for remote jobs
    (`environment.py:268-271`). The `"0"` is deliberate and stays: CatBoost
    with `task_type="GPU"` and no explicit `devices`
    (`backends/boosting/binary.py:41`) spreads over every visible card, and
    `tests/automl/test_environment.py:159` asserts the current behaviour. For
    tabnn, visibility follows `num_gpus` — without that the multi-GPU
    subprocess runner of N4b cannot see a second card, and `num_gpus` as
    "cards per trial" (N14) would be a lie locally.

  A fifth audit item — unsynchronized read-modify-write on `operation.json`
  (`lifecycle.py:182-193`, with a shared `operation.json.tmp` temp name) — is
  **not** a concurrency problem in this design: the driver is the only writer
  of that file, and trial jobs write `result.json` inside their own `run_dir`.
  It becomes one through N14's recovery story, where "the container died, I run
  `status()` again" can put two drivers on one operation. The answer is a
  single-writer check rather than a lock: the record already carries
  `owner.{hostname,pid,token}` (`lifecycle.py:173-178`) and
  `reconcile_local_operations` (`lifecycle.py:244`) already implements the
  liveness test, so a second driver either takes ownership or refuses.

## 4. Target layout

```
avatar/automl/backends/
  __init__.py
  interface.py        ModelBackend, TrainableBackend  (from boosting/interface.py)
  search.py           FitResult, fit_model, suggest_params  (from boosting/hyperopt.py)
  boosting/           unchanged except the three imports above
  tabnn/
    __init__.py
    interface.py      TabNNBackend: device handling, module construction
    encoding.py       TabularPreprocessor fit + packed-parquet cache (N12)
    assembly.py       build model/optimizer/scheduler/dataloaders/EarlyStopping
    metric.py         AutoMLMetric(ScalarMetric) adapter (N4c)
    loader.py         TabularDataset + TabularCollateFn over the cache
    runner.py         trial runners: in-process / torchrun subprocess / Osiris job (N4b)
    base.py           BaseTabNNBackend: prepare_fit_data / fit_prepared / save / load
    binary.py         BinaryTabNNBackend        (num_classes=1, classification)
    regression.py     RegressionTabNNBackend    (num_classes=1, regression)
    multiclass.py     MulticlassTabNNBackend    (num_classes=K, class_order in backend.json)
    uplift.py         UpliftTabNNBackend        (SLearner, wave 2)
    spaces.py         default_search_space(engine, preset)
```

No `registry.py` (N3): the family→class mapping is a dict in
`Task._backend_class_for()`.

## 5. Data path: sources -> encoded cache -> `TabularBatch`

Nothing is materialized. The three passes over the data are:

```
1. schema        scan().collect_schema()          parquet footers only, no rows
                 + hidden-state widths            from the schema, not the rows
2. encoder fit   TabularPreprocessor.fit(source)  streaming, batch_rows=250k
                   categorical_columns=schema.categorical
                   numeric_columns=schema.numerical
                   spec_tokens={"unk": 0}
3. encode        .transform(source, out, output="packed",
                            identity_cols=[target, treatment, group, date,
                                           *hidden_state_columns])
                 -> <cache_dir>/<key>/{train,valid}/part-*.parquet
```

Hidden-state columns pass through untouched (N13): they are identity columns in
the cache, `LayerNorm`-ed as whole vectors inside the model, never standardized
per coordinate and never expanded into scalar features.

Each trial then reads the cache with the classes that already exist:

```python
# avatar/data/tabular/
TabularDataset(path=str(cache / "train"), shuffle_files=True, shuffle_pq=True,
               hidden_state_columns=[…], shard=True, drop_tail=True)
```

`shard=True` splits the record stream across ranks and workers, and
`drop_tail=True` drops `total % world_size` records so every rank produces the
same number of steps. `predict_score` must score every row exactly once, so it
reads the cache with `shard=False` — single-process today (N4d), and
`avatar.train.predict`'s `local` reduction mode when a distributed prediction
path is added.

Late fusion is wired and the bug the first draft found is fixed:
`TabularClassification.forward` used to run `tab_features.hidden_states.isnan()`
unconditionally and raise on `hidden_states=None`;
`SupervisedLearner._external_embeddings` now returns `None` when there is
nothing to fuse. So the backend only has to pass `hidden_state_dim` (or
`normalize_hidden_states`, which layer-normalises each named embedding
separately) and let the pipeline concatenate after pooling — no pipeline change
at all.

## 6. Training: what the backend hands to `avatar.train.Trainer`

No loop of our own (N4). One trial is:

```python
model     = SupervisedLearner(                          # avatar/pipeline/tabular
                embedding=TabularEmbedding(num_numerical_features=n_num,
                                           vocab_size=preprocessor.vocab_size,
                                           hidden_size=params["hidden_size"]),
                tabular_encoder=TabularTransformer(**encoder_params),
                aggregation_config={"name": params["aggregation"]},
                num_classes=…, task_type=…, dropout_p=params["dropout_p"],
                normalize_hidden_states=schema.hidden_states or None)
loaders   = DataLoader(TabularDataset(str(cache / "train"), shuffle_files=True, …),
                       collate_fn=TabularCollateFn(target_column="target", …))
optimizer = AdamW(model.parameters(), lr=…, weight_decay=…)
scheduler = get_cosine_schedule_with_warmup(…)
stopping  = EarlyStopping(main_metric=metric_name,
                          patience=defaults["patience"], strategy=direction)

trainer = Trainer(
    model=model, optimizer=optimizer, scheduler=scheduler,
    train_dataloader=train_loader, valid_dataloader=valid_loader,
    training_arguments=TrainingArguments(num_epochs=defaults["max_epochs"],
                                         steps_before_evaluation=eval_every,
                                         seed=config.random_state,
                                         clip_grad_norm=params.get("clip_grad_norm")),
    run_config=RunConfig(amp=amp),                       # fixed, not searched
    valid_metrics=[AutoMLMetric(config.optimization_metric, task)],
    callbacks=[EarlyStoppingCallback(stopping),          # order matters: it
               CheckpointCallback(trial_dir,             # vetoes the next one
                                  max_checkpoints=1),
               *mlflow_callback],                        # N15, when configured
    checkpoint_dir=trial_dir)
trainer.train()
objective = trainer.state.best_metric
```

Five properties of that call are worth stating, because the design depends on
them:

- **The objective is not guessed.** `EarlyStoppingCallback.on_evaluate` writes
  `ctx.state.best_metric` on every evaluation, so `trainer.state.best_metric`
  is the best value of AutoML's own metric over the run. `train()`'s return
  value is the *test* score and stays `None` here — `test_dataloader` is never
  passed, because AutoML evaluates through its own `evaluate()`, on its own
  metrics and reports.
- **The selected weights are the last checkpoint under `trial_dir`** (§12, B5).
  The loop sets `control.should_save = True` before firing `on_evaluate` and
  lets callbacks veto it; `EarlyStoppingCallback` clears the flag whenever the
  metric did not improve. So a checkpoint exists only for an improvement, and
  `max_checkpoints=1` keeps exactly one file per trial.
- **Callback order is a contract, not a style choice.** Early stopping must run
  before the checkpoint callback, or the veto arrives after the write.
- **Evaluation cadence is a function of data size** (N9). On a 3 M-row split,
  once per epoch; on 100 M rows, every N steps with patience counted in
  evaluations.
- **Nothing else is switched on.** `build_default_callbacks` needs a Hydra
  `DictConfig` and brings the profiler, throughput and progress bars; AutoML
  builds its callbacks directly and keeps its own `progress.py` reporting.
- **A trial returns a path and a number, never a model.** `TrialResult` carries
  the `trial_id`, the objective and the checkpoint directory; the fitted module
  stays on disk, and the search loop holds no torch object between trials.
  Boosting does the opposite — `best_backend = backend`
  (`backends/boosting/hyperopt.py:313`) keeps the best fitted estimator in the
  coordinator's RAM — which is affordable for a booster and is not for a
  network, on top of being impossible for two of the three runners: a
  subprocess and an Osiris job cannot hand back a Python object. After the
  search the backend is rebuilt once, from the winning checkpoint. The boosting
  behaviour is left alone (constraint 1); it stays a separate backlog item.

What AutoML contributes on top, and nothing more:

- **assembly** — modules, optimizer, scheduler and dataloaders from the trial's
  parameters (`assembly.py`);
- **the metric bridge** (N4c) so the ranking number is AutoML's;
- **the trial runner** (N4b) so where a trial executes is not the backend's
  concern;
- **prediction** — `predict_score` streams the test source through the fitted
  preprocessor and the model. `avatar.train.predict` already does exactly this
  loop, including the three reduction modes and the `drop_tail` warning, so the
  backend calls it with `metrics=None` and collects the scores: the one array
  allowed to be O(rows).

**What this costs above the backend — two shared-code changes, not one.**

1. `TrainingCoordinator` currently calls `DataPreparation.read_source` and hands
   `_fit_one` two materialized frames (`tasks/training.py:83`). For
   `consumes="sources"` it passes a part descriptor (source + group predicate +
   schema) instead; the boosting `_fit_one` opens with two lines that
   materialize, so its behaviour is unchanged and only the call site of
   `read_source` moves. The plan step needs no new code —
   `ParquetSource.unique_column_values` already exists for exactly this reason
   (`tasks/base.py:226`, used by the remote path).
2. Schema construction must stop expanding hidden states for tabnn (N13), and
   `TabularSchema` must carry their widths as a field.

Both land in S2b, both gated on S0.

## 7. Config surface

No new required fields for the network itself. `model_params` (without hyperopt)
and `search_space` (with) carry the network settings, exactly as for boosting.
Defaults are calibrated for one A100 80 GB and splits of 3–100 M rows (§12,
F1–F3):

| parameter | default | searched? |
|---|---|---|
| `hidden_size` | 256 | categorical 128 / 256 / 512 |
| `num_layers` | 3 | int 1–6 |
| `num_heads` | 8 | categorical 4 / 8 / 16 (divisor of `hidden_size`) |
| `attn_dropout` | 0.15 | float 0.0–0.4 |
| `dropout_p` (head) | 0.2 | float 0.0–0.5 |
| `out_head_hidden_dim` | 256 | categorical 128 / 256 / 512 |
| `aggregation` | `mean` | categorical mean / sum_layernorm / linear |
| `lr` | 3e-4 | float 1e-5–3e-3, log |
| `weight_decay` | 1e-2 | float 1e-6–1e-1, log |
| `batch_size` | 4096 | categorical 1024 / 4096 / 8192 |
| `max_epochs` | 30 | **fixed** (N9) |
| `patience` | 5 | **fixed** (N9) |
| `amp` | `bf16` if `torch.cuda.is_bf16_supported()` else `no`; always `no` on CPU | **fixed** (§12, D4) |

`amp` is a launch parameter, not a search axis: the metric difference between
`bf16` and `no` is noise, it is coupled to `batch_size` through memory, trials
in different precision are not strictly comparable, and bf16 requires Ampere+
so a searched value does not travel with the artifact. The chosen value is
recorded in `backend.json`.

Two presets answer F3's "a night or two, with a small grid and a large one":
`search_preset: "fast" | "deep"` selects `n_trials` and the width of the ranges
above (`fast` ≈ 12 trials on the narrow ranges, `deep` ≈ 40 on the full ones).
`search_space` overrides still win over both.

New operational fields:

| field | meaning |
|---|---|
| `cache_dir` | where the encoded cache lives; default `<output_dir>/cache/`, NFS on Osiris (N12) |
| `cache_policy` | `keep` (default) or `delete` (N12) |
| `search_strategy` | `tpe` (local default) / `random` (Osiris default) / `grid` / `explicit` (N14) |
| `search_preset` | `fast` / `deep` |
| `max_parallel_jobs` | optional cap on concurrently submitted trial jobs (N14) |
| `num_gpus` | **cards per trial**, i.e. per job — docstring corrected (N14); locally it also sets `CUDA_VISIBLE_DEVICES` for tabnn (N16) |
| `unknown_job_grace_seconds` | how long a job missing from the scheduler listing stays `unknown` before it becomes `lost`; default 900 (N16) |

The existing `task_owned_model_params` guard (which rejects `device`, `seed`,
`verbose`, … inside `model_params`) applies unchanged.

## 8. Artifact

`<model part>/` gains, next to the existing `backend.json`:

```
backend.json        engine, params, random_state, verbose, task_state,
                    preprocessor (TabularPreprocessor.dump()),
                    dims {n_cat, n_num, vocab_size, hidden_states},
                    class_order, amp
preprocessor.yaml   the same preprocessor state as a standalone file (§12, D5),
                    loadable with TabularPreprocessor.load() without parsing the
                    AutoML artifact — for hand-written training and inference
model.safetensors   state_dict of the assembled module (the `embedding`,
                    `tabular_backbone`, `agg_layer`, `proj`, `out_head` key
                    prefixes SupervisedLearner fixes, so a checkpoint trained
                    by hand and one trained by AutoML are interchangeable)
```

No pickle: `.pt` / TorchScript / ONNX are not produced (§12, D5). `engine` keeps
its meaning for the manifest check, so `ArtifactRepository`, `lifecycle.py` and
the remote `run.py` spec need no change. Loading builds the module from
`params` + `dims` and then loads the tensors, and hidden-state widths are
validated on predict the same way `feature_order` is.

## 9. Staged plan

Every stage ends green on `pytest` **and** on the boosting parity harness.
Order confirmed in §12 (E4).

- **S0 — lock boosting.** Land the migration's V2 harness as
  `tests/automl/test_boosting_parity.py` (marked `slow`): the six
  train/save/load/predict/evaluate configurations, asserted against a checked-in
  JSON of metrics, `best_params` and score digests. This is the regression gate
  for everything below; without it the refactors are unverifiable.
- **S1 — backend-neutral seams.** N2 + N3 + the `fit_model` generalization. Pure
  refactor, no aliases, no new behaviour; parity must be byte-identical.
- **S2 — config.** N6 + N8 + N9: `engine="tabular_transformer"`, CPU allowed,
  hyperopt allowed for tabnn. Guards replaced by `UnsupportedBackendError`
  where a combination truly is unsupported. Also the `CUDA_VISIBLE_DEVICES`
  correction of N16 — scoped and restored, boosting-only pinning — because it
  is environment-layer work, it is covered by an existing test, and every local
  GPU run from S3 on depends on it.
- **S2b — the lazy data path and the un-expanded schema** (N12, N13, §6): the
  part descriptor on the backend contract, `TrainingCoordinator` passing
  sources, the lazy sibling of `prepare_data`, and `TabularSchema` carrying
  hidden-state widths. No tabnn code yet — this is the shared-code change and
  lands on its own, with parity byte-identical and boosting still taking
  materialized frames.
- **S2c — parallel encoding** (N12): `merge()` on both accumulators, a
  process pool over shards for `fit` and `transform`, one output file per
  shard, `order="first_seen"` rejected in that mode. It is `avatar.preprocessing`
  work with its own unit tests (a parallel fit must reproduce the serial
  `dump()` byte for byte) and touches no AutoML code, so it can land in
  parallel with S3. It is on the critical path to the first honest run on real
  data: without it every trial waits hours behind a single-process encode.
- **S3a — leak test: 50 sequential `Trainer` runs in one process** (§12, E3).
  With `accelerate` gone there is no global state to corrupt (N4b), so this is
  no longer a correctness question: it asserts that RSS, open file descriptors
  and CUDA memory stay flat across trials and that `state.best_metric` comes
  back for each. It runs before any tabnn code, because the in-process runner
  is the one that actually executes on the current hardware — if it leaks, the
  subprocess runner becomes mandatory instead of optional.
- **S3 — encoder cache + assembly + `BinaryTabNNBackend`,** `hyperopt=False`,
  CPU, global layout, late fusion (N13). First end-to-end
  `train -> save -> load -> predict -> evaluate` on synthetic data, with a
  memory assertion: peak RSS stays flat as the synthetic train split grows 10x.
  **This is v1** (§12, E5): the first result shown.
- **S4 — hyperopt for tabnn** (`fit_model` with the tabnn default space and the
  two presets), then `per_group` / `global_and_per_group`.
- **S4a — remote reliability** (N16): the `finalizing` state and idempotent
  finalization, job ids persisted per submit, the `unknown` deadline, the
  single-writer check on `operation.json`. This lands **before** S4b: trial
  fan-out multiplies every one of these by K, and the boosting fan-out that
  exists today gets the fixes for free. Parity is unaffected — none of it
  touches how a model is fitted.
- **S4b — Osiris trial fan-out** (N14): Optuna in the driver, persisted search
  state, job per trial, resumable submission, `FAIL` handling. Boosting is not
  touched.
- **S5 — regression + multiclass + response**, including `class_order`
  persistence and the multiclass metric path.
- **S6 — uplift via `SLearner`** (N10 wave 2).
- **S7 — docs + `examples/automl/tabnn_pipeline.ipynb`**, and a benchmark table
  boosting vs tabnn on the same splits, stating that the two see hidden states
  differently (N13).

## 10. Risks

| risk | mitigation |
|---|---|
| a refactor silently changes boosting results | S0 parity harness is the gate for every stage |
| GPU memory accumulates across Optuna trials | one model per trial, explicit `del` + `torch.cuda.empty_cache()` between trials; the best trial keeps CPU weights; S3a measures it |
| something global does not survive N sequential `Trainer` runs | there is nothing global left (N4b); S3a measures RSS, file descriptors and CUDA memory, and the subprocess runner is already specified (N4b) |
| AutoML's needs slowly bend `avatar.train` out of shape | the backend passes only what `Trainer` already accepts and adds nothing to it; a need that cannot be expressed as a callback is a design review, not a patch |
| `avatar` keeps moving under this design | it already did once, between the draft and this revision, and every seam it touched got better; the seams are named here (N4, N5, N12) so the next divergence is a diff, not a rewrite |
| the kept cache fills the disk | content-keyed directories, `cache_policy`, `cache_dir` on a sized volume; size it as rows × (numericals + embedding widths) × 4 B — a 768-dim embedding alone is 3 KB/row, so 100 M rows is 300 GB |
| encoding is a serial bottleneck in front of every trial | process pool over shards, mergeable accumulators, one file per shard (N12, S2c) |
| NFS cannot feed the cards | write the cache as many files, not one; if the split fits, copy it to node-local disk once at job start and read locally |
| an expired Kerberos ticket kills a long job | training jobs never read HDFS: the source is read once during encoding, the cache lives on NFS (N12) |
| a crash between submit and recording a job id duplicates work | deterministic per-trial `run_dir` (`mkdir(exist_ok=False)`) and `trial_id` in the job name, so an orphan is found instead of resubmitted (N14) |
| tabnn scales past data the boosting backend cannot load | real asymmetry, and it is the boosting side that is wrong — logged as a separate follow-up (chunked pool construction / lazy `prepare_data` for boosting), not smuggled into this design |
| NN results are not reproducible run to run | seed everything; byte-identical parity is claimed only for the local sequential CPU path, cluster runs assert metrics within tolerance (N14) |
| CatBoost GPU and torch GPU in one process | they never run in the same task; `device` is resolved per operation |
| `tabnn` quietly becomes the default | no: `backend` is a required explicit field |
| the driver dies between `succeeded` and finalization, losing a finished run | non-terminal `finalizing` state, idempotent finalization, `succeeded` written last (N16, S4a) |
| a trial job disappears from the scheduler and `wait=True` hangs all night | `unknown_job_grace_seconds` then terminal `lost` + `tell(FAIL)`; the surviving trials still finalize (N16) |
| an AutoML call narrows the notebook's GPU visibility for good | the variable is scoped to the call and restored, and pinned only for boosting (N16) |
| one OOM trial aborts a whole local tabnn search | per-trial exceptions are caught for tabnn and recorded as failed trials (N14) |

## 11. Decisions taken (2026-09-16)

1. **Engine name.** `"ste"` is rejected outright; see N6 (the chosen name was
   revised to `tabular_transformer` in §12).
2. **CPU — supported, with a warning.** See N8. `env_type="osiris"` stays GPU-only.
3. **Task order — binary -> regression/multiclass/response -> uplift,** i.e. the
   two waves of N10 as written. S6 (uplift via `SLearner`) stays last.
4. **Streaming — required.** Real datasets already exceed RAM under the boosting
   backend, so the NN backend is out-of-core by construction (N12): sources in,
   packed parquet cache, one batch resident. This is what S2b and §6 are for,
   and it is the largest piece of shared-code work in the plan.
5. **Training reuse — `avatar.train.Trainer`, not a private loop** (N4).
   `Trainer` takes plain arguments and provides DDP, AMP, early stopping,
   checkpoint-on-improvement and resume; everything else is a callback. AutoML
   contributes assembly, a metric adapter, a trial runner and prediction. On
   `master` this cost three fixes to `train.py`; on `refactor/data-sharding-hdfs`
   it costs none (N4a).

## 12. Review answers applied (2026-09-17)

The questionnaire behind this section is kept outside the repository. What was
answered, and where it landed:

| id | answer | effect |
|---|---|---|
| A1 | one package | N1 unchanged |
| A2 | rename without aliases | N2 rewritten |
| A3 | dict, no registry module | N3 rewritten, `registry.py` dropped from §4 |
| A4 | `engine="tabular_transformer"` | N6, S2 |
| B1 | `Trainer`, callbacks only | N4 |
| B2, B2a, B2b | in-process on one card, `torch.distributed.run` subprocess on several, Osiris job per trial — one interface | N4b and N4d rewritten, `runner.py` added |
| B3 | one metric adapter, by composition | N4c |
| B5 | best weights = last checkpoint | §6 |
| B6 | MLflow per trial | **N15 added**, N4 amended |
| C1 | out-of-core always, no in-memory fast path | N12; the fast-path risk row removed |
| C2 | keep the cache, content-keyed, configurable `cache_dir` | N12, §7 |
| C3 | `TabularPreprocessor`, dumped beside the artifact | N7, §8 |
| C4 | late fusion only, never expanded | **N13 added**, §5, §6, S2b, S3 |
| C5 | per-group by scan predicate | N11 |
| C6 | coordinator passes sources | §6, S2b |
| D1 | CPU allowed with a warning | N8 |
| D2 | epochs fixed + early stopping; steps on large splits | N9 rewritten, §7 |
| D3 | keep the default space | §7, minus the two axes D2 fixed |
| D4 | `amp` fixed, chosen by hardware | §7 |
| D5 | safetensors + standalone preprocessor file, no pickle | §8 |
| E1–E5 | wave split, S0 first, spike kept, order kept, v1 = S3 | §9 |
| F1–F3 | 3–100 M rows, ~200 features, 1×A100 80 GB, a night or two | §7 defaults and presets, N9 evaluation cadence |
| G1–G7 | Optuna in the driver with persisted state; no waves under `random`; `num_gpus` per trial; resumable submission; `FAIL` and continue; tabnn only; scoped reproducibility; cache on NFS | **N14 added**, S4b, §7, risks |

### Audit items required for tabnn (`obligatory.md`, 2026-09-17)

An independent audit listed eight items as prerequisites for the NN backend.
Each was re-checked against this branch; seven reproduce in the current code.

| audit item | verified at | where it lands |
|---|---|---|
| operation `succeeded` before finalization | `tasks/operations.py:375,381,388` | **N16**, S4a |
| partial submit loses job ids | `tasks/operations.py:178-208` | **N16**, S4a; the remaining window by N14 |
| no synchronization of shared lifecycle state | `lifecycle.py:182-193`, `write_json` :48 | **N16** — single-writer check, not a lock; not a parallel-jobs problem here |
| `status(wait=True)` waits forever on `unknown` | `environment.py:345,372`, `tasks/operations.py:428` | **N16**, S4a |
| `CUDA_VISIBLE_DEVICES` mutates the user's process | `environment.py:142`, no restore at :165 | **N16**, S2 |
| backend boundary hard-wired to boosting | `config/base.py`, `tasks/supervised.py` | N2, N3, N6, N8, N9; S1, S2 |
| the boosting Optuna loop is the wrong orchestrator | `hyperopt.py:336` (no `catch=`) | N14 (job per trial, driver-side Optuna) + per-trial `catch` for tabnn |
| the best NN model cannot live in coordinator RAM | `hyperopt.py:313` (`best_backend = backend`) | §6 — a trial returns a checkpoint path and a metric |

### Still open

- **V3** — the Osiris verification of the migration itself (needs the cluster).
- **Early fusion** of hidden states as a search axis (N13 defers it).
- **Wave-based TPE on the cluster** — supported by the persistence in N14,
  switched on only if a live driver turns out to be acceptable.

## 13. Cross-check against an independent design (2026-09-17)

A second design for the same migration was written independently
(`tabnn_design.md`, kept outside the repository). It reaches the same
architecture from the same starting point, which is worth recording because the
agreement was not coordinated: reuse the Avatar training stack rather than build
a second one; dispatch on `backend` and leave boosting untouched; fit the
preprocessor on train only and persist it beside the artifact; **hand hidden
states to Avatar the way Avatar takes them instead of expanding them into scalar
columns** (our N13); encode once and reuse across trials; one Osiris job per
trial, submitted without waiting, a failed trial not cancelling the others,
selection by the existing `optimization_metric`; uplift restricted to `SLearner`
or deferred; the `obligatory.md` fixes as a prerequisite (our N16).

What was taken from it:

- **Parallel encoding** (N12, S2c). The strongest idea in it and a genuine gap
  here: this design specified a streaming single-process fit and said nothing
  about its throughput, although it blocks every trial.
- **`search_strategy="grid"`** (N14): a deterministic expansion of an explicit
  finite space, which needs no persisted search state at all.

Where the two differ, and why this design keeps its choice:

- **Naming.** The other design leaves `BaseBoostingTask` and friends in place
  to keep the diff small and clean up later. N2 renames outright, without
  aliases, as decided in §12 (A2): it is a pure refactor under the S0 parity
  gate, and a rename that leaves both spellings alive tends to stay that way.
- **The artifact seam.** It proposes a `fit_model` / `predict_model` function
  pair plus teaching the repository to load a tabnn payload. `ArtifactRepository`
  already delegates to `item.backend.save(dir)` / `backend_class.load(dir)` and
  reads only `engine` from `backend.json` (`tasks/artifacts.py:50-90`), so
  following the `ModelBackend` contract leaves the artifact layer untouched —
  which better serves that design's own minimal-diff principle.
- **Optuna.** It drops Optuna for tabnn entirely. This design keeps it as
  bookkeeping under `random` (one `search_space` format for both families, one
  trial history in the report, TPE later as a one-line sampler change) and adds
  `grid` as the Optuna-free mode for the case where it is enough.

What it does not cover, and this design does: the in-memory assumption of
`TrainingCoordinator` (`tasks/training.py:83-84`) that makes every path start by
materializing both splits, which is the actual blocker at 100 M rows (S2b); the
metric bridge between `avatar.metrics` and AutoML's registry without importing
torch into `avatar/automl/**` (N4c); launcher and multi-GPU (N4b, N4d); target
hardware and the defaults derived from it (§7); cache lifetime, keying, sizing
and NFS-vs-HDFS under an expiring Kerberos ticket (N12); epochs versus steps for
early stopping on large splits (N9). Its `engine="ste"` also predates the
`STEv2 -> TabularTransformer` rename (N6).
