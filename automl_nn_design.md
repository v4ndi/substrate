# Design: neural-network backend for `avatar.automl`

Status: **proposed**, open questions resolved 2026-09-16 (§11). Stage 3 of
`repos/combine_avatar_automl/combine_avatar_fmlib_automl.md`, on top of
`automl_migration.md` (stages 1–2, done).

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

The training call chain for a supervised task is:

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
| `resolve_default_search_space(engine, …)` | per-family lookup in a registry |
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
    def prepare_fit_data(self, train, target, schema, *, valid, valid_target, reuse: bool) -> Prepared
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

## 3. Locked decisions (proposed)

- **N1 — one new package, `avatar/automl/backends/tabnn/`,** mirroring
  `backends/boosting/`: `interface.py` (shared runtime), `base.py` (training
  loop + persistence), `binary.py` / `regression.py` / `multiclass.py` /
  `uplift.py` (task adapters), `spaces.py` (default search space).
  `backends/boosting/**` keeps its files; only the three shared pieces move up.
- **N2 — backend-neutral names at the task layer, with aliases.**
  `BaseBoostingTask -> BaseTask`, `SupervisedBoostingTask -> SupervisedTask`,
  `fit_boosting_model -> fit_model` (`backends/search.py`),
  `BoostingFitResult -> FitResult`. The old names stay as deprecation aliases
  for one release — they are semi-public (`tools/`, notebooks, cluster code).
- **N3 — a backend registry,** `backends/registry.py`:
  `resolve_backend(family: str, task: str) -> type[ModelBackend]`.
  `Task._backend_class` becomes `Task._backend_class_for(config.backend)`.
  `ArtifactRepository` takes the resolved class as it does today.
- **N4 — the NN trains in-process with a plain torch loop,** not through
  `avatar/train.py`. `train.py` is a Hydra CLI built around Accelerate, MLflow,
  checkpoint directories and DDP sharding; AutoML needs N fits inside one
  process, ranked by one validation metric, with no CLI and no distributed
  setup. The new loop is ~200 lines in `backends/tabnn/trainer.py`; the
  *modules* it trains and the *dataset/collate* classes it reads through are
  avatar's, not new ones.

- **N5 — the modules are reused verbatim:**
  `avatar.nn.embedding.tabular.TabularEmbedding` → embedding,
  `avatar.nn.tabular.TabularTransformer` → encoder,
  `avatar.pipeline.tabular.TabularWithAggregatedStates` → embedding + encoder +
  pooling, `avatar.pipeline.tabular.TabularClassification` → head + loss
  (`BCEWithLogitsLoss` / `CrossEntropyLoss` / `MSELoss` already selected by
  `num_classes` + `task_type`), `avatar.pipeline.uplift.SLearner` → uplift,
  `avatar.data.tabular_batch.TabularBatch` → the batch contract.
- **N6 — the tabnn engine is `engine="transformer"`.** `"ste"` is *rejected*
  with an `UnsupportedBackendError` naming the replacement: it was only ever a
  reserved value, no tabnn artifact exists, and the class it referred to was
  already renamed `STEv2 -> TabularTransformer` in `tabular_refactor.md`. The
  `engine` string is persisted in `backend.json` and checked on load, so this is
  the last free moment to choose it. Adding an engine later (`mlp`,
  `ft_transformer`) is a registry entry plus a default search space; a
  sequence-model engine, if it ever arrives, gets its own name rather than
  overloading this one.
- **N7 — encoding reuses `avatar.preprocessing.local.TabularPreprocessor`.**
  It fits from any `pyarrow.dataset.Dataset`, so the prepared polars frame is
  handed over in memory (`ds.dataset(frame.to_arrow())`) with no parquet round
  trip. It gives exactly what `TabularEmbedding` expects: a *shared* categorical
  vocabulary with per-column offsets (`offset_map`, `vocab_size`, a reserved
  `unk` id 0 that unseen values map to) and standardized numericals
  (`mean_std`, optional `signed_log1p`). Its
  `dump()` is JSON-safe and goes straight into `backend.json`; `load()` restores
  it for inference. This also keeps AutoML's encoding identical to the encoding
  used by hand-written avatar training runs.
- **N8 — CPU is supported, with a warning.** The current `tabnn + device="cpu"`
  ban goes: the unit suite has no GPU (this host's driver is too old for torch
  CUDA), so a CPU path is the only way the backend is testable at all, and the
  boosting backend already supports both devices. `env_type="osiris"` keeps
  requiring `device="gpu"` (existing rule, unchanged). `device="cpu"` logs one
  explicit warning naming the training-row count, because CPU x `n_trials` x
  epochs is how a one-hour run silently becomes a one-day run.

- **N9 — hyperopt is enabled for tabnn.** The `"Optuna hyperopt is available
  only for boosting"` guard is removed once `fit_model` is family-agnostic.
  Epoch count is part of the search space, and every trial early-stops on the
  task's `optimization_metric` over the validation split — the same
  `objective_metric` callable the boosting backend is ranked by, so
  `validation_metrics` stay comparable across families.
- **N10 — task coverage lands in two waves.** Wave 1: `binary`, `response`,
  `regression`, `multiclass` (one `TabularClassification` head each). Wave 2:
  `uplift` via `SLearner` only — the boosting backend's S/T/X metalearner search
  is out of scope for v1, and `UpliftTaskConfig` gains no new field (an
  unsupported metalearner combination raises `UnsupportedBackendError`).
- **N11 — model-part layouts are unchanged, but partitioned lazily.** `global`,
  `per_group` and `global_and_per_group` are decided by `TrainingCoordinator`
  above the backend, so they work for free. Under `consumes="sources"` the
  per-group split becomes a scan predicate (`scan().filter(group == value)`)
  instead of `DataFrame.partition_by`, which is what makes per-group training
  possible at all when one split does not fit in RAM. Cost is unchanged in the
  docs' warning sense: `per_group` x `n_trials` x epochs is how a one-hour run
  becomes a one-day run.
- **N12 — out-of-core by construction: encode once to a packed parquet cache.**
  Real datasets already exceed RAM under the boosting backend, so the NN backend
  must not inherit AutoML's in-memory assumption. `prepare_fit_data` therefore:
  (1) fits `TabularPreprocessor` in one streaming pass (its `MeanStdAccumulator`
  / `ValueCountAccumulator` are already batched at `batch_rows=250_000`), then
  (2) `transform(..., output="packed", identity_cols=[target, treatment, …])`
  writes `<output_dir>/cache/<part>/{train,valid}/part-0.parquet` with the
  `cat_features` / `num_features` list columns that `avatar.data.TabularDataset`
  expects. Every Optuna trial then streams that cache through
  `TabularDataset` + `TabularCollateFn`; peak memory is one batch. The cache is
  the streaming analogue of CatBoost's shared quantized pool: written once,
  reused by every trial, deleted when the model part finishes (kept on
  `verbose`/debug). Encoding cost is paid once, not `n_trials` times.

## 4. Target layout

```
avatar/automl/backends/
  __init__.py
  interface.py        ModelBackend, TrainableBackend  (from boosting/interface.py)
  registry.py         resolve_backend(family, task)
  search.py           FitResult, fit_model, suggest_params  (from boosting/hyperopt.py)
  boosting/           unchanged except the three imports above
  tabnn/
    __init__.py
    interface.py      TabNNBackend: device handling, module construction
    encoding.py       TabularPreprocessor fit + packed-parquet cache (N12)
    trainer.py        the training loop: epochs, early stop, best-weights
    loader.py         TabularDataset + TabularCollateFn over the cache
    base.py           BaseTabNNBackend: prepare_fit_data / fit_prepared / save / load
    binary.py         BinaryTabNNBackend        (num_classes=1, classification)
    regression.py     RegressionTabNNBackend    (num_classes=1, regression)
    multiclass.py     MulticlassTabNNBackend    (num_classes=K, class_order in backend.json)
    uplift.py         UpliftTabNNBackend        (SLearner, wave 2)
    spaces.py         default_search_space(engine, …)
```

## 5. Data path: sources -> encoded cache -> `TabularBatch`

Nothing is materialized. The three passes over the data are:

```
1. schema        scan().collect_schema()          parquet footers only, no rows
                 + hidden-state dims              one small read
2. encoder fit   TabularPreprocessor.fit(ds)      streaming, batch_rows=250k
                   categorical_columns=schema.categorical
                   numeric_columns=schema.numerical
                   spec_tokens={"unk": 0}
3. encode        .transform(ds, out, output="packed",
                            identity_cols=[target, treatment, group, date])
                 -> cache/<part>/{train,valid}/part-0.parquet
```

`TabularPreprocessor` gives exactly what `TabularEmbedding` expects: a *shared*
categorical vocabulary with per-column offsets (`offset_map`, `vocab_size`, a
reserved `unk` id 0 that unseen values map to) and standardized numericals
(`mean_std`, optional `signed_log1p`). Its `dump()` is JSON-safe and goes
straight into `backend.json`; `load()` restores it for inference. This also
keeps AutoML's encoding identical to the encoding used by hand-written avatar
training runs.

Each trial then reads the cache with the classes that already exist:

```python
TabularDataset(path=cache / "train", shuffle_files=True, shuffle_pq=True,
               hidden_state_columns=[...])          # avatar/data/dataset/
TabularCollateFn(target_column="target", is_regression=…)  # -> TabularBatch + targets
```

Two details that are decisions, not mechanics:

- **Hidden states.** `prepare_data` flattens `hidden_state_columns` into scalar
  numerical features, which is right for a booster and lossy for a network:
  `TabularEmbedding` has a `hidden_state_aggregator`, `TabularClassification`
  has a late-fusion `extra_hidden_dim` path, and `TabularDataset` already reads
  `hidden_state_columns` natively. Under N12 the cache can simply carry the
  original list column through as an identity column, so the regrouping problem
  disappears. **Wave 1** still feeds them as plain numerical features (matches
  boosting, one less moving part); **wave 2** passes them through as
  `TabularBatch.hidden_states` and wires late fusion.
- **`TabularClassification.forward` is not hidden-state-optional today.** It
  runs `tab_features.hidden_states.isnan()` unconditionally, so it raises on
  `hidden_states=None` (and on the `dict` the `TabularBatch` docstring
  advertises). Wave 1 must fix that guard in
  `avatar/pipeline/tabular/classification.py` — a genuine pre-existing bug, not
  an AutoML concern.

## 6. Training loop (`trainer.py`)

One function, deliberately small:

```python
fit_network(model, train_cache, valid_cache, *, objective, direction,
            params, device, seed, verbose) -> FitOutcome
```

- batches come from `TabularDataset` + `TabularCollateFn` over the cache
  (`num_workers=0` — the encoding is already done, the reader is IO-bound and a
  worker pool inside an Optuna trial costs more than it saves);
- AdamW + linear warmup / cosine decay (`lr`, `weight_decay`, `warmup_ratio`);
- one validation pass per epoch, streaming the valid cache, accumulating scores
  and targets into two `float32` arrays (one value per row — that fits even when
  the feature matrix does not) and computing the task's `objective_metric`;
- `patience` epochs without improvement stops the trial; the best epoch's
  `state_dict` is kept in memory (moved to CPU) and restored at the end;
- `torch.manual_seed(random_state)` + deterministic algorithms where available,
  so a re-run of the same trial reproduces its ranking;
- progress through `avatar.automl.progress.log_progress`, in the existing
  `[stage i/n] …` format, so local and Osiris logs stay uniform.

Prediction mirrors it: `predict_score` streams the test source through the
fitted preprocessor and the model, concatenating scores — the one array that is
allowed to be O(rows).

**What this costs above the backend.** `TrainingCoordinator` currently calls
`DataPreparation.read_source` and hands `_fit_one` two materialized frames. For
`consumes="sources"` it must skip that and pass `ParquetSource` + the group
value instead; `prepare_data`'s validation work (dtype checks, hidden-state
dims, fitted-schema enforcement) gets a lazy sibling that operates on
`ParquetSource.scan()` — which already returns a `pl.LazyFrame` — plus one
sampled batch for the value-level checks. This is the single largest piece of
work in the plan and the only one that touches shared code, which is why S0's
parity harness exists before it.

## 7. Config surface

No new required fields. `model_params` (without hyperopt) and `search_space`
(with) carry the network settings, exactly as for boosting:

| parameter | default | search default |
|---|---|---|
| `hidden_size` | 128 | categorical 64 / 128 / 256 |
| `num_layers` | 3 | int 1–6 |
| `num_heads` | 8 | categorical 4 / 8 / 16 (divisor of `hidden_size`) |
| `attn_dropout` | 0.15 | float 0.0–0.4 |
| `dropout_p` (head) | 0.2 | float 0.0–0.5 |
| `out_head_hidden_dim` | 256 | categorical 128 / 256 / 512 |
| `aggregation` | `mean` | categorical mean / sum_layernorm / linear |
| `lr` | 3e-4 | float 1e-5–3e-3, log |
| `weight_decay` | 1e-2 | float 1e-6–1e-1, log |
| `batch_size` | 1024 | categorical 512 / 1024 / 4096 |
| `max_epochs` | 30 | int 5–60 |
| `patience` | 5 | fixed |

The existing `task_owned_model_params` guard (which rejects `device`, `seed`,
`verbose`, … inside `model_params`) applies unchanged.

## 8. Artifact

`<model part>/` gains, next to the existing `backend.json`:

```
backend.json        engine, params, random_state, verbose, task_state,
                    preprocessor (TabularPreprocessor.dump()),
                    dims {n_cat, n_num, vocab_size}, class_order
model.safetensors   state_dict of the assembled module
```

`engine` keeps its meaning for the manifest check, so `ArtifactRepository`,
`lifecycle.py` and the remote `run.py` spec need no change. Loading builds the
module from `params` + `dims` and then loads the tensors — no pickle, and the
artifact stays inspectable.

## 9. Staged plan

Every stage ends green on `pytest` **and** on the boosting parity harness.

- **S0 — lock boosting.** Land the migration's V2 harness as
  `tests/automl/test_boosting_parity.py` (marked `slow`): the six
  train/save/load/predict/evaluate configurations, asserted against a checked-in
  JSON of metrics, `best_params` and score digests. This is the regression gate
  for everything below; without it the refactors are unverifiable.
- **S1 — backend-neutral seams.** N2 + N3 + the `fit_model` generalization. Pure
  refactor: no new behaviour, parity must be byte-identical.
- **S2 — config.** N6 + N8 + N9: `engine="transformer"`, CPU allowed, hyperopt
  allowed for tabnn. Guards replaced by `UnsupportedBackendError` where a
  combination truly is unsupported.
- **S2b — the lazy data path** (N12, §6 "what this costs"): `consumes` on the
  backend, `TrainingCoordinator` passing `ParquetSource` for `"sources"`
  backends, and the lazy sibling of `prepare_data`. No tabnn code yet — this is
  a shared-code change and lands on its own, with parity byte-identical and the
  boosting path still taking materialized frames.
- **S3 — encoder cache + trainer + `BinaryTabNNBackend`,** `hyperopt=False`,
  CPU, global layout. First end-to-end `train -> save -> load -> predict ->
  evaluate` on synthetic data, with a memory assertion: peak RSS stays flat as
  the synthetic train split grows 10x.
- **S4 — hyperopt for tabnn** (`fit_model` with the tabnn default space), then
  `per_group` / `global_and_per_group`.
- **S5 — regression + multiclass + response**, including `class_order`
  persistence and the multiclass metric path.
- **S6 — uplift via `SLearner`** (N10 wave 2) and the hidden-state late-fusion
  path (§5).
- **S7 — docs + `examples/automl/tabnn_pipeline.ipynb`**, and a benchmark table
  boosting vs tabnn on the same splits.

## 10. Risks

| risk | mitigation |
|---|---|
| a refactor silently changes boosting results | S0 parity harness is the gate for every stage |
| GPU memory accumulates across Optuna trials | one model per trial, explicit `del` + `torch.cuda.empty_cache()` between trials; the best trial keeps CPU weights |
| the encoded cache needs disk next to `output_dir` | size it in the docs (packed float32 + int32 ≈ the source feature bytes); write under `output_dir/cache/<part>`, delete per model part, and fail early with a clear message if the write fails |
| the cache write becomes the new bottleneck for small data | `prepare_fit_data` keeps an in-memory fast path when the split is provably small (row count x width under a threshold); same code path, no cache files |
| tabnn scales past data the boosting backend cannot load | real asymmetry, and it is the boosting side that is wrong — logged as a separate follow-up (chunked pool construction / lazy `prepare_data` for boosting), not smuggled into this design |
| NN results are not reproducible run to run | seed everything; accept that GPU non-determinism makes only the *trial ranking*, not the loss, exactly reproducible — state this in the docs rather than pretending otherwise |
| CatBoost GPU and torch GPU in one process | they never run in the same task; `device` is resolved per operation |
| `tabnn` quietly becomes the default | no: `backend` is a required explicit field |

## 11. Decisions taken (2026-09-16)

1. **Engine name — `engine="transformer"`.** `"ste"` is rejected outright; see N6.
2. **CPU — supported, with a warning.** See N8. `env_type="osiris"` stays GPU-only.
3. **Task order — binary -> regression/multiclass/response -> uplift,** i.e. the
   two waves of N10 as written. S6 (uplift via `SLearner`) stays last.
4. **Streaming — required.** Real datasets already exceed RAM under the boosting
   backend, so the NN backend is out-of-core by construction (N12): sources in,
   packed parquet cache, one batch resident. This is what S2b and §6 are for,
   and it is the largest piece of shared-code work in the plan.

### Still needed before S2b

Concrete numbers for the split that does not fit, to size the cache and the
in-memory fast path: rows, feature count split into categorical / numerical /
hidden-state columns and their dimensions, the parquet size on disk, and the
RAM ceiling of the box where it OOMs. The likely first suspect is
`_expand_hidden_states`: every `hidden_state_column` of dimension *d* becomes
*d* separate `Float32` columns inside the polars frame, so a 512-dim embedding
is 512 new columns materialized in one `with_columns` call.
