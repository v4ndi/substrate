# Design: neural-network backend for `avatar.automl`

Status: **proposed** (2026-09-16). Stage 3 of
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
    def prepare_fit_data(self, frame, target, schema, *, valid_frame, valid_target, reuse: bool) -> Prepared
    def fit_prepared(self, prepared: Prepared) -> None
    def predict_prepared_score(self, features: Any) -> np.ndarray
    def can_reuse_prepared(self, search_space) -> bool
```

For `tabnn`, `Prepared` holds encoded tensors (§5) rather than CatBoost pools —
and unlike quantization, tensor encoding is *always* reusable across trials, so
`can_reuse_prepared` returns `True` whenever the search space does not touch the
encoding itself.

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
  `avatar/train.py`. `train.py` is a Hydra CLI built around Accelerate, parquet
  `IterableDataset`s, MLflow and checkpoint dirs; AutoML hands the backend two
  in-memory polars frames and expects N fits inside one process. Reusing it
  would mean serializing every trial to disk and re-reading it. The new loop is
  ~200 lines in `backends/tabnn/trainer.py`; the *modules* it trains are
  avatar's, not new ones.
- **N5 — the modules are reused verbatim:**
  `avatar.nn.embedding.tabular.TabularEmbedding` → embedding,
  `avatar.nn.tabular.TabularTransformer` → encoder,
  `avatar.pipeline.tabular.TabularWithAggregatedStates` → embedding + encoder +
  pooling, `avatar.pipeline.tabular.TabularClassification` → head + loss
  (`BCEWithLogitsLoss` / `CrossEntropyLoss` / `MSELoss` already selected by
  `num_classes` + `task_type`), `avatar.pipeline.uplift.SLearner` → uplift,
  `avatar.data.tabular_batch.TabularBatch` → the batch contract.
- **N6 — engine naming.** The canonical tabnn engine becomes
  `engine="tabular_transformer"` (the class was renamed from `STEv2` in
  `tabular_refactor.md`); `"ste"` is accepted as a deprecated alias so the
  reserved value in the shipped config keeps working. Adding an engine later
  (MLP, FT-Transformer) is a registry entry plus a default space.
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
- **N8 — CPU is supported.** The current `tabnn + device="cpu"` ban has to go:
  the unit suite has no GPU (and this host's driver is too old for torch CUDA),
  so a CPU path is the only way the backend can be tested at all. GPU stays the
  documented default and the only remote option.
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
- **N11 — model-part layouts are unchanged.** `global`, `per_group` and
  `global_and_per_group` come from `TrainingCoordinator`, above the backend, so
  they work for free — at N× the training cost. The docs will say plainly that
  `per_group` × `n_trials` × epochs is how a one-hour run becomes a one-day run.

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
    encoding.py       polars frame <-> TabularBatch via TabularPreprocessor
    trainer.py        the training loop: epochs, early stop, best-weights
    base.py           BaseTabNNBackend: prepare_fit_data / fit_prepared / save / load
    binary.py         BinaryTabNNBackend        (num_classes=1, classification)
    regression.py     RegressionTabNNBackend    (num_classes=1, regression)
    multiclass.py     MulticlassTabNNBackend    (num_classes=K, class_order in backend.json)
    uplift.py         UpliftTabNNBackend        (SLearner, wave 2)
    spaces.py         default_search_space(engine, …)
```

## 5. Encoding: polars frame -> `TabularBatch`

`prepare_data` (unchanged) already produces a frame with
`schema.categorical` (strings) and `schema.numerical` (floats, hidden-state
columns already expanded to `f"{column}__{i}"` scalars).

```
fit:     TabularPreprocessor(categorical_columns=schema.categorical,
                             numeric_columns=schema.numerical,
                             spec_tokens={"unk": 0})
             .fit(ds.dataset(frame.to_arrow()))
transform: -> cat codes  (B, n_cat) int64, globally offset, unseen -> unk
           -> num values (B, n_num) float32, standardized, NaN preserved
batch:   TabularBatch(cat_features=…, num_features=…, hidden_states=…)
```

Two details that are decisions, not mechanics:

- **Hidden states.** `prepare_data` flattens `hidden_state_columns` into scalar
  numerical features, which is right for a booster and lossy for a network:
  `TabularEmbedding` has a `hidden_state_aggregator` and
  `TabularClassification` has a late-fusion `extra_hidden_dim` path for exactly
  this. `_ModelEntry.hidden_dimensions` already records `{column: dim}`, so the
  grouping is recoverable. **Wave 1** feeds them as plain numerical features
  (simple, works, matches boosting). **Wave 2** regroups them into
  `TabularBatch.hidden_states` and wires the late-fusion path.
- **`TabularClassification.forward` is not hidden-state-optional today.** It
  runs `tab_features.hidden_states.isnan()` unconditionally, so it raises on
  `hidden_states=None` (and on the `dict` the `TabularBatch` docstring
  advertises). Wave 1 must fix that guard in `avatar/pipeline/tabular/
  classification.py` — a genuine pre-existing bug, not an AutoML concern.

## 6. Training loop (`trainer.py`)

One function, deliberately small:

```python
fit_network(model, train, valid, *, objective, direction, params, device, seed, verbose) -> FitOutcome
```

- full-batch tensors already on the device when they fit, otherwise an index
  sampler over pinned CPU tensors — no `DataLoader` workers, no parquet;
- AdamW + linear warmup / cosine decay (`lr`, `weight_decay`, `warmup_ratio`);
- one validation pass per epoch computing the task's `objective_metric`;
- `patience` epochs without improvement stops the trial; the best epoch's
  `state_dict` is kept in memory and restored at the end;
- `torch.manual_seed(random_state)` + deterministic algorithms where available,
  so a re-run of the same trial reproduces its score;
- progress through `avatar.automl.progress.log_progress`, in the existing
  `[stage i/n] …` format, so local and Osiris logs stay uniform.

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
- **S2 — config.** N6 + N8 + N9: engine alias, CPU allowed, hyperopt allowed for
  tabnn. Guards replaced by `UnsupportedBackendError` where a combination truly
  is unsupported.
- **S3 — encoding + trainer + `BinaryTabNNBackend`,** `hyperopt=False`, CPU,
  global layout. First end-to-end `train -> save -> load -> predict -> evaluate`
  on synthetic data.
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
| in-memory training does not scale like the streaming `train.py` | AutoML already materializes both splits as polars frames — the NN adds tensors of the same order; document the ceiling and fall back to index-sampled CPU tensors |
| NN results are not reproducible run to run | seed everything; accept that GPU non-determinism makes only the *trial ranking*, not the loss, exactly reproducible — state this in the docs rather than pretending otherwise |
| CatBoost GPU and torch GPU in one process | they never run in the same task; `device` is resolved per operation |
| `tabnn` quietly becomes the default | no: `backend` is a required explicit field |

## 11. Open questions

1. **Engine name.** `engine="tabular_transformer"` with `"ste"` as an alias
   (N6), or keep `"ste"` as the only public name until the avatar -> fmlib
   rename?
2. **CPU support** (N8) — acceptable, or must tabnn stay GPU-only even though
   that leaves the backend untestable in CI?
3. **Task order.** Is `uplift` wanted earlier than wave 2? It is the task where
   a network most plausibly beats the boosting S/T/X metalearners.
4. **Streaming.** Does any real dataset already fail to fit in memory under the
   boosting backend? If yes, the NN backend should read parquet directly
   (`avatar.data.TabularDataset`) instead of inheriting AutoML's in-memory
   assumption, and that changes §6 substantially.
