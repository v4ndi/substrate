# Design: repository documentation

Status: **implemented** (2026-08-31). All five phases landed; the numbers below
describe the state this work started from and are kept as written.

Written against `refactor/train-torch-distributed` (6ad5902), i.e. the state
after the `avatar/data` and `avatar/train` rewrites. All numbers below are
measured from that tree, not estimated.

## What changed against the plan

* **§0.4's target census counted `experiments/`.** In what a user actually
  reads — docs and examples — it is 25 distinct targets in 148 uses, with 9
  more used only under `experiments/`.
* **`mlflow:` is required for training**, not optional as §0.5 implied:
  `init_exp_run_name` reads it unconditionally and the checkpoint path is built
  from it.
* **A third dead config key** turned up beside `device_specific`:
  `mlflow.logging_dir` is in all 30 configs and is read by no code.
* **MkDocs was not adopted**, as §8 recommended. Coverage is now high enough
  that this is worth revisiting.
* **MLM for event sequences was not added** (§6, item 5). There is no such
  pipeline in the repository; writing one is model work, not documentation.
  Next-token and next-K are both covered by configs.
* **Six code bugs** were found by writing the checks and the runnable examples,
  and fixed: a deleted `MLPEmbedding` that a documented benchmark mode still
  needed; `amp: no` failing every config in the repo; `TabularClassification`
  unable to run at all; a collate deleting its own output; `RocAucScore`
  rejecting single-logit heads; and both synthetic generators giving train and
  valid different target functions.

---

## 0. Measured current state

### 0.1 Docstring coverage

| package | LOC | module docstrings | classes | functions |
|---|---:|---|---|---|
| `avatar/data` | 3667 | 52% (12/23) | 72% (16/22) | 83% (74/89) |
| `avatar/metrics` | 2687 | 0% (0/13) | 56% (14/25) | 29% (28/95) |
| `avatar/pipeline` | 2685 | 0% (0/15) | 42% (11/26) | 8% (3/37) |
| `avatar/preprocessing` | 2637 | 59% (13/22) | 75% (12/16) | 54% (45/82) |
| `avatar/train` | 2627 | 100% (24/24) | 90% (19/21) | 42% (41/96) |
| `avatar/nn` | 2293 | 15% (6/38) | 97% (33/34) | 56% (26/46) |
| `avatar/losses` | 1061 | 30% (3/10) | 26% (5/19) | 23% (5/21) |
| `avatar/utils` | 650 | 33% (1/3) | 50% (1/2) | 45% (10/22) |
| `avatar/*.py` | 524 | 55% (5/9) | 92% (12/13) | 66% (4/6) |
| **total** | **18831** | **40% (64/157)** | **69% (123/178)** | **47% (236/494)** |

Two patterns worth naming, because they need different fixes:

- **`avatar/nn` and `avatar/pipeline` document the class and nothing else.**
  `nn` is at 97% on classes and 15% on modules: every building block has a
  docstring, but nothing says which blocks compose with which. `pipeline` is
  worse — 42% of classes, 8% of functions, 0% of modules, over 2685 LOC that
  85 configs instantiate.
- **`avatar/train` is the mirror image**: 100% module docstrings, 90% classes,
  and no user-facing prose at all. Coverage is not the problem there; the
  absence of a "how do I launch a distributed run" page is.

So docstring coverage and documentation debt are **two different axes**, and
the plan treats them separately (§2).

### 0.2 What prose exists today

| file | lines | state |
|---|---:|---|
| `README.md` | 62 | current, thin — install, layout, tests, backends |
| `docs/data/event_sequence_batch.md` | 256 | good, current |
| `docs/data/working_with_parquet.md` | 132 | **stale** — documents `avatar.data.parquet`, which moved |
| `docs/best_practices/gradient_checkpointing.md` | 214 | current (updated during the train rewrite) |
| `docs/mlp_benchmark/instruction.md` | 231 | one broken target |
| `examples/README.md` | 15 | index only |
| 6 × `examples/*/README.md` | — | quality varies; 6 broken references between them |

Plus five root-level documents that are **not user documentation**:
`data_redesign.md`, `train_redesign.md`, `sequential_refactor.md`,
`tabular_refactor.md`, `MAINTENANCE.md`. These are decision records — valuable,
but they sit where a newcomer looks for a getting-started guide.

There is **no `docs/` index**, no navigation, and no stated ownership.

### 0.3 The docs are already wrong in ten places

A script resolving every `avatar.*` dotted path mentioned in markdown (skipping
the planning docs, which describe past states on purpose) finds **8 of 45
references broken**:

| broken reference | where |
|---|---|
| `avatar.data.parquet` | `docs/data/working_with_parquet.md` |
| `avatar.nn.tabular.STEv2`, `…STEv2Body`, `…BaseTabularBackbone` | `examples/tabular_hidden_states/README.md` |
| `avatar.nn.TabularWithAggregatedStates`, `avatar.nn.utils.agg.py` | `examples/tabular_hidden_states/README.md` |
| `avatar.nn.tabular.ste.STEv2Body` | `examples/uplift_modeling/s_learner/README.md` |
| `avatar.nn.tabular.MLPEmbedding` | `docs/mlp_benchmark/instruction.md` |

Two more, not dotted paths:

- `examples/basics/README.md` tells the reader to run `./install.sh`, which
  **does not exist** in the repository.
- The same file clones from `df-bitbucket.ca.sbrf.ru`; the remote is now
  `github.com:v4ndi/substrate`.

This is the strongest argument in this document: writing more prose without
adding a drift check just increases the amount of text that will be wrong in
six months. Enforcement (§7) is a prerequisite, not a follow-up.

### 0.4 What users actually configure

Across the 60 YAML files that instantiate anything there are **34 distinct
`avatar.*` Hydra targets in 525 uses**. That is the real public API — far
smaller than the 494 public functions.

| module | uses | docstring state |
|---|---:|---|
| `avatar.data` | 192 | good (83% functions) |
| `avatar.nn` | 125 | classes only |
| `avatar.metrics` | 117 | **0% modules, 29% functions** |
| `avatar.pipeline` | 80 | **0% modules, 8% functions** |
| `avatar.train` | 7 | good |
| `avatar.losses` | 4 | 23% functions |

The top ten targets carry 80% of the uses: `TabularDataset` (86),
`TabularEmbedding` (54), `TabularTransformer` (54),
`GroupAverageMetricWrapper` (50), `UpliftCollateFn` (45), `SLearner` (45),
`TabularCollateFn` (41), `InferenceMultiTaskCampaignMetrics` (23),
`UpliftMetrics` (21), `IgnoreTreatmentInteraction` (12).

**Documentation effort should be ordered by this table, not by LOC.**

### 0.5 The config schema is undocumented

30 run configs share an identical top-level shape, and there is no reference
for it anywhere:

```
distributed  amp  ddp  model  optimizer  scheduler  train
train_dataloader  valid_dataloader  [test_dataloader]  mlflow  metrics
[swa_model]  [product]  [PATH_TO_SAVE_PREDICT]
```

with `train:` holding `start_epoch`, `checkpoint_state`, `model_state`, `seed`,
`num_epochs`, `clip_grad_norm`, `max_saved_checkpoints`, `device_specific`,
`early_stopping`, `steps_before_evaluation`.

Two findings fall straight out of this census:

- **`device_specific` is in all 30 configs and is a documented no-op** —
  deprecated, always False. Nobody can tell that from the config.
- **The `logging:` block appears in zero configs**, yet `avatar/train/factory.py`
  reads `logging.enable`, `logging.enable_profiler` and
  `logging.performance_metrics`. The entire performance-metrics subsystem — the
  shard scan metrics, throughput, system counters — is reachable only by
  someone who reads the source. That is a documentation gap that is also,
  effectively, a dead feature.

### 0.6 No enforcement, no build

- No `.github/` — **there is no CI at all**. Any check has to run through
  `pytest` or `pre-commit`, both of which exist and work.
- `ruff` selects `E, F, I, W, B, UP, RUF`. **No `D` (pydocstyle) rules**, so
  docstring presence and style are unenforced; the 40/69/47% above is what
  unenforced looks like.
- No Sphinx / MkDocs / pdoc. Markdown in the repo is the only channel.
- `.pre-commit-config.yaml` and `[tool.ruff]` both exclude `*.ipynb`. The seven
  notebooks under `examples/` are linted by nothing and rot silently.

---

## 1. Audiences

Three, with genuinely different needs. Every item in §4–§6 names which one it
serves.

- **(A) The config author.** Wants to train a model on their data without
  reading Python. Needs: the config schema, a target reference for the 34
  instantiable classes, a launch guide, and worked examples to copy. This is
  the largest audience and the worst served today.
- **(B) The extender.** Adding a metric, a loss, a callback, a pipeline. Needs:
  the extension points and their contracts (`TrainerCallback`, `Loss`/
  `LossOutput`, `BaseMetric`, the dataset/collate contract), plus one small
  worked example per point.
- **(C) The maintainer.** Needs: architecture orientation, the invariants that
  are easy to break (rank/batch-count parity, scan/iterate predicate parity,
  callback ordering, state-dict key stability), and the decision records.

---

## 2. Two axes, two mechanisms

| | reference | guides |
|---|---|---|
| **lives in** | docstrings, next to the code | `docs/*.md` |
| **answers** | "what does this argument do" | "how do I do X", "why is it this way" |
| **audience** | B, C | A, B, C |
| **rots when** | signature changes | module moves, behaviour changes |
| **enforced by** | ruff `D` rules (§7) | reference check + config check (§7) |

The rule this plan proposes: **a docstring never explains architecture, and a
guide never restates a signature.** Guides link to the code; docstrings link to
guides. Duplicating between them is what produced the eight broken references.

---

## 3. Target layout

```
README.md                     # entry point: what this is, install, 5-minute path, map
TODO.md                       # unchanged
docs/
├── README.md                 # index — the only navigation surface
├── getting_started.md        # install -> preprocess -> train -> infer, one runnable path
├── configuration/
│   ├── schema.md             # every top-level key, every train: key, defaults
│   ├── targets.md            # the 34 instantiable targets, grouped, with links
│   └── distributed.md        # distributed:/amp:/ddp:/compile:, torchrun, multi-node
├── guides/
│   ├── training.md           # the loop, epochs vs steps, evaluation, checkpoints, resume
│   ├── callbacks.md          # extension point: events, ordering, writing one
│   ├── losses.md             # extension point: Loss/LossOutput, injection, composition
│   ├── metrics.md            # extension point: BaseMetric, wrappers, distributed gather
│   ├── datasets.md           # sharding, HDFS, filter cache, the parity contract
│   ├── preprocessing.md      # the two backends, artifact interchange
│   └── models.md             # how nn blocks compose into a pipeline
├── reference/
│   ├── data.md  metrics.md  pipeline.md  nn.md  train.md  losses.md
│   └──                       # one orientation page per package: what's here, what to use
├── best_practices/
│   ├── gradient_checkpointing.md      # exists
│   └── performance_metrics.md         # NEW — the undocumented logging: block
└── decisions/                # moved from the repo root, unchanged content
    ├── data_redesign.md  train_redesign.md
    ├── sequential_refactor.md  tabular_refactor.md
    └── maintenance.md
```

`docs/data/event_sequence_batch.md` folds into `docs/reference/data.md`;
`docs/data/working_with_parquet.md` is rewritten against `avatar.data.base.parquet`
and folded into `docs/guides/datasets.md`. `docs/mlp_benchmark/` stays where it
is — it is a benchmark recipe, not library documentation.

Moving the five root documents into `docs/decisions/` is a rename only. It
costs nothing and stops the root from reading like a scratchpad.

---

## 4. Module coverage plan

Ordered by config-facing weight (§0.4) crossed with current coverage (§0.1),
not by LOC.

### Tier 1 — heavily configured, effectively undocumented

**`avatar/metrics`** (2687 LOC, 117 config uses, 0% module docstrings, 29% functions)
— *audience A, B*
The second most configured module in the repo and the least documented.
`GroupAverageMetricWrapper` alone appears 50 times and has no prose anywhere.
- Module docstrings for all 13 modules: what family of metric lives here.
- Class + `update`/`compute`/`reset` docstrings for the 9 config-facing metric
  classes.
- `docs/guides/metrics.md`: the `BaseMetric` contract, when `update` is fed
  gathered vs local pairs, what the wrappers do, how to write one.
- `docs/reference/metrics.md`: the catalogue.

**`avatar/pipeline`** (2685 LOC, 80 config uses, 0% modules, 8% functions)
— *audience A, B*
`multi_task/` is 1543 LOC with **zero documented functions**. `SLearner` is the
single most used pipeline (45).
- Module docstrings for all 15 modules.
- Full docstrings on the six config-facing pipelines, each stating: which batch
  it consumes, which output dataclass it returns, and which loss it defaults to.
- `docs/guides/models.md`: backbone → pipeline → loss composition, and the rule
  that `avatar/nn/**` never computes a loss.
- `docs/reference/pipeline.md`.

**The config schema** — *audience A*
Nothing exists. Highest value per page in the whole plan: it is what 30 configs
and every new user need.
- `docs/configuration/schema.md` — every key, its type, default, and whether it
  is required. Explicitly mark `device_specific` as a deprecated no-op.
- `docs/configuration/targets.md` — the 34 targets, grouped by role, each with
  its required arguments and a link to the class.
- `docs/configuration/distributed.md` — `distributed:` / `amp:` / `ddp:` /
  `compile:`, the `torchrun` launch contract, multi-node, and the migration
  table from the old `accelerator:` block.

### Tier 2 — configured, partially documented

**`avatar/nn`** (2293 LOC, 125 config uses, 97% classes / 15% modules)
— *audience A, C*
The classes are documented; the composition is not. Needs orientation, not
docstrings.
- Module docstrings for the 32 undocumented modules (one or two lines each:
  what lives here, what it composes with).
- `docs/reference/nn.md`: the block catalogue — embeddings, encoders,
  backbones, aggregations — and which combine with which.
- Fix the four broken `avatar.nn.tabular.*` references in
  `examples/tabular_hidden_states/README.md`.

**`avatar/train`** (2627 LOC, 100% modules, 42% functions) — *audience A, B*
Coverage is fine; there is no guide. This is the newest subsystem and the one
people will get wrong.
- `docs/guides/training.md`: the loop, per-epoch vs per-step evaluation, the
  checkpoint format and layout, resume, what `TrainingArguments` controls.
- `docs/guides/callbacks.md`: the twelve events, the `ctx` contract, the one
  ordering constraint (early stopping before the checkpointer), the rule that
  hooks fire on every rank, and how to write one.
- `docs/best_practices/performance_metrics.md`: the `logging:` block, what each
  emitted metric means, and the cost of enabling it. Currently source-only.
- Docstrings for the 55 undocumented functions — mostly the private callback
  internals, so this is Tier 3 work in practice.

**`avatar/losses`** (1061 LOC, 26% classes, 23% functions) — *audience B*
Newly given a contract (`Loss` / `LossOutput`) that nothing explains.
- Docstrings for the 14 undocumented classes — the research losses in
  `kld_loss.py` and `multi_task.py` especially, where the name alone does not
  say what is being computed.
- `docs/guides/losses.md`: the contract, injection from config, `CompositeLoss`,
  and when `num_items` is needed (multi-head, token-weighted across ranks).

### Tier 3 — used through Python, decent coverage

**`avatar/preprocessing`** (2637 LOC, 59% modules, 54% functions) — *audience A*
Zero config uses because it is driven from notebooks and scripts, but it is the
first thing anyone touches. `local/pipeline/` is the weak spot: 634 LOC at 5%
function coverage.
- Docstrings for `local/pipeline/` and the 8 undocumented modules under `spark/`.
- `docs/guides/preprocessing.md`: the two backends, artifact interchange, when
  to pick which. Some of this exists in the two example READMEs and should be
  consolidated rather than rewritten.

**`avatar/data`** (3667 LOC, 192 config uses, best-covered module) — *audience A, C*
Docstrings are in good shape after the rewrite. What is missing is the prose
that currently only exists inside `data_redesign.md`.
- `docs/guides/datasets.md`: record-level sharding, `drop_tail` / `rotate_tail`,
  the scan/iterate parity contract, the filter cache, HDFS. Extract from the
  decision record, do not link to it.
- Rewrite `working_with_parquet.md` against `avatar.data.base.parquet`.

### Tier 4 — internal

`avatar/utils`, `avatar/data/sampler`, `avatar/nn/utils`. Docstrings on public
functions when touched; no guides. *Audience C only.*

---

## 5. Docstring policy

Adopt what the codebase already mostly does, and write it down:

- **Google style** (`Args:` / `Returns:` / `Raises:`) — already the de facto
  convention.
- **Every module** gets a one-to-three-line docstring saying what lives there
  and what it composes with. This is the cheapest, highest-value line in the
  file and coverage is 40%.
- **Every public class and function** in a module that appears in
  `docs/configuration/targets.md` gets full `Args:`/`Returns:`.
- **Private helpers** get a docstring only when the *why* is non-obvious. A
  docstring restating the signature is noise.
- **Comments explain why, docstrings explain what.** The invariants — rank/
  batch-count parity, predicate parity, callback ordering, state-dict key
  stability — belong in comments at the point of risk, and are already written
  that way in `avatar/data/base` and `avatar/train`.

---

## 6. Examples plan

Current: `basics`, `tabular_preprocessing`, `eventsequence_preprocessing`,
`tabular_hidden_states`, `next_event_prediction`, `uplift_modeling/s_learner`.
Format is mixed — `.ipynb` for narrative, `.py` for the preprocessing
comparisons.

**Policy proposal:** anything that must keep working is a `.py` script plus a
README; notebooks are for narrative only. Notebooks are excluded from ruff and
pre-commit (§0.6), so they cannot be kept honest and should not carry
load-bearing API usage.

### Fix first (drift, §0.3)

| example | fix |
|---|---|
| `basics` | remove `install.sh`, correct the clone URL |
| `tabular_hidden_states` | 5 broken `avatar.nn.*` references |
| `uplift_modeling/s_learner` | 1 broken reference |

### Add — ordered by gap severity

1. **`examples/distributed_training/`** — *audience A*. The `avatar/train`
   rewrite added multi-GPU and multi-node support and **there is no example at
   all**. A `torchrun` launcher script, a config with the `distributed:` block,
   a note on what sharding guarantees, and the single-GPU fallback. Highest
   priority: it is a new capability nobody can discover.
2. **`examples/custom_callback/`** — *audience B*. ~40 lines: a callback that
   logs something per epoch, registered through `callbacks:` in config. The
   extension point is new and has no worked example.
3. **`examples/custom_loss/`** — *audience B*. Same shape, for `Loss` /
   `LossOutput` injection into a pipeline.
4. **`examples/multi_task/`** — *audience A*. 1543 LOC of multi-task pipeline
   with 23 config uses of `InferenceMultiTaskCampaignMetrics` and no example.
5. **Extend `next_event_prediction`** — already listed as a TODO in
   `examples/README.md`: MLM and next-k variants beyond next-token.

### Rewrite `examples/README.md`

Into a table: example → what it teaches → which audience → runnable as
(notebook / script / torchrun) → data prerequisite. The current bullet list
does not say which examples need internal NFS data, which is the first thing a
new reader hits.

---

## 7. Enforcement

Without CI (§0.6), everything runs through `pytest` and `pre-commit`. Both are
already wired up.

1. **`tests/docs/test_doc_references.py`** — resolve every `avatar.*` dotted
   path in every markdown file; fail on anything that does not import.
   *This check, written while surveying, already finds the 8 real breaks in
   §0.3.* Cheapest, highest-value item in the plan: roughly 40 lines, and it is
   what keeps the rest of the work from decaying.
2. **`tests/docs/test_config_targets.py`** — resolve every `_target_` in every
   YAML. Also already written and already finds 2 genuinely broken targets
   (`avatar.nn.tabular.MLPEmbedding`, `avatar.pipeline.uplift.SLearnerExp`).
   Those two are pre-existing bugs, not doc bugs; the test would have caught
   them.
3. **`tests/docs/test_referenced_paths.py`** — every repo-relative path
   mentioned in a fenced block or inline code in markdown exists. Catches
   `install.sh`.
4. **ruff `D` rules, phased.** Enable `D` with
   `convention = "google"`, then use `per-file-ignores` to exempt everything
   except the packages already at high coverage (`avatar/train`, `avatar/data`).
   Remove exemptions one package at a time as §4 lands. Enabling `D` repo-wide
   at once would produce ~300 errors and get switched off.
5. **Notebook lint.** Either add `nbqa` to pre-commit, or accept the policy in
   §6 that notebooks carry no load-bearing API usage. Recommend the latter — it
   is free.

Items 1–3 are one small test package and should land **before** any new prose
is written.

---

## 8. Tooling: why not MkDocs yet

Deliberate recommendation: **stay with plain markdown for now.**

- There is no CI, so there is nothing to publish a built site from. A local-only
  site is worse than markdown the reader can already see in the repo browser.
- `mkdocstrings` autodoc over a codebase at 47% function coverage produces
  mostly empty pages, which reads as *worse* documented than no page.
- The gap identified in §0 is guides and a config reference — hand-written prose
  that a generator cannot produce.

Revisit after Phase 3, when coverage is high enough for autodoc to be worth
reading and there is a CI job to publish it. At that point the `docs/` tree
above is already MkDocs-shaped and the migration is a `mkdocs.yml` plus a
navigation block.

---

## 9. Phases

Each phase is independently useful and ends in a state worth merging.

**Phase 0 — stop the rot** (small)
Doc-reference, config-target and path tests (§7.1–7.3); fix the 10 known breaks
(§0.3); move the five decision records into `docs/decisions/`; add
`docs/README.md` as the index. Nothing new is written; everything that exists
becomes true.

**Phase 1 — the config author** (largest, highest value)
`docs/configuration/{schema,targets,distributed}.md`; `docs/getting_started.md`;
rewrite `README.md` around a five-minute path; rewrite `examples/README.md` as
a table. Serves audience A, who are the worst served today.

**Phase 2 — the two dark modules**
`avatar/metrics` and `avatar/pipeline`: module docstrings everywhere,
full docstrings on the config-facing classes, `docs/guides/metrics.md`,
`docs/guides/models.md`, `docs/reference/{metrics,pipeline}.md`. Enable ruff
`D` for both packages as the work lands.

**Phase 3 — extension points**
`docs/guides/{training,callbacks,losses,datasets}.md`;
`docs/best_practices/performance_metrics.md`; the three small examples
(distributed training, custom callback, custom loss). Serves audience B and
documents everything the two recent rewrites added.

**Phase 4 — the long tail**
`avatar/nn` module docstrings and reference; `avatar/preprocessing`
`local/pipeline` docstrings and guide; `examples/multi_task`; extend
`next_event_prediction`; remaining ruff `D` exemptions removed.

Phases 0 and 1 deliver most of the value. If only one phase happens, it should
be Phase 0 — it is small, and it is what makes every later phase durable.

---

## 10. Non-goals and risks

**Non-goals.** Translating the repository to one language (it is mixed
Russian/English today and that is a separate decision); documenting
`experiments/`; publishing a site; API docs for private helpers.

**Risks.**

- **Prose without enforcement decays.** Already demonstrated: 8 of 45
  references broken, and the two refactors that broke them shipped with tests
  passing. Phase 0 is the mitigation and must go first.
- **Documenting a moving target.** `avatar/train` and `avatar/data` were just
  rewritten and `train_redesign.md` §6.5 still lists open follow-ups. Guides
  for those two should be written after their follow-ups land, or written
  narrowly enough to survive them.
- **Docstring churn for its own sake.** 258 undocumented public functions is a
  large number, and most of them are private-in-spirit helpers. The policy in
  §5 — full docstrings only for config-facing API, one line per module
  everywhere — is what keeps this from becoming busywork.
- **Language choice per document.** The existing corpus is mixed: `README.md`
  and the decision records are English, `examples/README.md` and
  `MAINTENANCE.md` are Russian. Pick per audience and be consistent within a
  document; do not mix inside one page.
