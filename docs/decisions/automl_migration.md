# Design: migrate `fmlib.automl` -> `avatar.automl`

Status: **done** (2026-09-16), except V3 (Osiris, cluster-only). Stages 1-2 of
`repos/combine_avatar_automl/combine_avatar_fmlib_automl.md`.

Source: `repos/combine_avatar_automl/automl/fmlib-main` (`sber-amazme-fmlib`,
GitHub `ConstantIrritation/automl`), package `fmlib/automl`.
Target: `repos/combine_avatar_automl/substrate` (`avatar`, GitHub
`v4ndi/substrate`), new package `avatar/automl`.

After the whole migration `avatar` is renamed back to `fmlib`, so the eventual
import path is `fmlib.automl` again. Every rename below is therefore chosen to
survive that second rename without another sweep.

---

## 0. What is being moved

`fmlib/automl` — 91 `.py` files, 16 966 lines (9 795 non-test + 30 test modules
with 267 test functions).

```
automl/
  config/      typed frozen dataclass configs (+ from_mapping/from_yaml)
  data/        ParquetSource, FeatureSchema, CanonicalColumnMapper, prepare_data
  backends/
    boosting/  CatBoost/XGBoost adapters + Optuna hyperopt loop
  calibrators/ isotonic + beta calibration
  metrics/     registry: binary / multiclass / regression / uplift
  tasks/       Binary/Response/Regression/Multiclass/Uplift facades
               + planning/routing/preparation/training/prediction/evaluation/
                 calibration/artifacts/operations/state
  environment.py  local vs Osiris runners
  lifecycle.py    artifact save/load/restore
  reporting.py    matplotlib figures + XlsxWriter reports
  run.py          remote worker entrypoint (`--spec run_spec.json`)
  result_io.py, execution.py, progress.py, exceptions.py, types.py
```

**Key fact that makes this cheap:** `fmlib/automl` imports *nothing* from the
rest of `fmlib`.

```
$ grep -rhoE "from fmlib\.[a-zA-Z0-9_.]+|import fmlib\.[a-zA-Z0-9_.]+" fmlib/automl \
    | grep -v "fmlib\.automl"
(no output)
```

All 263 `fmlib` references inside the package are `fmlib.automl.*`. The module
is a self-contained, polars-based, sklearn/boosting subsystem that happens to
live in a torch repo. Migration is a copy + a namespace rewrite, not a port.

## 1. Gap analysis: source vs target

| | `fmlib` (source) | `avatar` (target) | action |
|---|---|---|---|
| python | `>=3.11,<3.13` | `>=3.10` | bump avatar to `>=3.11` (**D3**) |
| build | poetry | setuptools + `pyproject` | keep avatar's |
| dataframes | polars (automl), pyarrow (nn) | pandas + pyarrow | add `polars` dep |
| tests | in-package `**/tests/` | top-level `tests/`, `testpaths=["tests"]` | move (**D2**) |
| ruff | line-length 128, ~25 rule sets | line-length 88, `E,F,I,W,B,UP,RUF` | reformat (**D4**) |
| deps missing in avatar | — | — | `polars`, `optuna`, `xgboost`, `scikit-learn`, `scipy`, `matplotlib`, `XlsxWriter` |
| deps already in avatar | — | `catboost` (extra), `betacal`, `omegaconf`, `pandas`, `numpy<2` | reuse |
| `osiris` | optional, lazy `import osiris` | absent | stays optional (**D6**) |

Name collisions inside `avatar` — none. `avatar.metrics.uplift`,
`avatar.metrics.campaign` and `avatar.pipeline.uplift` are torch/pandas code;
the AutoML copies live under `avatar.automl.metrics` / `avatar.automl.tasks`
and never meet them on an import path. Deduplicating them is explicitly **out
of scope** (follow-up F2).

## 2. Locked decisions

- **D1 — one package, verbatim first.** `fmlib/automl/**` lands at
  `avatar/automl/**` with exactly one content change: `fmlib.automl` ->
  `avatar.automl` in imports and docstrings. No refactor, no re-layout, no
  renames in the same commit. Everything else is a separate, reviewable commit.
- **D2 — tests move to `tests/automl/`,** mirroring the package path
  (`avatar/automl/tasks/tests/test_binary.py` -> `tests/automl/tasks/test_binary.py`).
  Rationale: avatar sets `testpaths = ["tests"]`, and
  `[tool.setuptools.packages.find] include = ["avatar*"]` would otherwise ship
  `avatar.automl.tests` (and its 7 200 lines) inside the wheel — the top-level
  `exclude = ["tests*"]` does not match nested packages.
- **D3 — `requires-python = ">=3.11"`.** `typing.Self` (used by
  `backends/boosting/interface.py`) is 3.11+. avatar's own code is 3.10-clean
  but nothing runs on 3.10 here; the dev venv is 3.12. Cheaper than rewriting
  `Self` to a `TypeVar`.
- **D4 — style alignment is its own commit,** run after the test suite is green
  on the verbatim copy: `ruff format` at avatar's line-length 88 + `ruff check
  --fix` for `E,F,I,W,B,UP,RUF`. Diff is large but mechanical, and the green
  suite from the previous commit is the guard.
- **D5 — artifact-format strings are frozen.** `__FMLIB_NULL__` /
  `__FMLIB_UNKNOWN__` (`backends/boosting/base.py`) are baked into saved
  XGBoost category vocabularies, and `backend.json` / `manifest.json` keys are
  read by already-trained artifacts on the cluster. They are **not** renamed to
  `AVATAR`, now or after the avatar->fmlib rename. Same for the
  `binary_model`/`uplift_model`… artifact directory names.
- **D6 — Osiris stays optional and untested locally.** `environment.py` already
  lazy-imports `osiris` and raises `MissingDependencyError`. Only
  `env_type="local"` is exercised by the migrated suite; the remote path is
  verified on the cluster after the move (V3).
- **D7 — `backend="tabnn"` / `engine="ste"` are kept as-is** in stage 2 even
  though avatar's class is now `TabularTransformer` (`ste/` is a deprecation
  shim). Renaming the public enum value belongs to the neural-network design
  (stage 3), not to the migration.
- **D8 — `run.py` stays a module entrypoint** (`python -m avatar.automl.run
  --spec ...`). The generated Osiris command string is updated accordingly and
  is the one place where the import path leaks into runtime strings.

- **D9 — the `autocampaignxfm` parity harness is not migrated.** `tools/automl_parity`
  (1 949 lines) and the three `examples/automl/tests/*_autocampaignxfm_parity.ipynb`
  notebooks with their `parity_*.yaml` configs are dropped. They target the
  configuration API from before `model_scope`/`report_month_column` were removed,
  and the `autocampaignxfm` reference package is in neither repository: run
  against the *source* repo they are 22 failed / 6 passed / 4 skipped, i.e.
  already dead there. Restoring them is one `git checkout` from
  `automl/fmlib-main`, should the reference implementation resurface.

## 3. Execution plan

### Stage A — environment (no repo changes)

1. `uv venv --python 3.12 .venv` in `substrate/`.
2. Add the new dependencies to `pyproject.toml` (see §1). `avatar.automl` is
   not optional, so its deps go into `[project.dependencies]` — including
   `polars`, which appears in the public result types (`avatar.automl.types`).
   The two native boosting engines stay optional: `catboost` keeps its existing
   extra and `xgboost` joins it, so a pure-torch install stays slim and the
   engines raise `MissingDependencyError` when absent (they already do).
3. `uv pip install -e .[spark,catboost,dev]` + `polars optuna xgboost
   scikit-learn scipy matplotlib XlsxWriter` via the proxy from
   `~/rusakov/init.sh`.
4. Baseline: `pytest` on avatar (expect the current 272 passed / 2 skipped).

### Stage B — verbatim copy (commit 1)

1. `cp -r automl/fmlib-main/fmlib/automl substrate/avatar/automl`.
2. `grep -rl 'fmlib\.automl' | xargs sed -i 's/fmlib\.automl/avatar.automl/g'`;
   fix the three prose occurrences (`fmlib AutoML` in module docstrings ->
   `avatar AutoML`) and the `fmlib_automl_progress_logger` ContextVar name.
3. `python -c "import avatar.automl"`, then `pytest avatar/automl -q`
   (tests still in-package at this point, run by explicit path).
4. **Gate:** 267 test functions must pass with zero skips other than the
   engine-availability ones.

### Stage C — layout + style (commits 2–3)

1. `git mv` the 30 test modules to `tests/automl/**` per D2; delete the empty
   `**/tests/` packages; add `tests/automl/conftest.py` if any fixture relied on
   package-relative paths.
2. `pytest` (whole repo) — avatar's 272 + automl's 267.
3. `ruff format` + `ruff check --fix` per D4; `pytest` again.

### Stage D — docs, examples, tools (commit 4)

1. `examples/automl/` — 5 pipeline notebooks + 12 test notebooks + 4 YAML
   configs. Copy to `substrate/examples/automl/`, rewrite imports, and replace
   the installation section (fmlib poetry / shared `/home/datalab/nfs/...` env)
   with avatar's `pip install -e .`; keep the Osiris kernel recipe.
   Notebook outputs are stripped on copy.
2. `tools/automl_parity/` — **not migrated**, see D9.
3. `data/make_data/make_synth.ipynb` is **not** copied: it builds a one-row
   multimodal *sequence* parquet for `MultimodalParquetDataset`, not an AutoML
   tabular dataset. The pipeline notebooks read cluster paths
   (`/shared/data/...`) and are not executable here; the V2 synthetic data is
   generated by a script reusing the `_write_splits` helpers already present in
   the task tests.
4. `README.md` + `MAINTENANCE.md`: one section describing `avatar.automl`.

### Stage E — verification (§4), then plan status -> done.

## 4. Verification

Results as of 2026-09-16, python 3.12.12 venv, CPU only.

- **V1 — unit suite. PASS.** `pytest` on the whole repo: **740 passed**, 66 slow
  deselected — 165 avatar (119 + 46 Spark) + 575 automl. The same 575 automl
  cases run against the *source* repo with the same interpreter: 575 passed.
  No new skips. `ruff check .` clean.
- **V2 — local end-to-end parity. PASS, byte-identical.** Six configurations
  over synthetic parquet (4 000 / 1 500 / 1 500 rows, 2 categorical + 3
  numerical + one `List(Float32)` hidden-state column, a group role, a date
  role and a treatment role):

  | case | task | engine | hyperopt | model_layout |
  |---|---|---|---|---|
  | `binary_catboost_both` | binary | catboost | 2 trials | global_and_per_group |
  | `binary_xgboost_per_group` | binary | xgboost | off | per_group |
  | `response_catboost_global` | response | catboost | off | global (+treatment) |
  | `regression_xgboost_hyperopt` | regression | xgboost | 2 trials | global |
  | `multiclass_catboost_global` | multiclass | catboost | off | global |
  | `uplift_catboost_global` | uplift | catboost | off | global (+propensity) |

  Each case runs `train -> save -> load -> predict -> evaluate` and records
  `best_params`, `validation_metrics`, `feature_names`, `class_order`, the
  evaluation metrics, and a SHA-256 digest of every numeric score column. The
  JSON from `avatar.automl` and from `fmlib.automl` (source repo, same
  interpreter, same data) compare **equal byte for byte**.
- **V3 — Osiris. NOT RUN** (no scheduler here). One `env_type="osiris"` binary
  run on the cluster, checked against a pre-migration run id, plus loading one
  pre-migration artifact, still has to happen before anyone depends on the
  remote path.
- **V4 — packaging. PASS.** `python -m build --wheel` produces
  `avatar-0.1.0-py3-none-any.whl` with 55 `avatar/automl/*` modules and zero
  test files.

## 5. Risks

| risk | mitigation |
|---|---|
| `numpy<2` in avatar vs `numpy==1.26.4` in fmlib | same major; pin stays `<2`, V1 catches the rest |
| catboost GPU unavailable on this host (driver too old) | V2 runs `device="cpu"`; GPU path is V3 |
| reformatting 9 795 lines hides a real change | D4 is a separate commit on top of a green suite; V2 then reproduced byte-identical results |
| trained artifacts on the cluster stop loading | D5 freezes every persisted string; V3 loads a pre-migration artifact |
| `polars` becomes a hard dep of a torch library | accepted — automl's public result types are `pl.DataFrame` |

## 6. Follow-ups (not in this migration)

- **F1** — stage 3: neural-network backend (`backend="tabnn"`), separate design
  doc. The seams already exist: `BaseTaskConfig` validates `backend in
  {"boosting","tabnn"}`, and `config/base.py` currently rejects
  `tabnn+hyperopt` and `tabnn+cpu`. The work is generalizing
  `BaseBoostingTask`/`SupervisedBoostingTask`/`fit_boosting_model` to a
  backend-agnostic protocol.
- **F2** — reconcile `avatar.automl.metrics.uplift` with `avatar.metrics.uplift`
  and `avatar.automl.reporting` with `avatar.metrics.campaign`.
- **F3** — drop the `avatar/nn/tabular/ste` shim once D7 is resolved.
- **F5** — the internal scratch column names (`__fmlib_remote_row_id`,
  `__fmlib_calibration_row_id`, `__fmlib_key_occurrence`, `__fmlib_truth_row`)
  and the Osiris job-name prefix `fmlib-<action>-<run id>` are kept verbatim.
  They are transient, not persisted, so renaming them is free — but pointless
  until the avatar -> fmlib rename decides the final name.
- **F6** — `examples/automl/configs/fmlib_*.yaml` keep their names: the prefix
  used to contrast with the dropped `autocampaignxfm_*.yaml` and is consistent
  with the eventual rename. Two migrated tests reference the paths literally.
- **F4** — `EnvironmentConfig` defaults still point at the shared fmlib
  checkout (`/home/datalab/nfs/sber-amazme-fmlib/env`) and the gigachat image;
  revisit when avatar is renamed to fmlib.
