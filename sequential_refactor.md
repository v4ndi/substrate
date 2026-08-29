# Design: merge `avatar/nn/sequence` + `avatar/nn/feature_encoder` → `avatar/nn/sequential`

Status: proposal (2026-08-29). Owner: refactor track, follows the
`avatar/nn/embedding` split precedent (commit `719fe71`).

---

## 1. Why

`avatar/nn/feature_encoder` (raw event features → a sequence of event vectors) and
`avatar/nn/sequence` (that sequence → a contextualised sequence → model output)
are two halves of one pipeline. The dependency is already one-directional:

```
avatar.nn.sequence  ──imports──▶  avatar.nn.feature_encoder  ──imports──▶  avatar.nn.embedding
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
              avatar.data.event_seq_batch (EventSequenceBatch)
```

`feature_encoder` has **zero** dependency on `sequence`. So the two packages are
really "layer 1" and "layer 2" of the same stack, split across two top-level
names for historical reasons. Merging them:

- puts the whole event-sequence model stack under one import root
  (`avatar.nn.sequential`), symmetric with `avatar.nn.tabular`;
- lets the two layers share a `base.py` / test tree / docs;
- kills the "what's the difference between `sequence` and `feature_encoder`?"
  question (the word "sequence" currently means 3 different things — see §6).

## 2. Current inventory

### `avatar/nn/feature_encoder/` (426 LOC)

| symbol | file | role | public? |
|---|---|---|---|
| `BaseSequenceFeatureEncoder` | `base_feature_encoder.py` | `nn.Module`; embedding + optional pos-embedding + time encoding → `(B,S,H)` or `(B,S,F,H)` | yes (configs, pipeline, tests) |
| `IntraFeatureAttention` | `attention_encoder.py` | scaled dot-product attention across the feature axis within an event | tests only |
| `AggFFNBlock` | `attention_encoder.py` | attention + FFN + masked pool (`attention`/`mean`/`linear`) → `(B,S,H)` | no |
| `get_event_attention_mask` | `attention_encoder.py` | free fn: per-feature attn mask from `columns_meta` | no |
| `FeatureEncoder` | `attention_encoder.py` | `BaseSequenceFeatureEncoder` + feature-attention aggregation + optional id-embedding | yes |
| `FeatureAttentionEncoder` | `attention_encoder.py` | **deprecated** alias of `FeatureEncoder` — still the `_target_` in 100% of configs | yes (configs) |

### `avatar/nn/sequence/` (~220 LOC)

| symbol | file | role | public? |
|---|---|---|---|
| `BaseSequenceBackbone` | `base_sequence_model.py` | abstract `nn.Module`; `(inputs_embeds, attention_mask) → seq` | yes (tests) |
| `BaseSequenceModel` | `base_sequence_model.py` | composes `.feature_encoder` + `.backbone`; abstract `forward` | yes (pipeline, tests) |
| `TransformersWrapper` | `transformers_wrapper.py` | concrete model: encoder → HF backbone → `BaseSequenceOutput` | yes (configs) |
| `ModelPruningWrapper` | `pretrained_model.py` | `BaseSequenceBackbone`; HF model pruning | yes (`__init__` only) |

### External consumers (blast radius)

Python imports (8 sites):

```
avatar/nn/sequence/base_sequence_model.py   from avatar.nn.feature_encoder.base_feature_encoder import BaseSequenceFeatureEncoder
avatar/nn/sequence/transformers_wrapper.py  from avatar.nn.feature_encoder import BaseSequenceFeatureEncoder
avatar/pipeline/sequence/agg_hidden_states.py    from avatar.nn.sequence import BaseSequenceModel
avatar/pipeline/sequence/classification.py       from avatar.nn.sequence import BaseSequenceModel
avatar/pipeline/sequence/next_k_tokens.py        from avatar.nn.sequence import BaseSequenceModel
avatar/nn/__init__.py                        from . import embedding, sequence, tabular, utils
tests/pipeline/test_next_k.py               from avatar.nn.sequence import BaseSequenceBackbone, BaseSequenceModel
                                            from avatar.nn.feature_encoder import BaseSequenceFeatureEncoder
tests/nn/feature_encoder/test_attention_encoder.py   from avatar.nn.feature_encoder.attention_encoder import IntraFeatureAttention
```

Hydra `_target_` in configs (`examples/`, `experiments/`, `docs/` — 4 files each):

```
_target_: avatar.nn.sequence.TransformersWrapper
_target_: avatar.nn.feature_encoder.FeatureAttentionEncoder
```

Hydra interpolations that depend on the **attribute** name `feature_encoder`
(not the import path):

```
${model.model.feature_encoder.embedding.hidden_size}          # examples ×3, docs ×1
${model.sequence_model.feature_encoder.embedding.hidden_size} # experiments ×1
```

Pipeline code that reaches into the attribute:

```
avatar/pipeline/sequence/next_k_tokens.py:  model.feature_encoder.embedding.{columns_meta, hidden_size}
```

## 3. Bugs / smells found in review (fix opportunistically during the move)

Ordered by severity.

1. **`avatar.nn` is import-broken right now.** `pretrained_model.py` is deleted in
   the working tree (uncommitted) but `sequence/__init__.py` still does
   `from .pretrained_model import ModelPruningWrapper`, and `avatar/nn/__init__.py`
   imports `sequence`. `python -c "import avatar.nn"` → `ModuleNotFoundError`.
   The refactor must land with a decision: **restore** `ModelPruningWrapper`
   (into `backbone/pruning.py`) or **drop** it. It has no in-repo consumers other
   than the `__init__` export; nothing instantiates it. Recommendation: keep it,
   move to `backbone/pruning.py` — it's the only non-HF-native backbone we have.

2. **`BaseSequenceFeatureEncoder._set_time_encoding` — "no time encoding" is
   unreachable.** `time_encoding=None` is silently rewritten to `"delta"` (bare
   `print()`, line 49), so the `if self.time_encoding is not None:` guard on
   line 53 is always true and the `LinearEmbeddings` time layer is *always*
   built. The docstring advertises `None` as valid. Fix: honour `None`; use a
   logger, not `print`; fix the `"supportds"` typo.

3. **`AggFFNBlock.forward` crashes when `event_attention_mask is None`.**
   `_get_original_mask(None) → None`, then `output * feature_mask.unsqueeze(-1)`
   → `AttributeError`. So `FeatureEncoder` silently requires `seq_features.event_ids`
   to be set. Either make the mask optional end-to-end or assert early with a
   clear message.

4. **`FeatureEncoder` hard-codes domain column names.** `seq_features["epk_id"]`
   (line 308) for the id-embedding path. Make it a constructor arg
   (`id_column: str = "epk_id"`).

5. **`BaseSequenceModel` type contract is fiction.** `backbone: BaseSequenceBackbone`
   but every real config passes a bare `transformers.GPT2Model`. The true
   contract is duck-typed: `backbone(inputs_embeds=, attention_mask=,
   output_hidden_states=) -> obj with .last_hidden_state [, .hidden_states,
   .router_logits]`. Introduce a `typing.Protocol` (`SequenceBackbone`) or at
   least document it. `BaseSequenceBackbone` as an ABC buys us nothing today.

6. **`AggFFNBlock._get_original_mask` positional coupling.** Relies on "the last
   feature row is the time token" — a convention set by
   `BaseSequenceFeatureEncoder`'s "concatenate time on the right" step. Marked
   with a `# TODO`. Document the invariant in one place or pass the time-token
   index explicitly.

7. **`aggregation_mode="linear"` ignores the feature mask** (mean/attention
   apply it). Inconsistent; padded features leak into the linear pool.

8. **`FeatureAttentionEncoder` deprecation fires on every run** — it's the
   `_target_` in all configs. Migrate configs to `FeatureEncoder` (or a new
   name, §6) as part of this change so the warning means something again.

9. **`IntraFeatureAttention`**: method name typo `calcuate_attn_scores`; no
   attention dropout; fully-masked feature rows → `softmax(-inf)` → NaN
   (mitigated only because `get_event_attention_mask` ORs in the diagonal).

10. **`transformers_wrapper.py`**: dead `import torch  # noqa: F401`. Meta-token
    stripping (`attention_mask.shape[:2] != inputs_embeds.shape[:2]`) is implicit
    cross-module coupling with `FeatureEncoder.id_embedding` prepending a token —
    communicated only through a shape comparison. Add a comment linking the two,
    or return an explicit `n_prefix_tokens` from the encoder.

11. **Stale docstrings**: `base_feature_encoder.py` numpydoc mentions a
    `TemporalPositionEncoding` param (actual: `pos_embedding: BaseTemporalEmbedding`)
    and leaves a bare `...`; `FeatureEncoder` example calls `FeatureAttentionEncoder`.

12. **`BaseSequenceBackbone.forward` LSP**: `ModelPruningWrapper.forward` adds a
    required positional `output_hidden_states` the base doesn't declare.

## 4. Proposed layout

Two sub-packages as you suggested, plus `model.py` at the package root as the
composition layer (it's ~140 LOC and doesn't belong in either sub-package —
it *combines* them).

```
avatar/nn/sequential/
├── __init__.py                 # flat re-export of the full public API
├── model.py                    # BaseSequenceModel, TransformersWrapper
├── event_encoder/              # ex feature_encoder — raw features → (B,S,H)
│   ├── __init__.py
│   ├── base.py                 # BaseEventEncoder            (ex BaseSequenceFeatureEncoder)
│   ├── attention.py            # IntraFeatureAttention, AggFFNBlock, build_event_attention_mask
│   └── feature_attention.py    # AttentionEventEncoder       (ex FeatureEncoder)
└── backbone/                   # ex sequence — (B,S,H) → contextualised (B,S,H)
    ├── __init__.py
    ├── base.py                 # BaseBackbone / SequenceBackbone Protocol   (ex BaseSequenceBackbone)
    └── pruning.py              # PrunedHFBackbone            (ex ModelPruningWrapper)
```

`__init__.py` re-exports everything so `avatar.nn.sequential.<Name>` is flat,
exactly like `avatar/nn/embedding/__init__.py`:

```python
from avatar.nn.sequential.event_encoder import (
    AttentionEventEncoder, BaseEventEncoder, AggFFNBlock, IntraFeatureAttention,
)
from avatar.nn.sequential.backbone import BaseBackbone, PrunedHFBackbone
from avatar.nn.sequential.model import BaseSequenceModel, TransformersWrapper
```

Import order inside `__init__.py`: `event_encoder` → `backbone` → `model`
(model imports both), same discipline as the embedding split.

### Alternative considered: 3 sub-packages (`event_encoder/`, `backbone/`, `model/`)

Rejected — `model/` would hold 2 files and adds a level for no grouping benefit.
`model.py` at root is the natural "assembly" spot.

### Alternative considered: keep `model` logic in `backbone/`

Rejected — `BaseSequenceModel` is not a backbone; conflating them is the current
mistake.

## 5. Backward compatibility

Two viable strategies:

### Strategy A — hard rename, update every in-repo reference (recommended)

- Move code, rewrite the 8 Python imports and 8 config lines to
  `avatar.nn.sequential.*`.
- Update `avatar/nn/__init__.py`: `sequence, feature_encoder` → `sequential`.
- Leave **thin deprecation shims** for one release, matching how
  `FeatureAttentionEncoder` is handled today:

  ```python
  # avatar/nn/sequence/__init__.py
  import warnings
  from avatar.nn.sequential import *          # noqa: F401,F403
  from avatar.nn.sequential import BaseSequenceModel, TransformersWrapper  # ...
  warnings.warn("avatar.nn.sequence moved to avatar.nn.sequential", DeprecationWarning, stacklevel=2)
  ```

  Same for `avatar/nn/feature_encoder/__init__.py`. This keeps **out-of-repo
  experiment configs** on remote clusters working (they reference
  `avatar.nn.sequence.TransformersWrapper`).
- Delete the shims in a follow-up commit once external configs are migrated.

### Strategy B — keep both old names as permanent re-export packages

Cheaper (no config churn) but leaves 3 names (`sequence`, `feature_encoder`,
`sequential`) forever. The embedding refactor kept the *package* name precisely
to avoid touching ~60 configs; here there are only **8** config lines and you've
explicitly asked to unify the name, so A is worth it.

**Recommendation: Strategy A with shims kept for one release.**

### The `feature_encoder` attribute name

`BaseSequenceModel.feature_encoder` is a config-facing contract
(`${...feature_encoder.embedding.hidden_size}` in 5 configs + pipeline code).
Options:

1. Keep the attribute `self.feature_encoder` even though the class is renamed
   `BaseEventEncoder`. Zero interpolation churn. Slight name mismatch.
2. Rename to `self.event_encoder`, update 5 configs + `next_k_tokens.py`, add a
   read-only `feature_encoder` property alias for out-of-repo configs.

**Recommendation: option 2** — do the rename properly while we're here, keep the
property alias next to the shims, drop both together later.

## 6. Naming proposals (optional, but this is the moment)

The word **"sequence"** currently tags: the batch (`EventSequenceBatch`), the
encoder (`BaseSequenceFeatureEncoder`), the backbone (`BaseSequenceBackbone`),
and the model (`BaseSequenceModel`). Inside a package literally called
`sequential`, most of those qualifiers are noise.

| current | proposed | rationale |
|---|---|---|
| `BaseSequenceFeatureEncoder` | `BaseEventEncoder` | encodes events; "feature" collides with tabular "features" |
| `FeatureEncoder` | `AttentionEventEncoder` | it's specifically the intra-feature-attention aggregator; `FeatureEncoder` is too generic and clashes with the package concept |
| `FeatureAttentionEncoder` | *(drop / alias 1 release)* | already deprecated |
| `BaseSequenceBackbone` | `BaseBackbone` (+ `SequenceBackbone` Protocol) | "Sequence" redundant under `sequential.backbone` |
| `ModelPruningWrapper` | `PrunedHFBackbone` | it *is* a backbone, not a wrapper around a model |
| `BaseSequenceModel` | *(keep)* | most-imported name; churn not worth it |
| `TransformersWrapper` | *(keep)* — maybe `HFSequenceModel` | it's a `_target_`; rename only with a shim |
| `get_event_attention_mask` | `build_event_attention_mask` | verb; it constructs |
| `AggFFNBlock` | *(keep)* | internal |

All renames ship with re-export aliases in the shim `__init__`s for one release.

## 7. Migration plan (execution order)

1. **Unblock `import avatar.nn`** first (standalone commit): decide
   `ModelPruningWrapper` in/out; make `sequence/__init__.py` consistent with the
   working tree. (This is independent of the refactor and should not wait for it.)
2. Create `avatar/nn/sequential/` with the §4 layout, `git mv` files where
   possible to preserve blame:
   - `feature_encoder/base_feature_encoder.py` → `sequential/event_encoder/base.py`
   - `feature_encoder/attention_encoder.py` → split into
     `sequential/event_encoder/attention.py` (+`feature_attention.py`)
   - `sequence/base_sequence_model.py` → split: `BaseSequenceBackbone` →
     `backbone/base.py`, `BaseSequenceModel` → `model.py`
   - `sequence/transformers_wrapper.py` → `model.py` (merge)
   - `sequence/pretrained_model.py` → `backbone/pruning.py`
3. Rewrite internal imports to `avatar.nn.sequential.*`; apply renames from §6.
4. Fix the §3 bugs that are cheap and local (2, 4, 10, 11 at minimum).
5. Update `avatar/nn/__init__.py`, the 8 external Python imports, the 8 config
   `_target_` lines, and (if §5 option 2) the 5 interpolations +
   `next_k_tokens.py`.
6. Add deprecation shims `avatar/nn/sequence/__init__.py` +
   `avatar/nn/feature_encoder/__init__.py`.
7. Move tests: `tests/nn/feature_encoder/` → `tests/nn/sequential/event_encoder/`;
   add a `tests/nn/sequential/test_imports.py` asserting both the new flat API
   and the deprecation shims.
8. `ruff check . && ruff format --check .`; full suite `pytest -m ""`
   (baseline: 210 passed). No numerical change expected — this is a move +
   rename + dead-code fix, no math touched.
9. Follow-up commit (separate PR): delete shims + `feature_encoder` property
   alias once external experiment configs are migrated.

## 8. Risks

- **Out-of-repo configs** on experiment clusters pin `avatar.nn.sequence.*` /
  `avatar.nn.feature_encoder.*` / the `feature_encoder` attribute. Shims + the
  property alias cover the import paths and the attribute; a grep of the configs
  in `experiments/` is the in-repo proxy but not exhaustive.
- **Splitting `attention_encoder.py`** touches the most complex file (329 LOC).
  Keep it a pure move in step 2, do renames in step 3, so a `git diff -M` stays
  readable.
- **`__init__.py` circular imports** — mitigated by the fixed import order
  (event_encoder → backbone → model) proven in the embedding split.
- Bug fixes in §3 items 3, 5, 7 change behaviour on edge paths (no-mask,
  non-`BaseSequenceBackbone` backbone, fully-masked feature). Gate them behind
  their own commits with tests rather than folding into the move.
