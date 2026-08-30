# Design: merge `avatar/nn/sequence` + `avatar/nn/feature_encoder` → `avatar/nn/sequential`

Status: **accepted** (2026-08-29). Follows the `avatar/nn/embedding` split
precedent (commit `719fe71`).

Decisions locked in this revision:

- **3 sub-packages** inside `sequential`: `model/`, `event_encoder/`, `backbone/`.
- **Strategy A** — hard rename, update every in-repo reference (+ one-release
  deprecation shims for out-of-repo configs).
- **`ModelPruningWrapper` is deleted**, not migrated (no longer needed).
- `FeatureEncoder` → **`EventEncoder`** (not `AttentionEventEncoder` — the
  aggregator has `mean`/`attention`/`linear` modes, so "Attention" in the name
  would mislead).
- `AggFFNBlock` → **`EventAggregator`**.

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
- lets the layers share a test tree / docs / base modules;
- kills the "what's the difference between `sequence` and `feature_encoder`?"
  question (the word "sequence" currently means 4 different things — see §6).

## 2. Current inventory

### `avatar/nn/feature_encoder/` (426 LOC)

| symbol | file | role | public? | fate |
|---|---|---|---|---|
| `BaseSequenceFeatureEncoder` | `base_feature_encoder.py` | `nn.Module`; embedding + optional pos-embedding + time encoding → `(B,S,H)` or `(B,S,F,H)` | yes (configs, pipeline, tests) | → `event_encoder/base.py` as `BaseEventEncoder` |
| `IntraFeatureAttention` | `attention_encoder.py` | scaled dot-product attention across the feature axis within an event | tests only | → `event_encoder/attention.py` (keep name) |
| `AggFFNBlock` | `attention_encoder.py` | attention + FFN + masked pool (`attention`/`mean`/`linear`) → `(B,S,H)` | no | → `event_encoder/attention.py` as `EventAggregator` |
| `get_event_attention_mask` | `attention_encoder.py` | free fn: per-feature attn mask from `columns_meta` | no | → `event_encoder/attention.py` as `build_event_attention_mask` |
| `FeatureEncoder` | `attention_encoder.py` | `BaseSequenceFeatureEncoder` + feature-attention aggregation + optional id-embedding | yes (configs) | → `event_encoder/event.py` as `EventEncoder` |
| `FeatureAttentionEncoder` | `attention_encoder.py` | **deprecated** alias of `FeatureEncoder` — still the `_target_` in 100% of configs | yes (configs) | dropped; alias in shim for 1 release |

### `avatar/nn/sequence/` (~220 LOC)

| symbol | file | role | public? | fate |
|---|---|---|---|---|
| `BaseSequenceBackbone` | `base_sequence_model.py` | abstract `nn.Module`; `(inputs_embeds, attention_mask) → seq` | yes (tests) | → `backbone/base.py` as `BaseBackbone` (+ `SequenceBackbone` Protocol) |
| `BaseSequenceModel` | `base_sequence_model.py` | composes `.feature_encoder` + `.backbone`; abstract `forward` | yes (pipeline, tests) | → `model/base.py` (keep name) |
| `TransformersWrapper` | `transformers_wrapper.py` | concrete model: encoder → HF backbone → `BaseSequenceOutput` | yes (configs) | → `model/transformers.py` (keep name) |
| `ModelPruningWrapper` | `pretrained_model.py` | `BaseSequenceBackbone`; HF model pruning | yes (`__init__` only) | **DELETED** — file already removed in the working tree; drop the `__init__` export |

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
   **Fix (step 1, standalone commit): drop `ModelPruningWrapper` from
   `sequence/__init__.py` + `__all__`.** `ModelPruningWrapper` is gone for good —
   nothing in the repo instantiates it and it is no longer needed.

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
   .router_logits]`. Introduce a `typing.Protocol` (`SequenceBackbone`) in
   `backbone/base.py` and type `BaseSequenceModel.backbone` against it. Keep
   `BaseBackbone` as an optional `nn.Module` base for our own backbones.

6. **`AggFFNBlock._get_original_mask` positional coupling.** Relies on "the last
   feature row is the time token" — a convention set by
   `BaseSequenceFeatureEncoder`'s "concatenate time on the right" step. Marked
   with a `# TODO`. Document the invariant in one place or pass the time-token
   index explicitly.

7. **`aggregation_mode="linear"` ignores the feature mask** (mean/attention
   apply it). Inconsistent; padded features leak into the linear pool.

8. **`FeatureAttentionEncoder` deprecation fires on every run** — it's the
   `_target_` in all configs. Migrating configs to `EventEncoder` is part of
   this change (§7), so the warning starts meaning something again.

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

## 4. Proposed layout

Three sub-packages: `event_encoder/` (raw features → `(B,S,H)`), `backbone/`
(`(B,S,H)` → contextualised `(B,S,H)`), `model/` (assembles the two →
`BaseSequenceOutput`).

```
avatar/nn/sequential/
├── __init__.py                 # flat re-export of the full public API
├── event_encoder/              # ex feature_encoder
│   ├── __init__.py
│   ├── base.py                 # BaseEventEncoder            (ex BaseSequenceFeatureEncoder)
│   ├── attention.py            # IntraFeatureAttention, EventAggregator (ex AggFFNBlock),
│   │                           #   build_event_attention_mask (ex get_event_attention_mask)
│   └── event.py                # EventEncoder                (ex FeatureEncoder)
├── backbone/                   # ex sequence (backbone half)
│   ├── __init__.py
│   └── base.py                 # BaseBackbone (nn.Module) + SequenceBackbone (Protocol)
│                               #   ex BaseSequenceBackbone
└── model/                      # ex sequence (model half)
    ├── __init__.py
    ├── base.py                 # BaseSequenceModel          (keep name)
    └── transformers.py         # TransformersWrapper         (keep name)
```

`backbone/` has a single file today (`ModelPruningWrapper` is deleted). It stays
a package rather than a `backbone.py` module for symmetry with the other two and
so concrete backbones can be added later without another move.

`sequential/__init__.py` re-exports everything so `avatar.nn.sequential.<Name>`
is flat, exactly like `avatar/nn/embedding/__init__.py`:

```python
from avatar.nn.sequential.event_encoder import (
    BaseEventEncoder, EventEncoder, EventAggregator, IntraFeatureAttention,
    build_event_attention_mask,
)
from avatar.nn.sequential.backbone import BaseBackbone, SequenceBackbone
from avatar.nn.sequential.model import BaseSequenceModel, TransformersWrapper
```

Import order inside `__init__.py`: `event_encoder` → `backbone` → `model`
(`model` imports both), same discipline as the embedding split.

## 5. Backward compatibility — Strategy A (accepted)

- Move code, rewrite the 8 Python imports and 8 config `_target_` lines to
  `avatar.nn.sequential.*`; apply the §6 renames.
- Update `avatar/nn/__init__.py`: `embedding, sequence, tabular, utils` →
  `embedding, sequential, tabular, utils` (drop `sequence`; `feature_encoder`
  was never listed there).
- Leave **thin deprecation shims** for one release, matching how
  `FeatureAttentionEncoder` is handled today:

  ```python
  # avatar/nn/sequence/__init__.py
  import warnings
  from avatar.nn.sequential import (  # noqa: F401
      BaseSequenceModel, TransformersWrapper, BaseBackbone as BaseSequenceBackbone,
  )
  warnings.warn(
      "avatar.nn.sequence moved to avatar.nn.sequential",
      DeprecationWarning, stacklevel=2,
  )
  ```

  ```python
  # avatar/nn/feature_encoder/__init__.py
  import warnings
  from avatar.nn.sequential import (  # noqa: F401
      BaseEventEncoder as BaseSequenceFeatureEncoder,
      EventEncoder as FeatureEncoder,
      EventEncoder as FeatureAttentionEncoder,
  )
  warnings.warn(
      "avatar.nn.feature_encoder moved to avatar.nn.sequential",
      DeprecationWarning, stacklevel=2,
  )
  ```

  Purpose: keep **out-of-repo experiment configs** on remote clusters working
  (they reference `avatar.nn.sequence.TransformersWrapper` /
  `avatar.nn.feature_encoder.FeatureAttentionEncoder`).
- Delete the shims in a follow-up commit once external configs are migrated.

### The `feature_encoder` attribute name

`BaseSequenceModel.feature_encoder` is a config-facing contract
(`${...feature_encoder.embedding.hidden_size}` in 5 configs + pipeline code).
Per "update every in-repo reference": rename the attribute to
`self.event_encoder`, update the 5 interpolations + `next_k_tokens.py`, and add a
read-only `feature_encoder` property alias on `BaseSequenceModel` for out-of-repo
configs. The alias is dropped together with the shims.

```python
@property
def feature_encoder(self):  # deprecated alias, drop with the shims
    return self.event_encoder
```

## 6. Renames

The word **"sequence"** currently tags: the batch (`EventSequenceBatch`), the
encoder (`BaseSequenceFeatureEncoder`), the backbone (`BaseSequenceBackbone`),
and the model (`BaseSequenceModel`). Inside a package literally called
`sequential`, most of those qualifiers are noise.

| current | new | rationale |
|---|---|---|
| `BaseSequenceFeatureEncoder` | `BaseEventEncoder` | encodes events; "feature" collides with tabular "features" |
| `FeatureEncoder` | `EventEncoder` | generic event encoder; aggregation mode (`mean`/`attention`/`linear`) is a param, so no "Attention" in the name |
| `FeatureAttentionEncoder` | *(dropped; alias 1 release)* | already deprecated |
| `AggFFNBlock` | `EventAggregator` | aggregates a event's features into one vector |
| `get_event_attention_mask` | `build_event_attention_mask` | verb; it constructs |
| `BaseSequenceBackbone` | `BaseBackbone` (+ `SequenceBackbone` Protocol) | "Sequence" redundant under `sequential.backbone` |
| `ModelPruningWrapper` | *(deleted)* | not needed |
| `BaseSequenceModel` | *(keep)* | most-imported name; churn not worth it |
| `TransformersWrapper` | *(keep)* | it's a `_target_`; rename only with a shim |
| `IntraFeatureAttention` | *(keep)* | accurate |
| `BaseSequenceModel.feature_encoder` (attr) | `.event_encoder` (+ property alias) | see §5 |

Every rename ships with a re-export alias in the shim `__init__`s for one
release.

## 7. Migration plan (execution order)

1. **Unblock `import avatar.nn`** (standalone commit, independent of the rest):
   remove `ModelPruningWrapper` from `avatar/nn/sequence/__init__.py` +
   `__all__`; commit the already-staged `pretrained_model.py` deletion.
   Run `pytest -m ""` to confirm green.
2. Create `avatar/nn/sequential/` with the §4 layout. `git mv` to preserve blame:
   - `feature_encoder/base_feature_encoder.py` → `sequential/event_encoder/base.py`
   - `feature_encoder/attention_encoder.py` → `sequential/event_encoder/attention.py`,
     then split `FeatureEncoder` out into `sequential/event_encoder/event.py`
   - `sequence/base_sequence_model.py` → split: backbone half →
     `backbone/base.py`, model half → `model/base.py`
   - `sequence/transformers_wrapper.py` → `model/transformers.py`
   Keep step 2 a **pure move** (no renames) so `git diff -M` stays readable.
3. Apply the §6 renames; rewrite internal imports to `avatar.nn.sequential.*`;
   add the `SequenceBackbone` Protocol.
4. Fix the cheap local §3 bugs (2, 4, 10, 11 at minimum).
5. Update `avatar/nn/__init__.py`, the 8 external Python imports, the 8 config
   `_target_` lines, the 5 interpolations, and `next_k_tokens.py`
   (`feature_encoder` → `event_encoder`).
6. Add deprecation shims `avatar/nn/sequence/__init__.py` +
   `avatar/nn/feature_encoder/__init__.py`; add the `feature_encoder` property
   alias on `BaseSequenceModel`.
7. Move tests: `tests/nn/feature_encoder/` → `tests/nn/sequential/event_encoder/`;
   add `tests/nn/sequential/test_imports.py` asserting the new flat API **and**
   the deprecation shims (that they emit `DeprecationWarning` and resolve).
8. `ruff check . && ruff format --check .`; full suite `pytest -m ""`
   (baseline: 210 passed). Steps 2–3 + 5–7 are move + rename only — no
   numerical change expected.
9. Follow-up commit (separate PR): delete shims + `feature_encoder` property
   alias once external experiment configs are migrated.

Suggested commit breakdown: step 1 / step 2 / steps 3+5+6 / step 4 (one commit
per bug with its test) / step 7.

## 8. Risks

- **Out-of-repo configs** on experiment clusters pin `avatar.nn.sequence.*` /
  `avatar.nn.feature_encoder.*` / the `feature_encoder` attribute. Shims + the
  property alias cover the import paths and the attribute; the configs under
  `experiments/` are the in-repo proxy but not exhaustive.
- **Splitting `attention_encoder.py`** touches the most complex file (329 LOC).
  Pure move in step 2, renames in step 3 — keeps the diff reviewable.
- **`__init__.py` circular imports** — mitigated by the fixed import order
  (event_encoder → backbone → model), proven in the embedding split.
- §3 bug fixes 3, 5, 7 change behaviour on edge paths (no-mask, non-`BaseBackbone`
  backbone, `linear` mode). Gate them behind their own commits with tests rather
  than folding into the move.
