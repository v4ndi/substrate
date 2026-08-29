# Design: restructure `avatar/nn/tabular`

Status: **draft / proposal** (2026-08-29). Follows the `avatar/nn/embedding`
(commit `719fe71`) and `avatar/nn/sequential` (`3759c1a..734f2aa`) precedents.

Goal (from the ask): `avatar/nn/tabular` holds **reusable tabular neural nets**;
`avatar/pipeline/*` wraps them into task settings — classification, regression,
uplift, multi-label, multi-task. This doc answers: what lives in `nn/tabular` vs
`pipeline`, which base classes are needed, and what the I/O contract is.

---

## 0. Done in this pass — deletions

`avatar/nn/tabular/moe/` and `avatar/nn/tabular/uplift/` are **removed**
(unused: zero configs, zero tests, no consumers outside the package). Along with
them:

| removed | file | reason |
|---|---|---|
| `UniversalGate`, `GateTopK`, `MLPGate`, `MLPExperts`, `ExpertsWrapper`, `MoE`, `FFN` | `moe/moe_modeling.py` | MoE experiments abandoned; `avatar/pipeline/multi_task/mmoe.py` has its own expert/gate impl and never imported these |
| `MoEEncoderBlock`, `MoESTEv2` | `ste/ste_modeling.py` | only consumers of `moe/`; `MoESTEv2` was a broken subclass (`del self.encoder_blocks`) with no config/test |
| `FeedForwardHead`, `TreatmentCrossAttnEncoder`, `MultiTreatmentSTE` | `uplift/mt_modeling.py` | `MultiTreatmentSTE` is a task model (heads + sigmoid + tuple return), not an nn block; no config/test |
| `MoeTabularOutput` | `avatar/outputs.py` | only referenced by the deleted `MoE` (`moe_reg.py`'s `Entropy`/`Importance` read `MMoEOutput.task_gated_weights`, unaffected) |

`CrossAttentionEncoderBlock` (in `ste/ste_modeling.py`) is now unused too — its
only caller was `TreatmentCrossAttnEncoder`. Kept for now as a generic layer
(§3.2); flag if it should go.

Verification: `import avatar.nn.tabular` + all pipeline/metrics importers OK;
ruff clean; `155 passed` (not-slow), full suite unchanged.

---

## 1. What remains

```
avatar/nn/tabular/
├── __init__.py            # flat re-export
├── base_tabular.py        # BaseTabularBackbone, BaseTabularEncoder
├── dcnv2/dcn_v2.py        # CrossNetV2, DCNv2
└── ste/ste_modeling.py    # SublayerConnection, EncoderBlock,
                           #   CrossAttentionEncoderBlock, STEv2Block, STEv2
```

| symbol | role | used by |
|---|---|---|
| `BaseTabularBackbone` | `nn.Module` + `embedding`; `forward(TabularBatch)` | `STEv2`, `TabularWithAggregatedStates` |
| `BaseTabularEncoder` | `nn.Module`; `forward(input_embeds: Tensor)` | `STEv2Block`, pipeline type hints |
| `SublayerConnection` | Add&Norm residual wrapper | ste internals |
| `EncoderBlock` | 1 transformer encoder layer (MHA + FFN) | `STEv2Block` |
| `CrossAttentionEncoderBlock` | cross-attention variant | **now unused** |
| `STEv2Block` | stack of `EncoderBlock`; `[B,F,D] → [B,F,D]` tensor | ~40 configs, `s_learner`, `supervised`, `mmoe` |
| `STEv2` | `embedding` + `STEv2Block` → `BaseTabularOutput` | `examples/tabular_hidden_states/*` (5 configs) |
| `DCNv2`, `CrossNetV2` | `[B, d] → [B, d']` cross-feature interaction on the **pooled** vector | **no configs, no tests** |

---

## 2. Review — what's still wrong

### 2.1 The base classes don't define one contract

Two bases with overlapping, under-specified jobs:

- `BaseTabularBackbone(embedding)` — owns the embedding, `forward(TabularBatch)`.
- `BaseTabularEncoder()` — no embedding, `forward(input_embeds: Tensor)`.

and the return type is inconsistent:

| module | base | returns |
|---|---|---|
| `STEv2Block` | `BaseTabularEncoder` | bare `Tensor` `[B,F,D]` |
| `STEv2` | `BaseTabularBackbone` | `BaseTabularOutput` |
| `STEv2` (passed a raw tensor) | — | `BaseTabularOutput`, skips embedding — dual contract in one method |
| `DCNv2` | `nn.Module` | bare `Tensor` `[B, d']` (different rank — pooled, not per-feature) |

Consumers each assume something different, and it only works because
`avatar/nn/utils/agg.py::apply_expanded_mask` silently accepts both tensor and
output object (`if hasattr(states, "last_hidden_state")`). The polymorphic
aggregation layer papers over an undefined encoder contract.

### 2.2 `embedding` inside vs. outside the module — two composition styles

- **Old** (`STEv2` + `TabularWithAggregatedStates`): encoder owns the embedding.
- **New** (`SLearner`, `SupervisedLearner`, `MultiTaskResponse/Uplift`): pipeline
  takes `embedding` and `tabular_encoder` (`STEv2Block`, no embedding)
  **separately** and composes them.

The new style matches `avatar/nn/sequential` (`event_encoder` + `backbone`
separate, composed by `BaseSequenceModel`) and is what almost every real config
uses (`STEv2Block` in ~40 configs; `STEv2` in 5). Standardize on **embedding
separate**.

### 2.3 Generic transformer plumbing filed under "tabular"

`SublayerConnection`, `EncoderBlock`, `CrossAttentionEncoderBlock` are vanilla
transformer blocks — nothing tabular about them. → `base/layers.py` (§3.2).

### 2.4 Naming

- **`STE` = "Spatio-Temporal Encoder"** (its own docstring) for tabular data with
  neither space nor time. It's a set-transformer over feature tokens. Rename →
  `TabularTransformer`.
- `base_tabular.py` — module-name stutter.
- `STEv2Block` / `dcn_v2.py` — `v2` suffixes with no v1.

### 2.5 Concrete bugs

1. `base_tabular.py`: both `forward`s `raise ValueError` — should be
   `NotImplementedError`.
2. `ste_modeling.py`: `dict[str, any]` — lowercase `any` is the builtin, must be
   `typing.Any`.
3. `STEv2.output_dim` — comment: "for compatibility with
   `avatar.pipeline.tabular.TabularClassification`". Leaky; the pipeline should
   read `encoder.hidden_size`.
4. `docs/mlp_benchmark/train_mlp_with_target_2_target_3.yaml` references
   `_target_: avatar.nn.tabular.MLPEmbedding` — **already broken** (no such
   symbol). Fix or delete the config in this pass.
5. Zero tests for the whole `avatar/nn/tabular` tree.

---

## 3. Proposed structure — `base` / `models` / `utils`

```
avatar/nn/tabular/
├── __init__.py             # flat re-export of the public API
├── base/
│   ├── __init__.py
│   ├── encoder.py          # TabularEncoder (Protocol) + BaseTabularEncoder (nn.Module)
│   └── layers.py           # SublayerConnection, EncoderBlock, CrossAttentionEncoderBlock
├── models/
│   ├── __init__.py
│   ├── transformer.py      # TabularTransformer            (ex STEv2Block / STEv2)
│   └── dcn.py              # CrossNetV2, DCNv2
└── utils/
    ├── __init__.py
    └── masking.py          # build_feature_padding_mask, … (starts small; scaffold for future models)
```

`__init__.py` re-exports the flat public API so `avatar.nn.tabular.X` keeps
working for the top-level names (same trick as `embedding` / `sequential`).

### 3.1 `base/encoder.py` — the base classes

Mirror `avatar/nn/sequential`: a `@runtime_checkable` **Protocol** for structural
typing + a thin `nn.Module` base for in-repo implementations.

```python
from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn
from avatar.outputs import BaseTabularOutput


@runtime_checkable
class TabularEncoder(Protocol):
    """Structural type: contextualises a set of feature-token embeddings.

    Input  : inputs_embeds  (B, F, D)  — one vector per feature token
             attention_mask (B, F)     optional, 1 = keep
    Output : BaseTabularOutput, last_hidden_state (B, F, D)
    """
    hidden_size: int

    def __call__(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> BaseTabularOutput: ...


class BaseTabularEncoder(nn.Module):
    """Optional base for tabular encoders defined in this repo."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> BaseTabularOutput:
        raise NotImplementedError
```

Decisions to confirm:

- **D1 — one base, not two.** Drop `BaseTabularBackbone`. The embedding lives
  outside (in `avatar/nn/embedding/tabular`, already there), composed by the
  pipeline — same as `sequential`. Encoders are embeddings-in / hidden-states-out;
  they never see `TabularBatch`.
- **D2 — return `BaseTabularOutput`, always** (not a bare tensor). Matches
  `BaseSequenceOutput` / `SequenceBackbone`; `output_hidden_states=True` gives
  `hidden_states` for deep-layer aggregation (`agg.layer_idx < -1`).
  `agg.py` already unwraps `.last_hidden_state`, so aggregation callers are
  unaffected; the ~6 in-repo pipeline importers get a one-line change
  (`encoder(x)` → `encoder(x).last_hidden_state`).
- **D3 — `BaseTabularOutput` gets `router_logits`** (mirrors
  `BaseSequenceOutput`) so a future MoE model has somewhere to put gate logits
  without a separate dataclass. Kept minimal:
  ```python
  @dataclass
  class BaseTabularOutput:
      last_hidden_state: torch.FloatTensor = None
      hidden_states: tuple[torch.FloatTensor, ...] | None = None
      router_logits: tuple[torch.FloatTensor, ...] | None = None   # NEW
  ```

### 3.2 `base/layers.py` — common layers

`SublayerConnection`, `EncoderBlock`, `CrossAttentionEncoderBlock` move here
verbatim (fix `any` → `Any`). These are the shared building blocks for every
attention-based tabular model.

- **D4 — keep `CrossAttentionEncoderBlock`?** It is now unused (its only caller,
  `TreatmentCrossAttnEncoder`, was deleted). It's a small, genuinely reusable
  layer — recommend keeping it in `base/layers.py`. Delete if you'd rather not
  carry unused code.
- Note: the `sequential` side hand-rolls its own attention block in
  `event_encoder/attention.py`. Out of scope here, but `base/layers.py` could
  later be promoted to `avatar/nn/utils/` and shared. Keeping it under
  `tabular/base/` for now per the agreed structure — tabular stays
  self-contained.

### 3.3 `models/transformer.py` — `TabularTransformer` (ex `STEv2`)

Merge `STEv2Block` + `STEv2` into one class:

```python
class TabularTransformer(BaseTabularEncoder):
    """Set-transformer over feature tokens: (B, F, D) -> (B, F, D)."""

    def __init__(self, hidden_size, num_heads, num_layers,
                 attn_dropout=0.15, need_weights=False):
        super().__init__(hidden_size=hidden_size)
        self.blocks = nn.ModuleList(
            EncoderBlock(hidden_size, num_heads, attn_dropout)
            for _ in range(num_layers)
        )

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False):
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hs = []
        x = inputs_embeds
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
            if output_hidden_states:
                hs.append(x)
        return BaseTabularOutput(
            last_hidden_state=x,
            hidden_states=tuple(hs) or None,
        )
```

- **D5 — drop the embedding-owning bundle.** `STEv2` (embedding + block) is used
  by 5 configs, all in `examples/tabular_hidden_states`; migrate them to the
  `embedding` + `TabularTransformer` sibling style under
  `TabularWithAggregatedStates` (already how `s_learner` etc. are written). One
  composition style in the repo. (`sequential` needs `model/` because
  `TransformersWrapper` does real work — attention-mask padding; the tabular
  bundle does nothing but chain two calls.)
- `output_dim` field removed; pipelines read `encoder.hidden_size`.

### 3.4 `models/dcn.py` — `DCNv2`

`DCNv2` operates on a **pooled** `[B, d]` vector, so it does **not** satisfy
`TabularEncoder` (`[B, F, D] → [B, F, D]`). It's a cross-feature *interaction
head*, not a feature-token encoder.

- **D6 — how to place it:**
  - **D6-a (recommended):** keep `DCNv2` in `models/dcn.py` as a plain
    `nn.Module` with its own documented contract (`[B, d] → [B, d']`), and wire
    it as an injectable `out_head` in `TabularClassification` / `SLearner` (they
    already accept `out_head` / `output_head`). Add one example config so it's
    exercised.
  - **D6-b:** give it an FT-Transformer-style adapter so it *does* take
    `[B, F, D]` (flatten `F·D` or mean-pool → cross+deep → `[B, 1, d']`), making
    it conform to `TabularEncoder`. More work, changes its semantics.
  - **D6-c:** delete it (dead code; recover from git if needed).

  Recommend **D6-a** unless there's an out-of-repo experiment using it.
- If we expect several pooled-vector nets later, `base/` can grow a second base
  `BaseTabularMixer` (`[B, d] → [B, d']`); not needed for just `DCNv2`.

### 3.5 `utils/`

Tabular-specific helpers. Starts near-empty — a `build_feature_padding_mask`
(`(B, F) attention_mask → key_padding_mask` for `nn.MultiheadAttention`) is the
one clear candidate today. Scaffold for helpers that future models (FT-Transformer,
feature-type embeddings, feature bias) will want. If it stays empty after
migration, fold the one helper into `base/layers.py` and drop `utils/`.

### What stays in `avatar/pipeline`

loss / `num_classes` / `task_type`; output heads (`FeedForwardNetwork`,
`FeedForwardHead`); aggregation choice (`get_aggregation_layer`); treatment /
group / task-name handling; `TabularOutput` / `*UpliftOutput` / `MMoEOutput`
(already in `avatar/outputs.py`).

---

## 4. Migration strategy

Same as the `sequential` refactor — **hard rename, update every in-repo
reference, one-release deprecation shims for out-of-repo configs.**

### Renames

| old | new |
|---|---|
| `avatar.nn.tabular.base_tabular` (module) | `avatar.nn.tabular.base.encoder` |
| `BaseTabularEncoder` (tensor→tensor) | `BaseTabularEncoder` (kept name, new contract) + `TabularEncoder` Protocol |
| `BaseTabularBackbone` | **removed** (embedding moves out) |
| `STEv2Block`, `STEv2` | `TabularTransformer` |
| `avatar.nn.tabular.ste` (subpackage) | split → `avatar.nn.tabular.base` + `avatar.nn.tabular.models` |
| `SublayerConnection`, `EncoderBlock`, `CrossAttentionEncoderBlock` | `avatar.nn.tabular.base.layers.*` |
| `avatar.nn.tabular.dcnv2` | `avatar.nn.tabular.models.dcn` |
| `MoeTabularOutput` | **removed** (done) — `BaseTabularOutput.router_logits` |

### Blast radius

In-repo Python (6 importers of `from avatar.nn.tabular import BaseTabular*`):
`pipeline/tabular/{supervised,tabular_aggregation}.py`,
`pipeline/uplift/s_learner.py`, `pipeline/multi_task/{mmoe,response,uplift}.py` —
updated to the new base + `encoder(...).last_hidden_state`.

Configs:
- `_target_: avatar.nn.tabular.ste.STEv2Block` — **~40 configs** (mostly
  `experiments/sbercampaign_pilot/`, some `examples/uplift_modeling/`). In-repo
  ones rewritten to `avatar.nn.tabular.TabularTransformer`. Deprecation shim
  module `avatar/nn/tabular/ste/__init__.py` re-exports
  `STEv2Block = STEv2 = TabularTransformer` (+ `DeprecationWarning`) for one
  release to cover cluster configs.
- `_target_: avatar.nn.tabular.STEv2` — 5 configs in
  `examples/tabular_hidden_states/`; rewritten (D5) + shim alias.
- `docs/mlp_benchmark/train_mlp_with_target_2_target_3.yaml` — fix the broken
  `avatar.nn.tabular.MLPEmbedding` target.

### Step plan (each step leaves the repo green)

1. `avatar/nn/tabular/base/` — new `encoder.py` (Protocol + base), `layers.py`
   (move `SublayerConnection` / `EncoderBlock` / `CrossAttentionEncoderBlock`,
   fix `Any`).
2. Add `BaseTabularOutput.router_logits`.
3. `avatar/nn/tabular/models/transformer.py` — merge `STEv2Block` + `STEv2` →
   `TabularTransformer` on the new contract.
4. `avatar/nn/tabular/models/dcn.py` — move `DCNv2` (per D6).
5. `avatar/nn/tabular/utils/` — `build_feature_padding_mask` (or skip).
6. New flat `avatar/nn/tabular/__init__.py`; deprecation shim
   `avatar/nn/tabular/ste/__init__.py` + `base_tabular.py` shim.
7. Update the 6 in-repo pipeline importers to the new base + output contract.
8. Update in-repo configs (`examples/`) + fix the mlp_benchmark config.
9. Tests: `tests/nn/tabular/` — `test_imports.py` (flat API + shim warnings),
   `test_transformer.py` (shape + mask + `output_hidden_states`),
   `test_output_contract.py` (every model returns `BaseTabularOutput`),
   `test_dcn.py`.

---

## 5. Risks

- **Cluster configs** using `avatar.nn.tabular.ste.STEv2Block` /
  `avatar.nn.tabular.STEv2` — covered by the one-release shim; must be announced.
- **Return-type change** (`Tensor` → `BaseTabularOutput`) for `STEv2Block`'s
  successor. In-repo callers funnel through `agg.py` (handles both) or are
  updated in step 7. Out-of-repo callers that indexed the tensor directly break
  at the shim boundary — accept for one release, or have the shim wrap the output.
- **`DCNv2` has no test** — add a characterization test (forward shape / dtype /
  determinism) before moving it, or delete (D6-c).
