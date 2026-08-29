# Design: restructure `avatar/nn/tabular`

Status: **draft / proposal** (2026-08-29). Follows the `avatar/nn/embedding`
(commit `719fe71`) and `avatar/nn/sequential` (`3759c1a..734f2aa`) precedents.

Goal (from the ask): `avatar/nn/tabular` should hold **reusable tabular neural
nets**; `avatar/pipeline/*` wraps them into task settings — classification,
regression, uplift, multi-label, multi-task. This doc answers: what should live
in `nn/tabular` vs `pipeline`, which base classes are needed, and what the
input/output contract is.

---

## 1. What the module is today

```
avatar/nn/tabular/
├── __init__.py            # flat re-export
├── base_tabular.py        # BaseTabularBackbone, BaseTabularEncoder
├── dcnv2/dcn_v2.py        # CrossNetV2, DCNv2
├── moe/moe_modeling.py    # FFN, UniversalGate, GateTopK, MLPGate, MoE,
│                          #   ExpertsWrapper, MLPExperts
├── ste/ste_modeling.py    # SublayerConnection, EncoderBlock,
│                          #   CrossAttentionEncoderBlock, STEv2Block, STEv2,
│                          #   MoEEncoderBlock, MoESTEv2
└── uplift/mt_modeling.py  # FeedForwardHead, TreatmentCrossAttnEncoder,
                           #   MultiTreatmentSTE
```

### Inventory + proposed fate

| symbol | file | role | used by | fate |
|---|---|---|---|---|
| `BaseTabularBackbone` | `base_tabular.py` | `nn.Module`, holds `embedding`; `forward(TabularBatch)` | `STEv2`, `TabularWithAggregatedStates` | → `encoder/base.py`, merged into one base (see §3) |
| `BaseTabularEncoder` | `base_tabular.py` | `nn.Module`; `forward(input_embeds: Tensor)` | `STEv2Block`, pipeline type hints | → `encoder/base.py` as the single base + Protocol |
| `SublayerConnection` | `ste/ste_modeling.py` | Add&Norm residual wrapper | ste internals | → `avatar/nn/utils/transformer.py` (generic) |
| `EncoderBlock` | `ste/ste_modeling.py` | 1 transformer encoder layer (MHA + FFN) | ste, `mt_modeling` | → `avatar/nn/utils/transformer.py` |
| `CrossAttentionEncoderBlock` | `ste/ste_modeling.py` | cross-attention variant of `EncoderBlock` | `mt_modeling` | → `avatar/nn/utils/transformer.py` |
| `STEv2Block` | `ste/ste_modeling.py` | stack of `EncoderBlock`; `[B,F,D]→[B,F,D]` | ~40 configs, `s_learner`, `supervised`, `mmoe` | → `encoder/transformer.py` as **`TabularTransformerEncoder`** |
| `STEv2` | `ste/ste_modeling.py` | `embedding` + `STEv2Block`; → `BaseTabularOutput` | `examples/tabular_hidden_states/*` (5 configs) | → `encoder/transformer.py`; see §4 (keep as thin bundle or drop) |
| `MoEEncoderBlock` | `ste/ste_modeling.py` | `EncoderBlock` with MoE-FFN | `MoESTEv2` | → `encoder/moe.py` |
| `MoESTEv2` | `ste/ste_modeling.py` | `STEv2` subclass that `del`s + rebuilds `encoder_blocks` | **no configs, no tests** | → `encoder/moe.py` as **`MoETabularEncoder`**, rewritten as a sibling not a subclass |
| `FFN`, `UniversalGate`, `GateTopK`, `MLPGate`, `MLPExperts`, `ExpertsWrapper` | `moe/moe_modeling.py` | MoE gate/expert primitives | ste MoE, `mmoe` (partially) | stay in `moe/` |
| `MoE` | `moe/moe_modeling.py` | expert combinator; returns `MoeTabularOutput(loss=…)` | `MoEEncoderBlock` | stays in `moe/`, but return type folded into `BaseTabularOutput` (§3); `loss` field dropped |
| `DCNv2`, `CrossNetV2` | `dcnv2/dcn_v2.py` | `[B,d]→[B,d']` cross-feature interaction on the **aggregated** vector | **no configs, no tests** | → `head/dcn.py` (it's a head, not an encoder) — or delete, see §5 |
| `FeedForwardHead` | `uplift/mt_modeling.py` | LayerNorm+MLP head | `MultiTreatmentSTE` | → move with `MultiTreatmentSTE` into `avatar/pipeline` |
| `TreatmentCrossAttnEncoder` | `uplift/mt_modeling.py` | cross-attn: `(feature_embeds, treat_embeds)→treat_repr` | `MultiTreatmentSTE` | → `encoder/cross_attention.py` (it *is* a reusable encoder) |
| `MultiTreatmentSTE` | `uplift/mt_modeling.py` | **full task model**: heads + sigmoid + returns `(basic_probs, treatment_probs, targets)` tuple | **no configs, no tests** | → `avatar/pipeline/uplift/` (it is a pipeline, not an nn block) — or delete |

---

## 2. Review — what's wrong right now

### 2.1 The base classes don't define one contract

There are two bases with overlapping, under-specified jobs:

- `BaseTabularBackbone(embedding)` — owns the embedding, `forward(TabularBatch)`.
- `BaseTabularEncoder()` — no embedding, `forward(input_embeds: Tensor)`.

and the **return type is inconsistent across the tree**:

| module | base | returns |
|---|---|---|
| `STEv2Block` | `BaseTabularEncoder` | bare `Tensor` `[B,F,D]` |
| `STEv2` | `BaseTabularBackbone` | `BaseTabularOutput` |
| `STEv2` (when passed a raw tensor) | — | still `BaseTabularOutput`, but skips embedding — dual contract in one method |
| `MoE` | `nn.Module` | `MoeTabularOutput` (a *third* dataclass, with a `loss` field an encoder has no business setting) |
| `MoESTEv2` | `STEv2` | `Tensor` **or** `BaseTabularOutput` depending on a `return_tensor` ctor flag |

Consumers then each assume something different:

- `pipeline/uplift/s_learner.py`, `pipeline/tabular/supervised.py`: declare the
  arg as `tabular_encoder: BaseTabularEncoder`, then call it and feed the result
  straight to `agg_layer` — they expect a **tensor**.
- `pipeline/tabular/tabular_aggregation.py`: declares `backbone:
  BaseTabularBackbone`, also feeds `agg_layer`.
- `pipeline/multi_task/{response,uplift}.py`: call `tabular_encoder(...)` and use
  the result directly as `combined_features`.

It only works because `avatar/nn/utils/agg.py::BaseAggregation.apply_expanded_mask`
silently accepts both — `if hasattr(states, "last_hidden_state"): states =
states.last_hidden_state`. The polymorphic aggregation layer is papering over an
undefined encoder contract.

### 2.2 `embedding` inside vs. outside the module — two composition styles

- **Old style** (`STEv2` + `TabularWithAggregatedStates`): the encoder owns the
  embedding; the pipeline gets the whole bundle.
- **New style** (`SLearner`, `SupervisedLearner`, `MultiTaskResponse/Uplift`):
  the pipeline takes `embedding` and `tabular_encoder` (`STEv2Block`, no
  embedding) **separately** and composes them itself.

The new style is the one that matches `avatar/nn/sequential` (`event_encoder` +
`backbone` are separate, composed by `BaseSequenceModel`). It's also what almost
every real config uses (`STEv2Block` appears in ~40 configs; `STEv2` in 5).
Recommendation: standardize on **embedding separate**, encoder = embeddings-in /
hidden-states-out.

### 2.3 Task models are living in `avatar/nn/tabular`

`MultiTreatmentSTE` has classification heads, a `nn.Sigmoid()`, a
`training`-conditional embedding processor, and returns a bare
`(basic_probs, treatment_probs, targets)` tuple. That's a *pipeline* by the
definition in the ask. Same argument (weaker) for `MoE` owning a `loss` field.
`avatar/nn/tabular` should contain **no loss, no task head, no target handling**.

### 2.4 Generic transformer plumbing is filed under "tabular"

`SublayerConnection`, `EncoderBlock`, `CrossAttentionEncoderBlock` are vanilla
transformer building blocks — nothing tabular about them. They belong in
`avatar/nn/utils/` next to `FeedForwardNetwork` / `agg.py`, where the
`sequential` side can use them too (it currently hand-rolls its own in
`event_encoder/attention.py`).

### 2.5 Naming

- **`STE` = "Spatio-Temporal Encoder"** (per its own docstring) for tabular data
  that has neither space nor time. It's a set-transformer over feature tokens.
  Rename → `TabularTransformerEncoder`.
- `base_tabular.py` — module-name stutter (`avatar.nn.tabular.base_tabular`).
- `BaseTabularBackbone` / `BaseTabularEncoder` — the names don't signal the
  `TabularBatch`-vs-`Tensor` difference that is their only distinction.
- `MoESTEv2`, `STEv2Block`, `dcn_v2.py` — the `v2` suffixes are load-bearing on
  nothing (there is no v1).

### 2.6 Concrete bugs / smells

1. `base_tabular.py`: both `forward`s `raise ValueError(...)` — should be
   `NotImplementedError`.
2. `moe_modeling.py`, `ste_modeling.py`: `dict[str, any]` (~7 sites) — lowercase
   `any` is the builtin function; must be `typing.Any`.
3. `moe_modeling.py`: attribute typo `self.aggregateion_layer` (3 sites).
4. `MoeTabularOutput.task_gated_weights` typed `dict[str, ...]` but assigned a
   `Tensor`.
5. `MoESTEv2.__init__`: `del self.encoder_blocks` then rebuild; `self.emb =
   embedding` duplicates the inherited `self.embedding`; `return_tensor` flag
   switches the return type. Rewrite as a standalone encoder.
6. `STEv2.output_dim` — comment says "for compatibility with
   `avatar.pipeline.tabular.TabularClassification`". Leaky coupling; the pipeline
   should read `encoder.hidden_size`, not have the encoder pre-shape a field for
   one consumer.
7. `MoE.forward`: `_get_not_null_expert_positions` is passed the already-reshaped
   `[B, n_exp, 1, 1]` tensor; works only by accident of dim 0 surviving reshape.
8. `docs/mlp_benchmark/train_mlp_with_target_2_target_3.yaml` references
   `_target_: avatar.nn.tabular.MLPEmbedding` — **already broken** (no such
   symbol). Fix or delete the config as part of this pass.
9. Zero tests for the entire `avatar/nn/tabular` tree.

---

## 3. Proposed base classes + contract

Mirror `avatar/nn/sequential`: a `@runtime_checkable` **Protocol** for structural
typing + a thin `nn.Module` base for in-repo implementations.

```python
# avatar/nn/tabular/encoder/base.py
from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn
from avatar.outputs import BaseTabularOutput


@runtime_checkable
class TabularEncoder(Protocol):
    """Structural type: contextualises a set of feature-token embeddings.

    Input  : inputs_embeds  (B, F, D)  — one vector per feature token
             attention_mask (B, F)     optional, 1 = keep
    Output : BaseTabularOutput with last_hidden_state (B, F, D)
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

### Decisions to confirm

- **D1 — one base, not two.** Drop `BaseTabularBackbone`. The embedding lives
  outside the encoder (in `avatar/nn/embedding/tabular`, already there), composed
  by the pipeline — same as `sequential`. Encoders are embeddings-in /
  hidden-states-out.
- **D2 — return `BaseTabularOutput`, always.** Not a bare tensor. This matches
  `BaseSequenceOutput` / `SequenceBackbone`, and `output_hidden_states=True`
  gives `hidden_states` for deep-layer aggregation (`agg.layer_idx < -1`), which
  the tensor contract can't express. `agg.py` already unwraps `.last_hidden_state`,
  so aggregation callers are unaffected.
- **D3 — kill `MoeTabularOutput`.** Fold its one useful field into
  `BaseTabularOutput`:

  ```python
  @dataclass
  class BaseTabularOutput:
      last_hidden_state: torch.FloatTensor = None
      hidden_states: tuple[torch.FloatTensor, ...] | None = None
      router_logits: tuple[torch.FloatTensor, ...] | None = None   # NEW (mirrors BaseSequenceOutput)
  ```
  The MoE `aux_loss` / load-balancing term, if we keep one, is computed by the
  pipeline from `router_logits` (same as `router_logits` handling on the
  sequence side), never stored on the encoder output.
- **D4 — `attention_mask` in the signature even though most tabular batches are
  dense.** Costs nothing, needed for padded feature sets / masked-feature
  ablations (`HierarchicalFeatureGate` in `mmoe.py` already zeroes feature
  tokens), and keeps the signature identical to `SequenceBackbone`.
- **D5 — no `TabularBatch` in `avatar/nn/tabular` at all.** `TabularBatch →
  embeddings` is the embedding layer's job. The encoder never sees the raw batch.
  (`STEv2`'s `isinstance(tab_features, Tensor)` branch goes away.)

### What stays a task-model concern (in `avatar/pipeline`)

- loss functions, `num_classes`, `task_type`
- output heads (`FeedForwardNetwork`, `FeedForwardHead`)
- aggregation choice (`get_aggregation_layer`) — arguably; it's currently called
  from both sides. Proposal: pipeline owns it, as it does today.
- treatment / group / task-name handling, uplift double-forward
- `TabularOutput`, `*UpliftOutput`, `MMoEOutput`, … (already in `avatar/outputs.py`)

---

## 4. Proposed layout

```
avatar/nn/tabular/
├── __init__.py                 # flat re-export of the public API
├── encoder/
│   ├── __init__.py
│   ├── base.py                 # TabularEncoder (Protocol) + BaseTabularEncoder
│   ├── transformer.py          # TabularTransformerEncoder            (ex STEv2Block)
│   ├── moe.py                  # MoETabularEncoder                    (ex MoESTEv2 + MoEEncoderBlock)
│   └── cross_attention.py      # TreatmentCrossAttnEncoder            (unchanged role)
├── moe/
│   ├── __init__.py
│   └── modeling.py             # UniversalGate, GateTopK, MLPGate, MLPExperts,
│                               #   ExpertsWrapper, MoE  (ex moe_modeling.py)
└── head/                       # OPTIONAL — only if DCNv2 is kept (see §5)
    ├── __init__.py
    └── dcn.py                  # CrossNetV2, DCNv2
```

Moved out of the module:

```
avatar/nn/utils/transformer.py  # SublayerConnection, EncoderBlock, CrossAttentionEncoderBlock
avatar/pipeline/uplift/multi_treatment.py   # MultiTreatmentSTE + FeedForwardHead  (if kept)
```

`__init__.py` re-exports the flat public API so `avatar.nn.tabular.X` keeps
working for the top-level names (same trick as `embedding` / `sequential`).

### 4.1 The `STEv2` bundle (embedding + encoder)

Two options — **need a decision (D6)**:

- **D6-a (recommended): drop it.** Only 5 configs use `avatar.nn.tabular.STEv2`,
  all in `examples/tabular_hidden_states`. Migrate them to the pipeline-composes
  style (`embedding` + `TabularTransformerEncoder` as siblings under
  `TabularWithAggregatedStates`), which is already how `s_learner` etc. are
  written. One consistent composition style in the repo.
- **D6-b: keep a thin `TabularModel`** (`embedding` + `encoder` → `BaseTabularOutput`)
  as the tabular analogue of `BaseSequenceModel`, for callers who want one object.
  More symmetric with `sequential`, but adds a second composition path.

The `sequential` side needed `model/` because `TransformersWrapper` does
non-trivial work (attention-mask padding for prepended tokens). The tabular
bundle does nothing but call `embedding` then `encoder` — so D6-a.

---

## 5. `DCNv2` — decide (D7)

`DCNv2` is `[B, d] → [B, d']`: it operates on the **pooled** feature vector, so
it's a **head / interaction block**, not a feature-token encoder — it doesn't fit
`TabularEncoder`. It has no config and no test.

- **D7-a: delete** (dead code; resurrect from git if needed).
- **D7-b: keep** under `avatar/nn/tabular/head/dcn.py` and wire it as a valid
  `out_head` option in `TabularClassification` / `SLearner` (they already accept
  an injected `out_head` / `output_head`). Needs one example config to be real.

Recommendation: **D7-a** unless there's an active experiment using it out-of-repo.

---

## 6. Migration strategy

Same as the `sequential` refactor — **hard rename, update every in-repo
reference, one-release deprecation shims for out-of-repo configs.**

### Renames

| old | new |
|---|---|
| `avatar.nn.tabular.base_tabular` (module) | `avatar.nn.tabular.encoder.base` |
| `BaseTabularEncoder` (tensor→tensor) | `BaseTabularEncoder` (kept name, new contract) + `TabularEncoder` Protocol |
| `BaseTabularBackbone` | **removed** (embedding moves out) |
| `STEv2Block` | `TabularTransformerEncoder` |
| `STEv2` | **removed** (D6-a) |
| `MoESTEv2` | `MoETabularEncoder` |
| `avatar.nn.tabular.ste` (subpackage) | `avatar.nn.tabular.encoder` |
| `avatar.nn.tabular.moe.moe_modeling` | `avatar.nn.tabular.moe.modeling` |
| `SublayerConnection`, `EncoderBlock`, `CrossAttentionEncoderBlock` | `avatar.nn.utils.transformer.*` |
| `MultiTreatmentSTE`, `FeedForwardHead` | `avatar.pipeline.uplift.multi_treatment.*` |
| `MoeTabularOutput` | **removed** — `BaseTabularOutput.router_logits` |

### Blast radius

In-repo Python (6 importers of `from avatar.nn.tabular import BaseTabular*`):
`pipeline/tabular/{supervised,tabular_aggregation}.py`,
`pipeline/uplift/s_learner.py`, `pipeline/multi_task/{mmoe,response,uplift}.py` —
all updated to the new base + `encoder(...).last_hidden_state`.

Configs:
- `_target_: avatar.nn.tabular.ste.STEv2Block` — **~40 configs** (mostly
  `experiments/sbercampaign_pilot/`, some `examples/uplift_modeling/`). In-repo
  ones get rewritten to `avatar.nn.tabular.TabularTransformerEncoder`. A
  deprecation shim module `avatar/nn/tabular/ste/__init__.py` re-exports
  `STEv2Block = TabularTransformerEncoder` (+ `DeprecationWarning`) for one
  release to cover cluster configs.
- `_target_: avatar.nn.tabular.STEv2` — 5 configs in
  `examples/tabular_hidden_states/`; rewritten (D6-a) + shim alias.
- `docs/mlp_benchmark/train_mlp_with_target_2_target_3.yaml` — fix the
  pre-existing broken `avatar.nn.tabular.MLPEmbedding` target.

### Step plan (each step leaves the repo green)

1. `avatar/nn/utils/transformer.py` — move `SublayerConnection` / `EncoderBlock` /
   `CrossAttentionEncoderBlock` out; re-point `ste` + `mt_modeling` imports.
2. Add `BaseTabularOutput.router_logits`; delete `MoeTabularOutput`, update `MoE`.
3. `avatar/nn/tabular/encoder/` — new `base.py` (Protocol + base), move
   `STEv2Block`→`TabularTransformerEncoder`, `MoESTEv2`→`MoETabularEncoder`
   (rewritten), `TreatmentCrossAttnEncoder`.
4. `avatar/nn/tabular/moe/modeling.py` — rename from `moe_modeling.py`; fix
   `Any`, `aggregateion_layer` typo.
5. `MultiTreatmentSTE` + `FeedForwardHead` → `avatar/pipeline/uplift/multi_treatment.py`.
6. `DCNv2` — delete (D7-a) or move to `head/` (D7-b).
7. New flat `avatar/nn/tabular/__init__.py`; deprecation shims at
   `avatar/nn/tabular/ste/__init__.py` (+ `base_tabular.py` shim).
8. Update the 6 in-repo pipeline importers to the new base + output contract.
9. Update in-repo configs (`examples/`, in-repo `experiments/` if any) + fix the
   mlp_benchmark config.
10. Tests: `tests/nn/tabular/` — `test_imports.py` (flat API + shim warnings),
    `test_transformer_encoder.py` (shape + mask + `output_hidden_states`),
    `test_moe_encoder.py`, `test_output_contract.py` (every encoder returns
    `BaseTabularOutput`).

---

## 7. Risks

- **Cluster configs** using `avatar.nn.tabular.ste.STEv2Block` /
  `avatar.nn.tabular.STEv2` — covered by the one-release shim; must be announced.
- **Return-type change** (`Tensor` → `BaseTabularOutput`) for `STEv2Block`'s
  successor. In-repo callers all funnel through `agg.py` (handles both) or are
  updated in step 8. Out-of-repo callers that indexed the tensor directly will
  break at the shim boundary — the shim can return a thin tensor-compatible
  wrapper if needed, or we accept the break for one release.
- **`MoESTEv2` / `MultiTreatmentSTE` / `DCNv2` have no tests** — rewriting them is
  unverified against any baseline. Mitigation: add characterization tests
  (forward shape / dtype / determinism) *before* touching them, or confirm they
  can be deleted.
- **`nn/utils/transformer.py`** — check for a name clash with
  `transformers` (the HF package) in import lines; use explicit relative imports.
