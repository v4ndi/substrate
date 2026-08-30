# Design: restructure `avatar/nn/tabular`

Status: **accepted** (2026-08-29). Follows the `avatar/nn/embedding` (`719fe71`)
and `avatar/nn/sequential` (`3759c1a..734f2aa`) precedents.

Goal: `avatar/nn/tabular` holds **reusable tabular neural nets**;
`avatar/pipeline/*` wraps them into task settings (classification, regression,
uplift, multi-label, multi-task).

Locked decisions:

- **D1** — one base class (`BaseTabularEncoder`). `BaseTabularBackbone` removed.
- **D2** — encoders always return `BaseTabularOutput` (never a bare tensor).
- **D3** — no `router_logits` on `BaseTabularOutput` (no consumer today).
- **D4** — `CrossAttentionEncoderBlock` deleted (unused).
- **D5** — the embedding-owning bundle (`STEv2`) is dropped; encoders take
  embeddings, the pipeline composes `embedding + encoder + aggregation`.
- **D6** — `DCNv2` / `CrossNetV2` deleted (dead code, no config, no test).
- `moe/` + `uplift/` subpackages already deleted (commit `895c6d9`).

---

## 1. Target layout

```
avatar/nn/tabular/
├── __init__.py             # flat re-export of the public API
├── base/
│   ├── __init__.py
│   ├── encoder.py          # BaseTabularEncoder
│   └── layers.py           # SublayerConnection, EncoderBlock
├── models/
│   ├── __init__.py
│   └── transformer.py      # TabularTransformer   (ex STEv2Block + STEv2, merged)
└── utils/
    ├── __init__.py
    └── masking.py          # build_feature_padding_mask
```

Plus a one-release deprecation shim: `avatar/nn/tabular/ste/__init__.py`
re-exports `STEv2Block` / `STEv2` → `TabularTransformer` with a
`DeprecationWarning` (covers out-of-repo cluster configs).

`__init__.py` re-exports the flat public API so `avatar.nn.tabular.X` keeps
working (same trick as `embedding` / `sequential`).

---

## 2. Contract

### `avatar/nn/tabular/base/encoder.py`

```python
import torch
import torch.nn as nn

from avatar.outputs import BaseTabularOutput


class BaseTabularEncoder(nn.Module):
    """Base class for tabular feature-token encoders.

    Contextualises a set of feature-token embeddings: (B, F, D) -> (B, F, D).
    The embedding layer lives outside (avatar.nn.embedding.tabular); the pipeline
    composes embedding + encoder + aggregation.

    Contract:
        Input  : inputs_embeds  (B, F, D) - one embedding vector per feature token
                 attention_mask (B, F)    - optional, 1 = keep, 0 = pad
        Output : BaseTabularOutput, last_hidden_state (B, F, D);
                 hidden_states set iff output_hidden_states=True
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> BaseTabularOutput:
        raise NotImplementedError(
            "forward must be implemented by BaseTabularEncoder subclasses"
        )
```

`BaseTabularOutput` (in `avatar/outputs.py`) is unchanged:

```python
@dataclass
class BaseTabularOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
```

`avatar/nn/utils/agg.py::apply_expanded_mask` already unwraps
`.last_hidden_state` / indexes `.hidden_states[layer_idx]`, so every aggregation
caller works with the output object unchanged.

### `avatar/nn/tabular/base/layers.py`

`SublayerConnection` and `EncoderBlock` move here verbatim (pre-norm MHA + FFN
transformer block). `dict[str, any]` type hints fixed to `Any` where present.

### `avatar/nn/tabular/models/transformer.py`

```python
class TabularTransformer(BaseTabularEncoder):
    """Set-transformer over feature tokens: (B, F, D) -> (B, F, D).

    A stack of pre-norm transformer encoder blocks (MHA + FFN), no positional
    encoding - feature order is meaningless; the per-feature signal comes from
    the embedding layer.
    """

    def __init__(self, hidden_size, num_heads, num_layers, attn_dropout=0.15):
        super().__init__(hidden_size=hidden_size)
        self.blocks = nn.ModuleList(
            EncoderBlock(hidden_size, num_heads, attn_dropout)
            for _ in range(num_layers)
        )

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False):
        key_padding_mask = build_feature_padding_mask(attention_mask)
        collected = []
        x = inputs_embeds
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
            if output_hidden_states:
                collected.append(x)
        return BaseTabularOutput(
            last_hidden_state=x,
            hidden_states=tuple(collected) if output_hidden_states else None,
        )
```

Dropped from the old code: `need_weights` (inert — attention weights were
computed then discarded; no config sets it), `output_dim` (pipelines read
`hidden_size`), the `isinstance(tab_features, Tensor)` batch branch (D5).

### `avatar/nn/tabular/utils/masking.py`

```python
def build_feature_padding_mask(attention_mask):
    """(B, F) attention mask (1 = keep) -> (B, F) bool key_padding_mask for
    nn.MultiheadAttention (True = ignore). None -> None."""
    if attention_mask is None:
        return None
    return ~attention_mask.bool()
```

---

## 3. Renames

| old | new |
|---|---|
| `avatar.nn.tabular.base_tabular` (module) | `avatar.nn.tabular.base.encoder` + `avatar.nn.tabular.base.layers` |
| `BaseTabularEncoder` (tensor→tensor) | `BaseTabularEncoder` (same name, new contract) |
| `BaseTabularBackbone` | **removed** |
| `STEv2Block`, `STEv2` | `TabularTransformer` |
| `avatar.nn.tabular.ste` (subpackage) | `base/` + `models/` (+ shim `ste/__init__.py`) |
| `SublayerConnection`, `EncoderBlock` | `avatar.nn.tabular.base.layers.*` |
| `CrossAttentionEncoderBlock` | **removed** |
| `avatar.nn.tabular.dcnv2`, `DCNv2`, `CrossNetV2` | **removed** |

---

## 4. Blast radius + migration

### Pipeline (Python)

| file | change |
|---|---|
| `pipeline/tabular/tabular_aggregation.py` | `TabularWithAggregatedStates` takes `embedding` + `encoder` (was `backbone: BaseTabularBackbone` owning the embedding); `forward` = `embedding(batch) → encoder(embeds) → agg` |
| `pipeline/tabular/supervised.py` | import path only (`BaseTabularEncoder` moves); already composes embedding + encoder separately, agg unwraps the output object |
| `pipeline/uplift/s_learner.py` | import path only (same) |
| `pipeline/multi_task/{mmoe,response,uplift}.py` | import path only; these pass `MMoEBackbone` (own class), the `BaseTabularEncoder` hint is decorative |

### Configs

- `_target_: avatar.nn.tabular.STEv2` — **8 configs**, all
  `examples/tabular_hidden_states/`. Rewritten: `backbone: {STEv2, embedding}` →
  sibling `embedding:` + `encoder: {TabularTransformer}` under
  `TabularWithAggregatedStates`; interpolations
  `${...tabular_model.backbone.embedding.hidden_size}` →
  `${...tabular_model.embedding.hidden_size}`.
- `_target_: avatar.nn.tabular.ste.STEv2Block` — **46 configs**
  (`examples/uplift_modeling/` ×2, `experiments/sbercampaign_pilot/` ×44).
  Mechanical: `→ avatar.nn.tabular.TabularTransformer` (signature-compatible).
  Also covered by the shim.
- `docs/mlp_benchmark/train_mlp_with_target_2_target_3.yaml` — pre-existing broken
  `_target_: avatar.nn.tabular.MLPEmbedding`; fix to `TabularEmbedding` or delete.

### Step plan (repo green at every commit)

1. `base/` (encoder + layers), `utils/masking.py`.
2. `models/transformer.py` — `TabularTransformer`.
3. New flat `__init__.py`; `ste/__init__.py` shim; delete `base_tabular.py`,
   `ste/ste_modeling.py`, `dcnv2/`.
4. Pipeline: `TabularWithAggregatedStates` + the 5 import-path updates.
5. Configs: `examples/` rewrite, `experiments/` sed, mlp_benchmark fix.
6. Tests: `tests/nn/tabular/{test_imports,test_transformer}.py`.

---

## 5. Risks

- **Cluster configs** on `avatar.nn.tabular.ste.STEv2Block` /
  `avatar.nn.tabular.STEv2` — one-release shim; `STEv2`'s old
  `embedding=`-owning signature cannot be shimmed (it's a different composition),
  so out-of-repo `STEv2` users must migrate manually — announce.
- **Return-type change** (`Tensor` → `BaseTabularOutput`) — in-repo callers all
  funnel through `agg.py` (handles both); out-of-repo callers indexing the tensor
  break at the shim boundary.
- `experiments/sbercampaign_pilot/` configs reference non-local data paths and
  pipeline classes (`SLearnerExp`) not exercised by the test suite — the
  `_target_` rename is mechanical and safe, but unverified end-to-end.
