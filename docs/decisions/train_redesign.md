# Design: rewrite `avatar/train.py` — pure `torch.distributed`, callbacks, loss placement

Status: **implemented** (2026-08-30). See §6 for what shipped and where it
deviates from this proposal.

Three questions from the ask:

1. Drop `accelerate`, run on plain `torch.distributed`.
2. Add a callback system ("колбэки для масштабирования").
3. Loss: computed inside the model, or as a separate `nn.Module`?

---

## 0. What `accelerate` does for us today (the coupling to remove)

`avatar/train.py` (772 LOC) + `avatar/train_utils.py` + `avatar/accelerate_utils.py`
+ `avatar/inference.py` + `avatar/cam_inference.py`. Every use of the
`Accelerator`:

| accelerate feature | where | plain-torch replacement |
|---|---|---|
| process group init, `InitProcessGroupKwargs(timeout)` | `init_accelerate` | `dist.init_process_group("nccl", timeout=...)` from `torchrun` env |
| `accelerator.device`, `is_main_process`, `is_local_main_process`, `num_processes`, `process_index` | everywhere | a small `DistEnv` dataclass reading `RANK`/`LOCAL_RANK`/`WORLD_SIZE` |
| `accelerator.prepare(model, optimizer, scheduler, dataloader)` | `train`, `evaluation` | manual `DDP(model, device_ids=[local_rank], find_unused_parameters=…)`; optimizer/scheduler/dataloader used as-is |
| `accelerator.backward(loss)` | step loop | `loss.backward()` (+ `scaler.scale(loss).backward()` under AMP) |
| `accelerator.accumulate(model)` ctx + `accelerator.sync_gradients` | step loop | explicit micro-step counter + `model.no_sync()` on non-boundary micro-steps |
| `accelerator.autocast()` | step loop, `cam_inference` | `torch.autocast("cuda", dtype=…)` |
| `accelerator.clip_grad_norm_` | step loop | `scaler.unscale_(opt)` then `torch.nn.utils.clip_grad_norm_` |
| `accelerator.gather` / `gather_for_metrics((batch, output))` | metrics, early-stop, loss reduce | `dist.all_gather_object` (objects) / `dist.all_reduce` (tensors) |
| `accelerator.log({...}, step=)` + `MLflowTracker` + `trackers[0].store_init_configuration` | logging | direct `mlflow` client, rank-0 only (moves to a callback, §2) |
| `accelerator.save_state` / `load_state` (model+opt+sched+RNG) | `save_checkpoint`, resume | explicit `torch.save`/`load` of a state dict (§1.6) |
| `accelerator.unwrap_model` | eval, EMA, save | `model.module if isinstance(model, DDP) else model` |
| `accelerator.wait_for_everyone` | barriers | `dist.barrier()` |
| `accelerate.utils.set_seed` | `main`, `train` | local `seed_everything` (torch + numpy + random + cuda) |
| `accelerator.end_training` | end | `dist.destroy_process_group()` |

**Already pure `torch.distributed`** (no change needed, only the init contract):

- `avatar/data/` — `TabularDataset` / `EventSequenceDataset` shard the record
  stream by `dist.get_rank()` / `WORLD_SIZE`, do their own `dist.all_reduce` /
  `dist.barrier` for the filter cache, and expose `set_epoch()` / `shard`.
  *(Updated 2026-08-30: this section originally described
  `ShardTabularDataset` / `ShardEventSequenceDataset` in
  `avatar/data/dataset/`, and claimed `avatar/data/**` needed no change. See
  `data_redesign.md` — the module was restructured, the sequence dataset's
  sharding was rewritten, and `train.py`'s `shard_by_rank` guards did have to
  change.)*
- `avatar/utils/performance_metrics.py` — `_reduce()` is already
  `dist.all_reduce`.

So the data + perf layers only need `dist.init_process_group` to have run before
the dataset is built (it already handles the not-initialised case via env vars).

### Accelerate quirks the current code works around (delete on rewrite)

- `steps_before_evaluation // accelerator.num_processes`,
  `min_num_steps // accelerator.num_processes`,
  `num_training_steps // grad_accumulation_steps` — compensation for accelerate's
  `AcceleratedScheduler` stepping semantics. With an explicit scheduler `.step()`
  in the loop these divisions go away; step budgets become literal.
- The `if shard_by_rank: prepare(model, opt, sched)` **else** `prepare(…, dataloader)`
  branch — only needed because accelerate otherwise tries to shard the loader
  itself. Gone: we never hand the loader to anything.
- `evaluation()` re-`prepare`-ing the model mid-run.

---

## 1. Q1 — pure `torch.distributed`

### 1.1 Launch contract

`torchrun --standalone --nproc_per_node=<gpus> -m avatar.train +config=…`
(single process = `--nproc_per_node=1`, or run the module directly — guard on
`WORLD_SIZE` unset ⇒ world_size 1, no `init_process_group`).

`torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`.
Multi-node = same command per node with `--nnodes` / `--node_rank` / `--rdzv_*`.

The NCCL env block currently hard-coded at the top of `train.py`
(`NCCL_P2P_DISABLE`, timeouts, `CUDA_LAUNCH_BLOCKING=1` — note: that one
**serialises every CUDA call**, likely a debugging leftover that hurts throughput)
moves to a documented launcher script / config, not module import side-effects.

### 1.2 `DistEnv` helper (`avatar/train/dist.py`)

```python
@dataclass(frozen=True)
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool: return self.rank == 0
    @property
    def is_local_main(self) -> bool: return self.local_rank == 0
    @property
    def distributed(self) -> bool: return self.world_size > 1

    @classmethod
    def from_env(cls) -> "DistEnv":
        ws = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        lr = int(os.environ.get("LOCAL_RANK", 0))
        if ws > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(hours=2))
        dev = torch.device(f"cuda:{lr}" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        return cls(rank, lr, ws, dev)

    def barrier(self): 
        if self.distributed: dist.barrier()
    def all_reduce_mean(self, t): ...      # tensor -> global mean
    def gather_objects(self, obj) -> list: ...  # all_gather_object, returns [] off main if you want
```

Collectives used: `all_reduce` (SUM/MAX for scalars — loss, grad-norm,
early-stop flag, timings), `all_gather_object` (batch+output for
non-additive metrics), `barrier`. Nothing else.

### 1.3 Model wrapping

```python
model.to(env.device)
if env.distributed:
    model = DDP(model, device_ids=[env.local_rank],
                find_unused_parameters=cfg.ddp.find_unused_parameters,
                gradient_as_bucket_view=True)
```

`find_unused_parameters` comes straight from config (today it is buried in a
`DistributedDataParallelKwargs` handler — same value, simpler path).
`unwrap(model)` = `model.module if isinstance(model, DDP) else model`.

### 1.4 Mixed precision

```python
amp_dtype = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg.amp]
scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
...
with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
    out = model(**batch)
    loss = out.loss / grad_accum
scaler.scale(loss).backward()
```

bf16 needs no scaler; fp16 does. Config: `amp: no | fp16 | bf16`.

### 1.5 Gradient accumulation (no `accelerator.accumulate`)

```python
is_boundary = (micro_step + 1) % grad_accum == 0
ctx = model.no_sync() if (env.distributed and not is_boundary) else nullcontext()
with ctx:
    <forward + scaled backward>
if is_boundary:
    if clip_grad_norm: scaler.unscale_(opt); gnorm = clip_grad_norm_(params, clip)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    scheduler.step()
```

`no_sync()` skips the all-reduce on non-boundary micro-steps — the real win of
accumulation, which `accelerator.accumulate` also does but opaquely.

### 1.6 Checkpoint / resume (explicit)

One file per checkpoint, `torch.save`:

```python
{
  "model": unwrap(model).state_dict(),
  "optimizer": opt.state_dict(),
  "scheduler": scheduler.state_dict(),
  "scaler": scaler.state_dict(),
  "epoch": epoch, "global_step": step,
  "rng": {"torch": ..., "cuda": ..., "numpy": ..., "python": ...},
  "config_hash": ...,
}
```

Written by rank 0 only, `dist.barrier()` after. Resume restores all of it.
Replaces `accelerator.save_state/load_state` + the ad-hoc `model.bin` +
`best_models/<exp>/<run>/<step>/checkpoints/pytorch_model.bin` layout (the eval
path currently reaches into accelerate's checkpoint dir by hand — §train.py:603).

### 1.7 Evaluation

`evaluation()` loses the `distributed_evaluate` / re-`prepare` branch. Two modes:

- **gather-all (default, non-additive metrics — ROC-AUC etc.):** every rank runs
  its shard, `all_gather_object((batch, output))` per step → rank 0 feeds
  `metric.update`. Requires the eval dataset to shard (it does, via
  `shard_by_rank`) or a `DistributedSampler`.
- **reduce (additive — loss, counts):** `all_reduce(SUM)` of local sums + counts.

Rank 0 computes and returns `scores`; other ranks return `None`. Same as today,
minus accelerate.

### 1.8 MLflow

No `accelerate.tracking.MLflowTracker`. A `MLflowCallback` (§2) owns an `mlflow`
run on rank 0: `log_params` at start, `log_metrics(dict, step)` on log events,
`log_artifact` for checkpoints, `end_run` at the end. Off-main ranks: no-op.

### 1.9 Blast radius

- `avatar/train.py`, `avatar/train_utils.py`, `avatar/accelerate_utils.py`
  (deleted), `avatar/inference.py`, `avatar/cam_inference.py` — rewritten.
- `avatar/utils/init_modules.py` — `init_scheduler` drops the `// grad_accum`
  fudge; `init_optimizer` unchanged.
- Configs: `accelerator:` block → `distributed:` / `amp:` / `ddp:` keys. ~50
  train configs (`examples/`, `experiments/sbercampaign_pilot/`). The
  `accelerator:` → new mapping is mechanical; provide a shim that reads the old
  block for one release.
- `pyproject.toml` — drop `accelerate` dependency.
- `avatar/training_arguments.py` — `distributed_evaluate` field removed;
  `device_specific` (already dead) removed.
- **No change** to `avatar/data/**` (already pure torch.distributed), metrics,
  losses, pipelines. *(Updated 2026-08-30: `avatar/data/**` was in fact
  restructured first — see `data_redesign.md`. It is now pure
  `torch.distributed` and needs nothing further from this rewrite, but the
  `shard_by_rank` → `shard` rename already landed in `train.py`.)*
- Tests: **there are none for `train.py`** — the rewrite must add a
  `torchrun --nproc_per_node=2` smoke test on tiny synthetic data (1 epoch,
  assert loss decreases + checkpoint round-trips). This is the single biggest
  risk mitigation.

---

## 2. Q2 — callback system

### 2.1 Why

The `train()` function is a ~400-line loop that inlines: perf metrics, shard
metrics, profiler, EMA/SWA, early stopping, checkpoint rotation, MLflow logging,
grad-norm logging, per-step vs per-epoch eval, startup timing. Every new
cross-cutting feature grows it. A callback boundary makes the core loop ~80 lines
and each concern a testable unit.

### 2.2 Shape (HF-Trainer-like, deliberately)

```python
@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    max_steps: int | None = None
    log_history: list[dict] = field(default_factory=list)
    best_metric: float | None = None
    # scaling-relevant, filled by the loop:
    world_size: int = 1
    samples_seen: int = 0
    micro_step: int = 0

@dataclass
class TrainerControl:
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_evaluate: bool = False
    should_save: bool = False
    should_log: bool = False

class TrainerCallback:
    def on_train_begin(self, ctx): ...
    def on_epoch_begin(self, ctx): ...
    def on_batch_begin(self, ctx): ...          # after move_to_device
    def on_forward_end(self, ctx): ...          # ctx.output available
    def on_backward_end(self, ctx): ...         # ctx.loss, grads populated
    def on_optimizer_step(self, ctx): ...       # after opt.step (boundary only); ctx.grad_norm
    def on_step_end(self, ctx): ...             # one full (accumulated) step
    def on_evaluate(self, ctx): ...             # ctx.metrics
    def on_save(self, ctx): ...
    def on_epoch_end(self, ctx): ...
    def on_train_end(self, ctx): ...
    def on_log(self, ctx): ...
```

`ctx` (a `CallbackContext`) carries: `env` (DistEnv), `state`, `control`, `model`
(unwrapped), `optimizer`, `scheduler`, `batch`, `output`, `loss`, `grad_norm`,
`metrics`, `config`. Callbacks mutate `control` to steer the loop (early stop,
force eval/save/log). Only rank-0-relevant callbacks guard on `ctx.env.is_main`
internally — the handler calls every callback on every rank (some need the
collective, e.g. a throughput callback that all-reduces).

`CallbackHandler` iterates a list in order; order is config-defined.

### 2.3 Built-in callbacks (migrate today's inline logic)

| callback | replaces |
|---|---|
| `MLflowCallback` | `accelerator.log` + tracker; owns the run, logs params/metrics/artifacts on rank 0 |
| `CheckpointCallback` | `save_checkpoint` + `save_model` + rotation + "best" tracking; writes the §1.6 state dict |
| `EarlyStoppingCallback` | wraps the existing `EarlyStopping`; sets `control.should_training_stop` (all-reduced) |
| `EMACallback` (SWA) | `update_ema_weights` + `init_swa_model`; owns the `AveragedModel`, swaps it in for eval/save |
| `LRSchedulerCallback` *(optional)* | if we want scheduler stepping observable; otherwise the loop steps it |
| `GradNormCallback` | grad-norm accumulation + logging |
| `ProfilerCallback` | `torch.profiler` start/step/stop (rank 0) |
| `ThroughputCallback` | **scaling**: samples/s, tokens/s, step time, DDP wait time, MFU estimate; all-reduced across ranks; the "масштабирование" observability |
| `PerfMetricsCallback` | `SystemMetricsCollector` + `reduce_epoch_performance` + shard metrics |
| `ProgressBarCallback` | the `tqdm` bar (rank 0) |

Everything below the loop core becomes opt-in via config
(`callbacks: [ {_target_: avatar.train.MLflowCallback}, … ]`), with a sensible
default list when the key is absent.

### 2.4 "Для масштабирования" — two readings, both served

- **Extensibility** (primary): the callback seam is what lets the trainer grow
  (new logging, new schedules, new checkpoint policies) without editing the loop.
- **Distributed scaling** (secondary): `ThroughputCallback` +
  `PerfMetricsCallback` give per-rank / aggregate throughput, comm overhead, and
  load-imbalance signals — the numbers you need when scaling GPU/node count.
  A `BatchSizeRampCallback` / `GradAccumScheduleCallback` (progressive batch
  size) also fits here if wanted later.

### 2.5 What stays in the core loop (not a callback)

forward / backward / optimizer step / grad-accum boundary / AMP / DDP no_sync /
scheduler step / `move_to_device`. These are the loop; callbacks observe and
steer, they don't run the math.

---

## 3. Q3 — loss inside the model, or a separate `nn.Module`?

### 3.1 Where it is today (inconsistent)

- **Inside `forward`, hard-coded:** `TabularClassification` builds
  `nn.BCEWithLogitsLoss()` / `CrossEntropyLoss()` / `MSELoss()` in `__init__`
  from `(num_classes, task_type)` and calls it in `forward`; returns
  `TabularOutput(logits, loss)`.
- **Inside `forward`, injected:** `SLearner` / `SupervisedLearner` take
  `loss_fn: nn.Module` from config and call it in `forward`.
- **Inside `forward`, bespoke multi-term:** `NextKTokensPrediction` computes a
  `dict` of per-head losses + a `dict` of `num_items`, returns
  `SequenceOutput(loss=None, losses=…, num_items=…)`; the trainer's
  `calculate_output_loss` then does the **cross-rank token-weighted average**
  (`loss_k * world_size / sum_ranks(num_items_k)`).
- Standalone loss `nn.Module`s already exist in `avatar/losses/`
  (`ContrastiveLoss`, `DirectUpliftLoss`, `KLDLoss`, `L1RegularizationLoss`,
  multi-task) — but they are consumed *inside* pipelines, never by the trainer.

The trainer only ever reads `output.loss` **or** `output.losses`+`output.num_items`.
It is already loss-agnostic; the mess is *inside* the pipelines.

### 3.2 Recommendation: **separate `nn.Module`, owned by the task-pipeline, never by the trainer**

```
trainer  ──calls──▶  task-pipeline.forward(batch)  ──►  output (has .loss)
                          │
                          ├─ backbone(...)              # avatar.nn.*  (no loss, ever)
                          └─ self.loss(output, batch)   # avatar.losses.*  (nn.Module)
```

Rules:

1. **`avatar/nn/**` never computes a loss.** Encoders/backbones return
   representations (`BaseTabularOutput`, `BaseSequenceOutput`). This is already
   true after the `sequential` / `tabular` refactors — keep it.
2. **Loss is an `nn.Module` in `avatar/losses/`**, injected into the
   task-pipeline via config (the `SLearner` pattern, made universal). It may take
   the model / its submodules as a constructor arg when it needs internals
   (e.g. `L1RegularizationLoss(apply_substr="embed")` walks `named_parameters`;
   MoE aux-loss needs `router_logits` — which is already on the output object).
3. **The task-pipeline `forward` orchestrates**: run the backbone, call
   `self.loss(...)`, pack the result into the output dataclass. The pipeline is
   the only place that knows both "what the model produced" and "what the targets
   mean".
4. **Standard loss contract:**

   ```python
   class Loss(nn.Module):
       def forward(self, output, batch, *, model=None) -> LossOutput: ...

   @dataclass
   class LossOutput:
       loss: torch.Tensor                       # scalar the trainer reads
       components: dict[str, torch.Tensor] = {} # for logging (bce=…, l1=…, aux=…)
       num_items: dict[str, torch.Tensor] | None = None  # for token-weighted DDP reduce
   ```

   The task-pipeline copies `LossOutput.loss → output.loss`,
   `.components → output.loss_components`, `.num_items → output.num_items`.
   `calculate_output_loss` in the trainer keeps working unchanged (reads
   `output.loss` or reduces `output.num_items`), just moves into
   `avatar/train/` and drops the `accelerator` arg for `DistEnv`.

### 3.3 Why not "loss inside the model"

- The model must stay usable for **inference with no targets** — a loss inside
  `forward` means dead code / `if targets is not None` branches on every path
  (exactly what `TabularClassification.forward` has now).
- **Swappability**: contrastive vs BCE vs focal is a config decision, not a
  code change. Half the pipelines already inject `loss_fn`; the other half
  hard-code it — unify on injection.
- **Composition**: `total = task_loss + λ₁·l1 + λ₂·contrastive + λ₃·moe_aux` is a
  `CompositeLoss(nn.Module)` that wraps sub-losses — clean only if losses are
  modules with a shared contract.
- **Multi-task**: `NextKTokensPrediction`'s per-head dict + `num_items` becomes a
  `MultiHeadLoss` module returning `LossOutput(num_items=…)`; the distributed
  token-weighting stays in one place.

### 3.4 Why not "loss as a thing the trainer holds"

- The trainer would need to know the target semantics of every task (uplift
  double-forward, next-k shifting, group masking). That knowledge belongs in the
  pipeline. Keeping the trainer's contract at "call `model(**batch)`, read
  `output.loss`" is what makes the trainer reusable and the callback system
  clean.

### 3.5 Migration

- Add `LossOutput` to `avatar/outputs.py`; add `avatar/losses/base.py::Loss`.
- Wrap the existing `nn.*Loss` uses: `TabularClassification` /
  `SequenceClassification` gain a `loss: Loss` ctor param (default =
  `ClassificationLoss(num_classes, task_type)` built from a factory, so old
  configs that pass `num_classes`/`task_type` still work).
- `NextKTokensPrediction`'s loss logic → `NextKTokensLoss(Loss)`.
- Trainer: `calculate_output_loss` → `avatar/train/loss_reduce.py`, `DistEnv`
  instead of `accelerator`.
- Configs: optional `loss:` block under each `model:`; absent ⇒ factory default.

---

## 4. Proposed file layout

```
avatar/train/
├── __init__.py
├── __main__.py          # torchrun entrypoint (hydra), was avatar/train.py:main
├── loop.py              # Trainer: the ~80-line core loop
├── dist.py              # DistEnv + collectives
├── state.py             # TrainerState, TrainerControl, CallbackContext
├── callbacks/
│   ├── base.py          # TrainerCallback, CallbackHandler
│   ├── mlflow.py
│   ├── checkpoint.py
│   ├── early_stopping.py
│   ├── ema.py
│   ├── profiler.py
│   ├── throughput.py    # scaling observability
│   ├── perf_metrics.py
│   └── progress.py
├── loss_reduce.py       # token-weighted cross-rank loss (ex calculate_output_loss)
└── evaluate.py          # evaluation() without accelerate

avatar/losses/base.py    # Loss, LossOutput
```

`avatar/inference.py` + `avatar/cam_inference.py` collapse into
`avatar/train/evaluate.py` + a thin `avatar/infer.py` entrypoint (they are 80%
the eval loop already).

---

## 5. Risks

- **No existing tests for the training loop.** A 2-rank `torchrun` smoke test on
  synthetic data is a prerequisite, not a follow-up.
- **Numerical parity**: DDP grad-averaging vs accelerate's, scheduler step
  counts (the `// num_processes` fudges must be removed *and* verified against a
  known-good run — LR schedule shape will change if done wrong).
- **fp16 + `clip_grad_norm`**: must `scaler.unscale_` before clipping; easy to
  get wrong and silently train worse.
- **Checkpoint format change** breaks resume from existing accelerate
  checkpoints — provide a one-off converter or accept a clean break.
- **~50 train configs** need the `accelerator:` → `distributed:`/`amp:`/`ddp:`
  rewrite; shim for one release.
- **`cam_inference.py`** has campaign-specific batch mutation logic — must be
  preserved when folding into the shared eval loop.
- Scope: this is 3 refactors (distributed, callbacks, loss). Recommend landing
  in that order, each behind the smoke test, `accelerate` removed only after all
  three are green.

---

## 6. What shipped

Landed in the order §5 recommends, each behind the smoke test, with
`accelerate` removed only once all three were green.

### 6.1 Layout as built

```
avatar/train/
├── __init__.py          # flat re-export (40 names)
├── __main__.py          # torchrun entrypoint (hydra), was avatar/train.py:main
├── loop.py              # Trainer: the core loop
├── dist.py              # DistEnv + collectives + seed_everything + unwrap_model
├── config.py            # RunConfig; reads the new keys or the legacy accelerator: block
├── state.py             # TrainerState, TrainerControl, CallbackContext
├── checkpoint.py        # explicit save/load/rotate
├── early_stopping.py    # EarlyStopping, moved from train_utils
├── evaluate.py          # evaluate() + predict(), no accelerate
├── factory.py           # build_callbacks: config -> callback list
├── loss_reduce.py       # token-weighted cross-rank loss
├── utils.py             # move_to_device, wrap_metrics, log params, loader checks
└── callbacks/
    ├── base.py mlflow.py checkpoint.py early_stopping.py ema.py
    ├── metrics.py perf_metrics.py profiler.py progress.py
    └── throughput.py train_stats.py

avatar/losses/base.py             # Loss, CompositeLoss
avatar/losses/classification.py   # ClassificationLoss, build_task_loss_fn
avatar/losses/next_k_tokens.py    # NextKTokensLoss, HeadPrediction
avatar/infer.py                   # inference entrypoint
```

`avatar/accelerate_utils.py` is deleted. `avatar/train_utils.py` and
`avatar/inference.py` remain as deprecation shims.

### 6.2 Deviations from the proposal

- **`init_scheduler`'s `// grad_accumulation_steps` is kept**, contrary to
  §1.9. It is not an accelerate fudge: the loop steps the scheduler once per
  accumulation boundary, so `epochs * batches / grad_accum` is the correct step
  budget. The two divisions that *were* fudges —
  `steps_before_evaluation // num_processes` and EMA
  `min_num_steps // num_processes` — are gone, and those budgets are now
  literal.
- **`GradNormCallback` merged into `TrainStatsCallback`**, which reports the
  epoch means of loss, learning rate and gradient norm together. Three
  single-number callbacks accumulating over the same loop was more machinery
  than the job needs.
- **`TrainMetricsCallback` added** (not in §2.3). Training-metric updates need
  exactly `batch` and `output`, which `on_forward_end` already carries, so the
  update moved out of the loop with everything else.
- **`ProgressBarCallback` advances the bar from `on_batch_begin`** rather than
  wrapping the dataloader; that keeps the loop iterating a plain iterator and
  the bar an ordinary, removable callback.
- **Checkpoint saves are vetoed, not requested.** The loop sets
  `control.should_save` before firing `on_evaluate`, and
  `EarlyStoppingCallback` clears it when the metric did not improve. This
  reproduces the old "only checkpoint the best" behaviour and is the one place
  where callback order matters (early stopping before the checkpointer).
- **Rank 0's verdict is broadcast, not all-reduced.** Only rank 0 holds the
  evaluation scores, so `should_training_stop` and `should_save` go through
  `DistEnv.broadcast_flag`. An `any`-reduction would have let a vetoed save
  happen anyway, because the other ranks never saw the metric.
- **`evaluate()` picks its reduction from whether the dataset shards**, rather
  than from a `distributed_evaluate` flag (which no config ever set and which
  is now removed from `TrainingArguments`). A replicated dataset would count
  every sample `world_size` times if gathered, so every rank still runs the
  forward pass — nobody blocks at a collective the others never reach — but
  only rank 0 scores.
- **`NextKTokensLoss` does not own the prediction heads.** §3.5 says the loss
  logic moves wholesale; the heads carry parameters, so moving them would
  rename every `lm_heads.*` key in existing checkpoints. The heads stay on the
  pipeline and hand their logits to the loss, which owns label shifting, the
  criteria and the per-head weighting. The horizon loop also stays on the
  pipeline: the event-id embedding feeds back into the hidden state between
  horizons.
- **The already-injected pipelines were left alone.** `SLearner`,
  `SupervisedLearner`, `MMoE` and the multi-task pipelines already take
  `loss_fn` from config, which is what §3.2 asks for. Rewrapping them in the
  `Loss`/`LossOutput` contract would be churn without a behaviour change.
- **One pre-existing asymmetry is preserved deliberately**: the timedelta head
  divides by its horizon coefficient in eval as well as in training, while
  every other head applies the coefficient only while training. It is carried
  across as `HeadPrediction.scale_by_coef_in_eval` and pinned by a test rather
  than silently normalised.
- **NCCL env tuning is not set at import.** The old block wrote eight
  `os.environ` keys as an import side effect, one of them
  `CUDA_LAUNCH_BLOCKING=1`, which serialises every CUDA call. Set what a run
  needs in its launcher.

### 6.3 Config migration

60 config files moved from `accelerator:` to `distributed:` / `amp:` / `ddp:` /
`compile:`. The whole surface was five distinct blocks, so the mapping is
mechanical:

| old | new |
|---|---|
| `dataloader_config.dispatch_batches` | *dropped* — accelerate's own loader wrapping |
| `gradient_accumulation_steps` | `distributed.gradient_accumulation_steps` |
| `kwargs_handlers[DistributedDataParallelKwargs].*` | `ddp.*` |
| `dynamo_plugin.backend` | `compile` |
| `mixed_precision` | `amp` |

`resolve_run_config` still reads the old block, with a `DeprecationWarning`,
so out-of-repo configs keep working. New keys win when both are present.
`avatar.train_utils.EarlyStopping` targets were repointed to
`avatar.train.EarlyStopping` in 7 configs.

### 6.4 Test coverage

The loop had **no tests at all**; §5 called that the biggest risk. Added:

- `tests/train/test_trainer_loop.py` (11) — loss falls over epochs, step/batch/
  sample counts, gradient accumulation, callback event ordering, per-epoch and
  per-step evaluation, checkpoint write and restore, resume, early stopping
  (both halting and vetoing a save), and a callback stopping training.
- `tests/train/test_train_config.py` (10) — new keys, legacy block, precedence,
  AMP validation.
- `tests/train/test_train_callbacks.py` (17) — dispatch and ordering, each
  callback in isolation, MLflow param sanitising, loss reduction.
- `tests/pipeline/test_losses.py` (15) — criterion selection, `LossOutput`
  shape, L1 penalty, `CompositeLoss`, and `NextKTokensLoss` weighting.
- `tests/train/test_distributed_training.py` (8, `slow`) — real `torchrun`
  processes on gloo with the GPUs hidden: 1- and 2-rank, a simulated 2-node
  4-rank job, gradient accumulation, clipping, DataLoader workers, and bf16.
  Each asserts that the loss falls, that every rank ends on identical weights
  and step counts, and that checkpoints round-trip.

Hiding the GPUs is what makes this runnable here: gloo collectives run on CPU,
so several ranks — and several nodes — fit on a single-GPU host, which NCCL
cannot do.

### 6.5 Follow-ups not done

- Numerical parity against a known-good accelerate run (§5). The LR schedule
  shape is unchanged by construction, but nothing has been trained end to end
  on real data yet.
- A converter for existing accelerate checkpoints; the format change is a clean
  break for now.
- Removing the `avatar.train_utils` / `avatar.inference` shims once out-of-repo
  configs and scripts have moved.
