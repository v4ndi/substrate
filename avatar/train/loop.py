"""The training loop.

Everything that is not forward / backward / optimizer step / accumulation
boundary / AMP / DDP ``no_sync`` / scheduler step / ``move_to_device`` lives in
a callback. See :mod:`avatar.train.callbacks`.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from avatar.metrics import BaseMetric
from avatar.train.callbacks.base import CallbackHandler, TrainerCallback
from avatar.train.checkpoint import load_checkpoint
from avatar.train.config import RunConfig
from avatar.train.dist import DistEnv, seed_everything, unwrap_model
from avatar.train.evaluate import evaluate
from avatar.train.loss_reduce import calculate_output_loss
from avatar.train.state import CallbackContext, TrainerControl, TrainerState
from avatar.train.utils import apply_basic_loader_checks, move_to_device, prefix_metrics
from avatar.training_arguments import TrainingArguments
from avatar.utils.performance_metrics import infer_batch_size

AMP_DTYPES: dict[str, torch.dtype | None] = {
    "no": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class Trainer:
    """Run a training job on plain ``torch.distributed``.

    Args:
        model: Module to train. Wrapped in DDP here when the run is distributed.
        optimizer: Optimizer over ``model``'s parameters.
        scheduler: LR scheduler, stepped once per accumulation boundary.
        train_dataloader: Training loader. Its dataset is expected to shard
            itself by rank; a replicated dataset trains on duplicated data.
        training_arguments: Epoch budget, seed, clipping, resume paths.
        run_config: Distributed / AMP / DDP settings.
        env: Distributed environment. Built from the process environment when
            omitted.
        callbacks: Observers, in firing order.
        valid_dataloader: Optional validation loader.
        test_dataloader: Optional test loader, evaluated once at the end.
        valid_metrics: Metrics for validation.
        test_metrics: Metrics for the final test pass.
        checkpoint_dir: Where checkpoints are written; also where the final test
            pass looks for the best weights.
        config: The full Hydra config, passed through to callbacks.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_dataloader: DataLoader,
        training_arguments: TrainingArguments,
        run_config: RunConfig | None = None,
        env: DistEnv | None = None,
        callbacks: list[TrainerCallback] | None = None,
        valid_dataloader: DataLoader | None = None,
        test_dataloader: DataLoader | None = None,
        valid_metrics: BaseMetric | list[BaseMetric] | None = None,
        test_metrics: BaseMetric | list[BaseMetric] | None = None,
        checkpoint_dir: str = "checkpoints",
        config: Any = None,
    ):
        apply_basic_loader_checks(train_dataloader, valid_dataloader, test_dataloader)

        self.run_config = run_config or RunConfig()
        self.env = env or DistEnv.from_env(
            backend=self.run_config.distributed.backend,
            timeout_sec=self.run_config.distributed.timeout_sec,
        )
        self.training_arguments = training_arguments
        seed_everything(training_arguments.seed, rank=self.env.rank)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.valid_metrics = valid_metrics
        self.test_metrics = test_metrics
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.callbacks = CallbackHandler(callbacks)

        self.grad_accum = max(1, self.run_config.gradient_accumulation_steps)
        self.amp_dtype = AMP_DTYPES[self.run_config.amp]
        # Only fp16 needs loss scaling, and only CUDA implements it.
        self.scaler = torch.amp.GradScaler(
            self.env.device.type,
            enabled=self.amp_dtype is torch.float16 and self.env.device.type == "cuda",
        )

        if training_arguments.model_state is not None:
            model.load_state_dict(
                torch.load(training_arguments.model_state, map_location="cpu")
            )

        model = model.to(self.env.device)
        if self.run_config.compile:
            model = torch.compile(model, backend=self.run_config.compile)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = self._wrap(model)

        self.state = TrainerState(
            num_epochs=training_arguments.num_epochs,
            world_size=self.env.world_size,
        )
        self.control = TrainerControl()
        self.ctx = CallbackContext(
            env=self.env,
            state=self.state,
            control=self.control,
            run_config=self.run_config,
            config=config,
            model=unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_dir=checkpoint_dir,
            log=self.log,
        )
        self.ctx.extra.update({
            "train_dataloader": train_dataloader,
            "batch_size": train_dataloader.batch_size,
            "scaler": self.scaler,
        })
        self._logging = False

        if training_arguments.checkpoint_state is not None:
            load_checkpoint(
                training_arguments.checkpoint_state,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                state=self.state,
                map_location=self.env.device,
            )
        self.env.barrier()

    # -- setup ---------------------------------------------------------------

    def _wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.env.distributed:
            return model
        ddp = self.run_config.ddp
        return DDP(
            model,
            device_ids=[self.env.local_rank]
            if self.env.device.type == "cuda"
            else None,
            find_unused_parameters=ddp.find_unused_parameters,
            gradient_as_bucket_view=ddp.gradient_as_bucket_view,
            broadcast_buffers=ddp.broadcast_buffers,
            static_graph=ddp.static_graph,
        )

    # -- logging -------------------------------------------------------------

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        """Publish metrics to the logging callbacks."""
        if self._logging or not values:
            return
        self._logging = True
        try:
            self.ctx.logs = dict(values)
            self.ctx.log_step = self.state.global_step if step is None else int(step)
            self.state.log_history.append({"step": self.ctx.log_step, **self.ctx.logs})
            self.callbacks.fire("on_log", self.ctx)
        finally:
            self.ctx.logs = {}
            self._logging = False

    # -- the loop ------------------------------------------------------------

    def train(self) -> dict[str, Any] | None:
        """Run every epoch, then the final test pass. Returns the test scores."""
        state, control, env = self.state, self.control, self.env
        start_epoch = self.training_arguments.start_epoch or 0
        state.epoch = start_epoch

        self.callbacks.fire("on_train_begin", self.ctx)
        self.log(
            {
                "world_size": env.world_size,
                "total_number_model_parameters": sum(
                    p.numel() for p in self.model.parameters()
                ),
                "gradient_accumulation_steps": self.grad_accum,
            },
            step=0,
        )

        for epoch in range(start_epoch, self.training_arguments.num_epochs):
            state.epoch = epoch
            self.ctx.extra["first_epoch"] = epoch == start_epoch
            dataset = self.train_dataloader.dataset
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

            self.model.train()
            control.should_epoch_stop = False
            self.callbacks.fire("on_epoch_begin", self.ctx)
            self._run_epoch()
            self.callbacks.fire("on_epoch_end", self.ctx)

            if self.valid_dataloader is not None and self._evaluates_per_epoch():
                self._evaluate(self.valid_dataloader, self.valid_metrics, step=epoch)
            env.barrier()
            if control.should_training_stop:
                break

        self.callbacks.fire("on_train_end", self.ctx)
        scores = self._final_test()
        env.barrier()
        return scores

    def _evaluates_per_epoch(self) -> bool:
        return self.training_arguments.steps_before_evaluation is None

    def _run_epoch(self) -> None:
        state, control = self.state, self.control
        state.epoch_step = 0
        state.epoch_batches = 0
        state.epoch_samples = 0
        state.micro_step = 0
        fallback_batch_size = self.train_dataloader.batch_size or 0

        for batch in self.train_dataloader:
            batch = move_to_device(batch, self.env.device)
            self.ctx.batch = batch
            state.micro_step += 1
            state.epoch_batches += 1
            # Counted once here so callbacks do not each re-derive it.
            samples = infer_batch_size(batch)
            samples = fallback_batch_size if samples is None else int(samples)
            state.epoch_samples += samples
            state.samples_seen += samples
            is_boundary = state.micro_step % self.grad_accum == 0
            state.learning_rate = self.optimizer.param_groups[0]["lr"]
            self.callbacks.fire("on_batch_begin", self.ctx)

            # Skipping the gradient all-reduce on non-boundary micro-steps is
            # the entire point of accumulation.
            sync = (
                self.model.no_sync()
                if self.env.distributed and not is_boundary
                else nullcontext()
            )
            with sync:
                with self._autocast():
                    output = self.model(**batch)
                    loss = calculate_output_loss(output, self.env)
                self.ctx.output, self.ctx.loss = output, loss
                self.callbacks.fire("on_forward_end", self.ctx)

                if torch.is_tensor(loss) and loss.requires_grad:
                    self.scaler.scale(loss / self.grad_accum).backward()
                self.callbacks.fire("on_backward_end", self.ctx)

            if is_boundary:
                self._optimizer_step()
                if control.should_evaluate or self._evaluates_per_step():
                    self._evaluate(
                        self.valid_dataloader,
                        self.valid_metrics,
                        step=state.global_step,
                    )
            if control.should_epoch_stop or control.should_training_stop:
                break

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(self.env.device.type, dtype=self.amp_dtype)

    def _optimizer_step(self) -> None:
        state = self.state
        grad_norm = None
        clip = self.training_arguments.clip_grad_norm
        if clip is not None:
            # Gradients are still scaled at this point; clipping a scaled
            # gradient would clip to the wrong norm.
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.ctx.grad_norm = grad_norm

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        state.global_step += 1
        state.epoch_step += 1
        self.callbacks.fire("on_optimizer_step", self.ctx)
        self.callbacks.fire("on_step_end", self.ctx)

    def _evaluates_per_step(self) -> bool:
        every = self.training_arguments.steps_before_evaluation
        return (
            self.valid_dataloader is not None
            and every is not None
            and self.state.global_step % every == 0
        )

    # -- evaluation ----------------------------------------------------------

    def _evaluate(
        self,
        dataloader: DataLoader | None,
        metrics: BaseMetric | list[BaseMetric] | None,
        step: int,
        prefix: str = "valid_",
    ) -> dict[str, Any] | None:
        if dataloader is None:
            return None
        env, control = self.env, self.control
        started = time.perf_counter()
        model = self.ctx.extra.get("eval_model") or self.model
        scores = evaluate(model, dataloader, env, metrics)

        self.ctx.metrics = scores or {}
        # Save by default and let a callback veto: that is how "keep only the
        # best checkpoint" survives the move out of the loop.
        control.should_save = True
        self.callbacks.fire("on_evaluate", self.ctx)

        # Only rank 0 saw the scores, so its verdict is the one that counts.
        control.should_training_stop = env.broadcast_flag(control.should_training_stop)
        control.should_save = env.broadcast_flag(control.should_save)

        if scores:
            self.log(prefix_metrics(scores, prefix), step=step)
        self.log({"eval/wall_sec": time.perf_counter() - started}, step=step)

        if control.should_save:
            self.callbacks.fire("on_save", self.ctx)
        env.barrier()
        control.should_evaluate = False
        self.model.train()
        return scores

    def _final_test(self) -> dict[str, Any] | None:
        """Load the best saved weights and score the test set."""
        if self.test_dataloader is None:
            return None
        from avatar.train.checkpoint import latest_checkpoint_step, model_path

        best_step = latest_checkpoint_step(self.checkpoint_dir)
        if best_step is not None:
            weights = model_path(self.checkpoint_dir, best_step)
            unwrap_model(self.model).load_state_dict(
                torch.load(weights, map_location=self.env.device, weights_only=False)
            )
        scores = evaluate(
            self.model,
            self.test_dataloader,
            self.env,
            self.test_metrics,
            description="Test step",
        )
        if scores:
            self.env.print(scores)
            self.log(prefix_metrics(scores, "test_"), step=self.state.epoch)
        return scores
