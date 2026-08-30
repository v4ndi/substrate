"""MLflow run ownership, replacing ``accelerate.tracking.MLflowTracker``."""

from __future__ import annotations

import warnings
from typing import Any

from avatar.train.callbacks.base import TrainerCallback
from avatar.train.state import CallbackContext

# MLflow rejects params longer than this and keys outside a small charset.
MAX_PARAM_LENGTH = 500
_ALLOWED_KEY_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./ "
)


def sanitize_param_key(key: str) -> str:
    return "".join(char if char in _ALLOWED_KEY_CHARS else "_" for char in str(key))


def sanitize_params(params: dict[str, Any]) -> dict[str, str]:
    """Drop what MLflow cannot store and stringify the rest."""
    prepared: dict[str, str] = {}
    for key, value in params.items():
        text = str(value)
        if len(text) > MAX_PARAM_LENGTH:
            warnings.warn(
                f"Skipping MLflow param {key!r}: value is {len(text)} characters, "
                f"the limit is {MAX_PARAM_LENGTH}.",
                stacklevel=2,
            )
            continue
        prepared[sanitize_param_key(key)] = text
    return prepared


class MLflowCallback(TrainerCallback):
    """Own an MLflow run on rank 0; a no-op everywhere else.

    Args:
        experiment_name: MLflow experiment to log into.
        run_name: Name of the run.
        tracking_uri: Tracking server URI; falls back to the ambient MLflow config.
        params: Parameters to record once at the start of training.
        enabled: Set False to disable logging without removing the callback.
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        params: dict[str, Any] | None = None,
        enabled: bool = True,
        **run_kwargs: Any,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.params = params or {}
        self.enabled = enabled
        self.run_kwargs = run_kwargs
        self._active = False

    def _mlflow(self):
        import mlflow

        return mlflow

    def on_train_begin(self, ctx: CallbackContext) -> None:
        if not (self.enabled and ctx.env.is_main):
            return
        mlflow = self._mlflow()
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name, **self.run_kwargs)
        self._active = True
        params = dict(self.params)
        params.setdefault("world_size", ctx.env.world_size)
        mlflow.log_params(sanitize_params(params))

    def on_log(self, ctx: CallbackContext) -> None:
        if not (self._active and ctx.logs):
            return
        numeric = {
            sanitize_param_key(key): float(value)
            for key, value in ctx.logs.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        if numeric:
            self._mlflow().log_metrics(numeric, step=int(ctx.log_step))

    def log_artifact(self, path: str) -> None:
        if self._active:
            self._mlflow().log_artifact(path)

    def on_train_end(self, ctx: CallbackContext) -> None:
        if self._active:
            self._mlflow().end_run()
            self._active = False
