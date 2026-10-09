"""MLflow run ownership, replacing ``accelerate.tracking.MLflowTracker``."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from fmlib.train.callbacks.base import TrainerCallback
from fmlib.train.state import CallbackContext

# MLflow rejects params longer than this and keys outside a small charset.
MAX_PARAM_LENGTH = 500
_ALLOWED_KEY_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./ "
)
_SECRET_KEY_PARTS = ("password", "secret", "token", "api_key")


def _is_secret_key(key: str) -> bool:
    return any(word in str(key).lower() for word in _SECRET_KEY_PARTS)


def sanitize_param_key(key: str) -> str:
    """Replace characters MLflow rejects in a param or metric key with ``_``."""
    return "".join(char if char in _ALLOWED_KEY_CHARS else "_" for char in str(key))


def sanitize_params(params: dict[str, Any]) -> dict[str, str]:
    """Drop what MLflow cannot store and stringify the rest."""
    prepared: dict[str, str] = {}
    for key, value in params.items():
        if _is_secret_key(key):
            continue
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


def _redact_config(value: Any) -> Any:
    """Keep credentials out of the configuration artifact."""
    if isinstance(value, Mapping):
        return {
            key: "<redacted>"
            if _is_secret_key(key)
            else _redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


class MLflowCallback(TrainerCallback):
    """Own an MLflow run on rank 0; a no-op everywhere else.

    Args:
        experiment_name: MLflow experiment to log into.
        run_name: Name of the run.
        tracking_uri: Tracking server URI; falls back to the ambient MLflow config.
        params: Parameters to record once at the start of training.
        enabled: Set False to disable logging without removing the callback.
        server_cert_path: CA bundle used to verify the tracking server certificate.
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        params: dict[str, Any] | None = None,
        enabled: bool = True,
        server_cert_path: str | None = None,
        **run_kwargs: Any,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.server_cert_path = server_cert_path
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
        if self.server_cert_path:
            cert_path = Path(self.server_cert_path).expanduser().resolve()
            if not cert_path.is_file():
                raise FileNotFoundError(
                    f"MLflow server CA certificate not found: {cert_path}"
                )
            os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"] = str(cert_path)
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
        if ctx.config is not None:
            config = OmegaConf.to_container(OmegaConf.create(ctx.config), resolve=True)
            config_text = yaml.safe_dump(
                _redact_config(config), allow_unicode=True, sort_keys=False
            )
            mlflow.log_text(
                config_text,
                "config.yaml",
            )

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

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self._active:
            self._mlflow().log_artifact(str(path), artifact_path=artifact_path)

    def on_after_save(self, ctx: CallbackContext) -> None:
        if not (self._active and ctx.env.is_main):
            return
        for path in ctx.extra.get("saved_artifacts", ()):
            self.log_artifact(
                path, artifact_path=f"checkpoints/{ctx.state.global_step}"
            )

    def on_after_train(self, ctx: CallbackContext) -> None:
        if not self._active:
            return
        for path in ctx.extra.get("profiler_traces", ()):
            self.log_artifact(
                path, artifact_path=f"profiler/{Path(path).parent.name}"
            )
        self._mlflow().end_run(status="FINISHED")
        self._active = False
