"""Binary-classification CatBoost and XGBoost adapter."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from avatar.automl.backends.boosting.base import BaseBoostingBackend
from avatar.automl.exceptions import MissingDependencyError, UnsupportedBackendError


class BinaryBoostingBackend(BaseBoostingBackend):
    """Adapt CatBoost or XGBoost to binary positive-class probabilities.

    Args:
        engine: Native engine name, ``catboost`` or ``xgboost``.
        params: Parameters passed to the native classifier.
        random_state: Reproducibility seed.
        device: Runtime ``cpu`` or ``gpu`` device.
        verbose: Native training logging setting.
    """

    task_name: ClassVar[str] = "binary"

    def _make_model(self):
        """Construct the configured native classifier."""
        if self.device not in {"cpu", "gpu"}:
            msg = f"Unsupported local boosting device: {self.device!r}"
            raise UnsupportedBackendError(msg)
        if self.engine == "catboost":
            params = self._model_params_with_defaults()
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                msg = "CatBoost engine requires the 'catboost' package"
                raise MissingDependencyError(msg) from exc
            defaults = {
                "random_seed": self.random_state,
                "allow_writing_files": False,
                "task_type": "GPU" if self.device == "gpu" else "CPU",
            }
            return CatBoostClassifier(**(defaults | params))
        elif self.engine == "xgboost":
            params = self._model_params_with_defaults()
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                msg = "XGBoost engine requires the 'xgboost' package"
                raise MissingDependencyError(msg) from exc
            defaults = {
                "random_state": self.random_state,
                "verbosity": 1 if self.verbose else 0,
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": "cuda" if self.device == "gpu" else "cpu",
            }
            effective_params = defaults | params
            effective_params["enable_categorical"] = True
            return XGBClassifier(**effective_params)
        else:
            msg = f"Unsupported boosting engine: {self.engine!r}"
            raise UnsupportedBackendError(msg)

    def predict_prepared_score(self, features: Any) -> np.ndarray:
        """Return positive-class probabilities for engine-ready features."""
        kwargs = {"task_type": "CPU"} if self.engine == "catboost" else {}
        probabilities = np.asarray(self.model.predict_proba(features, **kwargs))
        return probabilities[:, 1]
