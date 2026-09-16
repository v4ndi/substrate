"""Multiclass CatBoost and XGBoost probability adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from avatar.automl.backends.boosting.base import BaseBoostingBackend
from avatar.automl.exceptions import (
    ArtifactIntegrityError,
    MissingDependencyError,
    UnsupportedBackendError,
)


@dataclass
class MulticlassBoostingBackend(BaseBoostingBackend):
    """Adapt CatBoost or XGBoost to stable multiclass probabilities.

    Args:
        engine: Native engine name, ``catboost`` or ``xgboost``.
        params: Parameters passed to the native classifier.
        random_state: Reproducibility seed.
        device: Runtime ``cpu`` or ``gpu`` device.
        verbose: Native training logging setting.
        num_classes: Number of encoded target classes.
    """

    task_name: ClassVar[str] = "multiclass"

    num_classes: int = 0
    estimator_class_order: tuple[int, ...] = field(default_factory=tuple)

    def _make_model(self):
        """Construct a native multiclass probability estimator."""
        if self.device not in {"cpu", "gpu"}:
            msg = f"Unsupported local boosting device: {self.device!r}"
            raise UnsupportedBackendError(msg)
        if self.num_classes < 3:
            msg = (
                f"Multiclass backend requires num_classes >= 3; got {self.num_classes}"
            )
            raise ArtifactIntegrityError(msg)
        if self.engine == "catboost":
            params = self._model_params_with_defaults(excluded=("boost_from_average",))
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                msg = "CatBoost engine requires the 'catboost' package"
                raise MissingDependencyError(msg) from exc
            defaults = {
                "random_seed": self.random_state,
                "allow_writing_files": False,
                "task_type": "GPU" if self.device == "gpu" else "CPU",
                "loss_function": "MultiClass",
                "eval_metric": "MultiClass",
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
                "objective": "multi:softprob",
                "num_class": self.num_classes,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "device": "cuda" if self.device == "gpu" else "cpu",
            }
            effective_params = defaults | params
            effective_params["enable_categorical"] = True
            return XGBClassifier(**effective_params)
        else:
            msg = f"Unsupported boosting engine: {self.engine!r}"
            raise UnsupportedBackendError(msg)

    def _after_fit(self) -> None:
        classes = np.asarray(
            getattr(self.model, "classes_", np.arange(self.num_classes))
        ).reshape(-1)
        self.estimator_class_order = tuple(int(value) for value in classes)

    def predict_prepared_score(self, features: Any) -> np.ndarray:
        """Return a finite normalized matrix aligned to encoded classes ``0..K-1``."""
        kwargs = {"task_type": "CPU"} if self.engine == "catboost" else {}
        probabilities = np.asarray(
            self.model.predict_proba(features, **kwargs), dtype=float
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != len(
            self.estimator_class_order
        ):
            msg = (
                "Estimator probability output is inconsistent with its class mapping: "
                f"shape={probabilities.shape}, classes={self.estimator_class_order}"
            )
            raise ArtifactIntegrityError(msg)
        aligned = np.zeros((probabilities.shape[0], self.num_classes), dtype=float)
        for source_index, encoded_class in enumerate(self.estimator_class_order):
            if not 0 <= encoded_class < self.num_classes:
                msg = f"Estimator returned unknown encoded class: {encoded_class}"
                raise ArtifactIntegrityError(msg)
            aligned[:, encoded_class] = probabilities[:, source_index]
        if not np.isfinite(aligned).all() or np.any(aligned < 0) or np.any(aligned > 1):
            msg = (
                "Estimator returned non-finite or out-of-range multiclass probabilities"
            )
            raise ArtifactIntegrityError(msg)
        totals = aligned.sum(axis=1)
        if np.any(totals <= 0):
            msg = (
                "Estimator returned a multiclass probability row with non-positive sum"
            )
            raise ArtifactIntegrityError(msg)
        aligned /= totals[:, None]
        return aligned

    def _artifact_state(self) -> Mapping[str, Any]:
        return {
            "num_classes": self.num_classes,
            "estimator_class_order": list(self.estimator_class_order),
        }

    def _restore_artifact_state(self, state: Mapping[str, Any]) -> None:
        self.num_classes = int(state.get("num_classes", 0))
        self.estimator_class_order = tuple(
            int(value) for value in state.get("estimator_class_order", ())
        )
