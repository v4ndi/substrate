"""One backend class for every supervised TabNN task.

Boosting keeps a module per task because each one needs a different native
estimator. TabNN does not: the model is always ``SupervisedLearner`` and the
tasks differ only in ``num_classes``, ``task_type`` and how a logit becomes a
score -- a table, not four classes. The table is ``templates/``, and the
task-specific values reach it through the ``_backend_options()`` hook the task
layer already has.

What class identity used to buy, and how it is bought back: a ``BinaryTask``
could not load a multiclass artifact because the class would not match. Now
``task_name`` is written into ``backend.json`` and checked on ``load()``.

Nothing here is pickled. The weights are safetensors, the config and the
preprocessor are YAML, the metadata is JSON.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
import yaml

from fmlib.automl.backends.interface import ModelBackend
from fmlib.automl.data.schema import FeatureSchema
from fmlib.automl.exceptions import ArtifactIntegrityError, NotFittedError

from .assembly import CONFIG_CONTRACT_VERSION
from .metric import to_scores

__all__ = ["TabNNBackend"]

BACKEND_METADATA = "backend.json"
TRAIN_CONFIG = "train_config.yaml"
PREPROCESSOR = "preprocessor.yaml"
WEIGHTS = "model.safetensors"


@dataclass
class TabNNBackend(ModelBackend):
    """A fitted tabular network, plus everything needed to score with it.

    Attributes:
        task_name: AutoML task this model was trained for. Checked on load.
        num_classes: Head width.
        score_transform: How logits become scores for this task.
        class_order: Multiclass labels in training id order.
        train_config: The generated fmlib config, as a plain container.
        column_names: Canonical-to-physical names of the configured columns.
        hidden_states: ``{column: width}`` of the pass-through embeddings.
        preprocessor_state: The fitted preprocessor's artifact dict.
        state_dict: The trained weights, on CPU.
        validation_metric: The objective of the trial that produced these
            weights, carried so a report can name it without re-scoring.
    """

    task_name: str = "binary"
    num_classes: int = 1
    score_transform: str = "sigmoid"
    class_order: tuple[Any, ...] | None = None
    train_config: Mapping[str, Any] = field(default_factory=dict)
    column_names: Mapping[str, str] = field(default_factory=dict)
    hidden_states: Mapping[str, int] = field(default_factory=dict)
    preprocessor_state: Mapping[str, Any] | None = None
    state_dict: Mapping[str, Any] | None = None
    batch_size: int = 4096
    validation_metric: float = float("nan")

    # -- runtime -----------------------------------------------------------
    def _model(self):
        """Rebuild the network from its config and load the trained weights."""
        import torch
        from hydra.utils import instantiate

        if self.state_dict is None:
            msg = "TabNN backend has no weights; it was never fitted or loaded"
            raise NotFittedError(msg)
        model = instantiate(dict(self.train_config)["model"])
        model.load_state_dict({
            name: torch.as_tensor(value) for name, value in self.state_dict.items()
        })
        model.eval()
        return model

    def _preprocessor(self):
        from fmlib.preprocessing.local import TabularPreprocessor

        if self.preprocessor_state is None:
            msg = "TabNN backend has no fitted preprocessor"
            raise NotFittedError(msg)
        return TabularPreprocessor.load(dict(self.preprocessor_state))

    def _physical(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Rename canonical role columns back to what the preprocessor saw.

        The encoding reads raw parquet, so the preprocessor learned the names
        the files carry. The prediction path hands over canonical names. One
        rename is cheaper, and far easier to follow, than teaching the
        streaming encoder about aliases.
        """
        active = {
            canonical: physical
            for canonical, physical in self.column_names.items()
            if canonical in frame.columns and canonical != physical
        }
        return frame.rename(active) if active else frame

    def predict_score(self, frame: pl.DataFrame, schema: FeatureSchema) -> np.ndarray:
        """Score a frame, preserving row order.

        Args:
            frame: Rows to score, with canonical column names and embeddings
                still vectors.
            schema: The fitted schema; unused here because the encoded layout
                is fixed by the preprocessor, and kept for the contract.

        Returns:
            ``(n,)`` probabilities or values, or ``(n, K)`` for multiclass.
        """
        import pyarrow.dataset as ds
        import torch

        from fmlib.data.tabular.batch import TabularBatch

        if frame.height == 0:
            width = self.num_classes if self.score_transform == "softmax" else 1
            return np.zeros((0, width) if width > 1 else (0,), dtype=np.float64)

        encoded = self._preprocessor().transform(
            ds.dataset(self._physical(frame).to_arrow()),
            None,
            identity_cols=list(self.hidden_states),
            output="packed",
        )
        table = pl.from_arrow(encoded)
        model = self._model()

        outputs = []
        with torch.no_grad():
            for start in range(0, table.height, self.batch_size):
                chunk = table.slice(start, self.batch_size)
                batch = TabularBatch(
                    cat_features=_stack(chunk, "cat_features", torch.long),
                    num_features=_stack(chunk, "num_features", torch.float32),
                    hidden_states={
                        name: _stack(chunk, name, torch.float32)
                        for name in self.hidden_states
                    }
                    or None,
                )
                if batch.hidden_states is not None:
                    batch._hidden_states = {
                        name: torch.where(value.isnan(), 0, value)
                        for name, value in batch.hidden_states.items()
                    }
                logits = model(tab_features=batch).logits
                outputs.append(to_scores(logits, self.score_transform))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(outputs, axis=0)

    def feature_importance(self, schema: FeatureSchema) -> pl.DataFrame | None:
        """Networks have no native importances, and a guess would be worse."""
        return None

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write the four files that are this model: metadata, config, preprocessor, weights."""
        import torch
        from safetensors.torch import save_file

        if self.state_dict is None:
            msg = "TabNN backend has no weights to save"
            raise NotFittedError(msg)
        path.mkdir(parents=True, exist_ok=True)
        (path / BACKEND_METADATA).write_text(
            json.dumps(
                {
                    "engine": self.engine,
                    "backend": "tabnn",
                    "task_name": self.task_name,
                    "num_classes": self.num_classes,
                    "score_transform": self.score_transform,
                    "class_order": list(self.class_order)
                    if self.class_order is not None
                    else None,
                    "column_names": dict(self.column_names),
                    "hidden_states": dict(self.hidden_states),
                    "batch_size": self.batch_size,
                    "validation_metric": self.validation_metric,
                    "params": dict(self.params),
                    "random_state": self.random_state,
                    "config_contract_version": CONFIG_CONTRACT_VERSION,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        (path / TRAIN_CONFIG).write_text(
            yaml.safe_dump(dict(self.train_config), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        (path / PREPROCESSOR).write_text(
            yaml.safe_dump(
                dict(self.preprocessor_state or {}), sort_keys=True, allow_unicode=True
            ),
            encoding="utf-8",
        )
        save_file(
            {
                name: torch.as_tensor(value).contiguous().cpu()
                for name, value in self.state_dict.items()
            },
            str(path / WEIGHTS),
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> Self:
        """Restore a fitted model, refusing an artifact of a different task.

        Raises:
            ArtifactIntegrityError: If the directory is incomplete or was
                written for another backend family.
        """
        from safetensors.torch import load_file

        metadata_path = path / BACKEND_METADATA
        if not metadata_path.is_file():
            msg = f"TabNN artifact is missing {BACKEND_METADATA}: {path}"
            raise ArtifactIntegrityError(msg)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("backend") != "tabnn":
            msg = f"Artifact at {path} was written by backend {metadata.get('backend')!r}, not 'tabnn'"
            raise ArtifactIntegrityError(msg)
        for name in (TRAIN_CONFIG, PREPROCESSOR, WEIGHTS):
            if not (path / name).is_file():
                msg = f"TabNN artifact is missing {name}: {path}"
                raise ArtifactIntegrityError(msg)
        class_order = metadata.get("class_order")
        return cls(
            engine=metadata["engine"],
            params=metadata.get("params", {}),
            random_state=int(metadata.get("random_state", 42)),
            device=device,
            verbose=False,
            task_name=metadata["task_name"],
            num_classes=int(metadata["num_classes"]),
            score_transform=metadata["score_transform"],
            class_order=tuple(class_order) if class_order is not None else None,
            train_config=yaml.safe_load(
                (path / TRAIN_CONFIG).read_text(encoding="utf-8")
            ),
            column_names=metadata.get("column_names", {}),
            hidden_states=metadata.get("hidden_states", {}),
            batch_size=int(metadata.get("batch_size", 4096)),
            validation_metric=float(metadata.get("validation_metric", "nan")),
            preprocessor_state=yaml.safe_load(
                (path / PREPROCESSOR).read_text(encoding="utf-8")
            ),
            state_dict=load_file(str(path / WEIGHTS)),
        )

    def expect_task(self, task_name: str) -> None:
        """Refuse an artifact trained for another task.

        Class identity used to make this impossible; one class for four tasks
        means saying it out loud.

        Raises:
            ArtifactIntegrityError: If the artifact belongs to another task.
        """
        if self.task_name != task_name:
            msg = (
                f"This artifact was trained for task {self.task_name!r}, "
                f"not {task_name!r}"
            )
            raise ArtifactIntegrityError(msg)


def _stack(chunk: pl.DataFrame, column: str, dtype) -> Any:
    """Turn one packed list column of a slice into a ``(B, width)`` tensor."""
    import torch

    if column not in chunk.columns:
        return None
    values = chunk[column].to_numpy()
    return torch.as_tensor(np.stack(values), dtype=dtype)
