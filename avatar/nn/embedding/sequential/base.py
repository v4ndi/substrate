"""Base classes for event-sequence embeddings and their temporal component."""

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from avatar.data.sequential.batch import EventSequenceBatch
from avatar.nn.embedding.base.embedding import BaseEmbedding


class BaseEventSequenceEmbedding(BaseEmbedding):
    """Base class for event sequence embedding layers.

    Args:
        hidden_size: Dimensionality of the embeddings
        columns_meta: Dictionary mapping column names to their metadata. Can be:
            - Dict[str, Dict[str, Any]] (regular dictionary)
            - DictConfig (OmegaConf configuration)

    Raises:
        ValueError: If columns_meta is invalid or columns have inconsistent metadata
    """

    def __init__(self, hidden_size: int, columns_meta: dict[str, dict[str, Any]]):
        super().__init__(hidden_size=hidden_size)
        assert BaseEventSequenceEmbedding._check_columns_meta(columns_meta), (
            "Invalid columns_meta"
        )
        has_event_ids = ["event_id" in v for v in columns_meta.values()]
        if not any(has_event_ids):
            for attr in columns_meta.values():
                attr["event_id"] = None
        elif not all(has_event_ids):
            raise ValueError(
                "The event_id must be specified for everyone or not specified at all"
            )
        if isinstance(columns_meta, DictConfig):
            self._columns_meta = OmegaConf.to_container(columns_meta, resolve=True)
        elif isinstance(columns_meta, dict):
            self._columns_meta = deepcopy(columns_meta)
        else:
            raise ValueError("Union type of columns_meta must be dict or OmegaConf")
        self._columns_keys = tuple((self._columns_meta).keys())

    @staticmethod
    def _check_columns_meta(columns_meta):
        required_keys = {"type", "n_classes"}
        for col_info in columns_meta.values():
            # Check all required keys are present
            if not required_keys.issubset(col_info.keys()):
                return False
            # Validate 'type'
            typ = col_info["type"]
            if not isinstance(typ, str) or typ not in ("numeric", "categorical"):
                return False
            # Validate 'n_classes'
            n_classes = col_info["n_classes"]
            if not isinstance(n_classes, int) or n_classes < 1:
                return False
        return True

    @property
    def columns(self) -> list[str]:
        """List of column names in this embedding."""
        return list(self._columns_meta.keys())

    @property
    def columns_meta(self) -> dict[str, any]:
        """Dictionary mapping column names to their validated metadata."""
        return deepcopy(self._columns_meta)

    def forward(self, features) -> torch.FloatTensor:
        raise NotImplementedError("Forward method must be implemented by child classes")


class BaseTemporalEmbedding(nn.Module):
    """Base class for temporal embeddings.

    Args:
        embedding_dim: Dimensionality of the embeddings.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, features: EventSequenceBatch) -> torch.FloatTensor:
        raise NotImplementedError("Forward method must be implemented by child classes")
