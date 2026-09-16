"""Canonical internal names for user-configurable role columns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import polars as pl

from avatar.automl.config.base import BaseTaskConfig
from avatar.automl.exceptions import SchemaError

_CANONICAL_ROLES = {
    "target_column": "target",
    "client_id_column": "epk_id",
    "group_column": "group",
    "treatment_column": "treatment",
    "date_column": "date",
}


@dataclass(frozen=True)
class CanonicalColumnMapper:
    """Map configurable role columns to stable internal names.

    Attributes:
        to_canonical: Physical-to-internal column-name mapping.
        to_external: Internal-to-physical reverse mapping.
    """

    to_canonical: Mapping[str, str]
    to_external: Mapping[str, str]

    @classmethod
    def from_config(cls, config: BaseTaskConfig) -> "CanonicalColumnMapper":
        """Build a reversible role mapping from task configuration.

        Args:
            config: Configuration containing physical role-column names.

        Returns:
            A reversible mapper for all non-canonical role names.
        """
        forward: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for field_name, canonical in _CANONICAL_ROLES.items():
            external = getattr(config, field_name, None)
            if external is not None and external != canonical:
                forward[external] = canonical
                reverse[canonical] = external
        return cls(to_canonical=forward, to_external=reverse)

    def normalize_config(self, config: BaseTaskConfig) -> BaseTaskConfig:
        """Replace role and matching feature names with canonical names.

        Args:
            config: Public task configuration.

        Returns:
            An equivalent immutable configuration used inside the pipeline.
        """
        role_values = {
            field_name: None if getattr(config, field_name) is None else canonical
            for field_name, canonical in _CANONICAL_ROLES.items()
            if hasattr(config, field_name)
        }
        return replace(
            config,
            **role_values,
            categorical_columns=tuple(self.to_canonical.get(name, name) for name in config.categorical_columns),
            numerical_columns=tuple(self.to_canonical.get(name, name) for name in config.numerical_columns),
            hidden_state_columns=tuple(self.to_canonical.get(name, name) for name in config.hidden_state_columns),
        )

    def normalize_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Apply role aliases after rejecting canonical-name collisions.

        Args:
            frame: Input frame with physical role-column names.

        Returns:
            Frame with configured role columns renamed to internal names.

        Raises:
            SchemaError: If an alias and its canonical name are both present.
        """
        active = {source: target for source, target in self.to_canonical.items() if source in frame.columns}
        collisions = sorted(target for source, target in active.items() if target in frame.columns and target != source)
        if collisions:
            msg = f"Custom role aliases collide with canonical input columns: {collisions}"
            raise SchemaError(msg)
        return frame.rename(active)

    def restore_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Restore physical role names in a result frame.

        Args:
            frame: Result frame containing canonical role names.

        Returns:
            Frame with applicable public names restored.
        """
        active = {source: target for source, target in self.to_external.items() if source in frame.columns}
        return frame.rename(active)
