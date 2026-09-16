"""Shared parquet preparation without model or lifecycle state."""

from dataclasses import dataclass
from typing import Any

import polars as pl

from avatar.automl.data import CanonicalColumnMapper, ParquetSource, normalize_date
from avatar.automl.execution import ExecutionContext
from avatar.automl.types import ParquetPath


@dataclass(frozen=True)
class DataPreparation:
    context: ExecutionContext

    def read(self, path: ParquetPath) -> pl.DataFrame:
        return self.read_source(ParquetSource.resolve(path))

    def read_source(self, source: ParquetSource) -> pl.DataFrame:
        return self.normalize_date(self.read_roles(source))

    def read_roles(self, source: ParquetSource) -> pl.DataFrame:
        config = self.context.config
        mapper = CanonicalColumnMapper.from_config(config)
        # Validate alias collisions against the complete schema before projection.
        mapper.normalize_frame(pl.DataFrame(schema=source.scan().collect_schema()))
        columns = tuple(
            dict.fromkeys(
                name
                for name in (
                    config.target_column,
                    config.client_id_column,
                    config.date_column,
                    config.group_column,
                    getattr(config, "treatment_column", None),
                    *config.categorical_columns,
                    *config.numerical_columns,
                    *config.hidden_state_columns,
                )
                if name is not None
            )
        )
        return mapper.normalize_frame(source.read(columns))

    def normalize_date(self, frame: pl.DataFrame) -> pl.DataFrame:
        date_column = self.context.internal_config.date_column
        return frame if date_column is None else normalize_date(frame, date_column)

    @staticmethod
    def source_manifest(source: ParquetSource) -> tuple[dict[str, Any], ...]:
        """Record deterministic file identity metadata for reproducibility checks."""
        return tuple(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "modified_ns": path.stat().st_mtime_ns,
            }
            for path in source.files
        )
