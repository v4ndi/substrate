"""Local (pyarrow + numpy) standard scaler, artifact-compatible with the Spark one.

Mirrors :class:`avatar.preprocessing.spark.StandardScaler`:

* optional ``signed_log1p`` on ``to_log_columns`` (applied in both fit and transform),
* sample std (ddof=1),
* ``(x - mean) / (std + 1e-8)``,
* ``fillna=True`` -> null values become ``0.0`` after scaling (nulls only, not NaNs).
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pyarrow as pa

from avatar.preprocessing.base.accumulators import MeanStdAccumulator
from avatar.preprocessing.base.encode import EPS, signed_log1p
from avatar.preprocessing.base.io import Source, iter_record_batches

from .label_encoder import _run_batches


class StandardScaler:
    def __init__(
        self,
        columns: Sequence[str],
        fillna: bool = True,
        to_log_columns: Sequence[str] | None = None,
        batch_rows: int = 250_000,
    ):
        assert len(columns) > 0, "Expected list of columns"
        self.columns = list(columns)
        self.fillna = fillna
        self.to_log_columns = list(to_log_columns) if to_log_columns is not None else []
        self.batch_rows = batch_rows
        self.mean_std = {c: {"mean": 0.0, "std": 0.0} for c in self.columns}

    # -- fit ---------------------------------------------------------------
    def fit(self, source: Source) -> StandardScaler:
        acc = MeanStdAccumulator(self.columns, self.to_log_columns)
        for batch in iter_record_batches(
            source, columns=self.columns, batch_rows=self.batch_rows
        ):
            acc.update(batch)
        self.mean_std = acc.finalize()
        return self

    # -- transform -------------------------------------------------------------
    def transform_batch(self, batch: pa.RecordBatch) -> dict[str, pa.Array]:
        log_set = set(self.to_log_columns)
        col_set = set(self.columns)
        out: dict[str, pa.Array] = {}
        for name in batch.schema.names:
            col = batch.column(batch.schema.get_field_index(name))
            if name not in col_set:
                out[name] = col
                continue
            null_mask = np.asarray(col.is_null().to_numpy(zero_copy_only=False), bool)
            x = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
            if name in log_set:
                x = signed_log1p(x)
            mean = self.mean_std[name]["mean"]
            std = self.mean_std[name]["std"]
            z = (x - mean) / (std + EPS)
            if self.fillna:
                z = np.where(null_mask, 0.0, z)
            out[name] = pa.array(z, type=pa.float64())
        return out

    def transform(self, source: Source, output_path: str | None = None):
        return _run_batches(source, self.transform_batch, output_path, self.batch_rows)

    def fit_transform(self, source: Source, output_path: str | None = None):
        self.fit(source)
        return self.transform(source, output_path)

    # -- (de)serialization ---------------------------------------------------
    def dump(self) -> dict:
        return deepcopy({
            "columns": list(self.columns),
            "fillna": self.fillna,
            "to_log_columns": list(self.to_log_columns),
            "mean_std": deepcopy(self.mean_std),
        })

    @classmethod
    def load(cls, attr_dict: dict) -> StandardScaler:
        attr_dict = deepcopy(attr_dict)
        inst = cls.__new__(cls)
        inst.columns = list(attr_dict["columns"])
        inst.fillna = attr_dict.get("fillna", True)
        inst.to_log_columns = attr_dict.get("to_log_columns") or []
        inst.batch_rows = attr_dict.get("batch_rows", 250_000)
        inst.mean_std = attr_dict["mean_std"]
        return inst
