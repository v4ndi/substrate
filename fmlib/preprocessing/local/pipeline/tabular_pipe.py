"""Local ``TabularPreprocessor`` -- artifact-compatible with the Spark one.

Adds, on top of :class:`NumCatPipeline`: cumulative categorical ``offset_map`` /
``vocab_size`` and packing into ``cat_features`` (``list<int64>``) /
``num_features`` (``list<float32>``) array columns, plus ``identity_cols``
pass-through in raw form. ``output="wide"`` keeps one column per feature instead
of packing (for dataset-level / GPU preprocessing).
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fmlib.preprocessing.base.io import Source, iter_record_batches
from fmlib.preprocessing.base.offsets import build_offset_map

from .base_pipe import NumCatPipeline


def _pack_list(arr_2d: np.ndarray, arrow_type: pa.DataType) -> pa.ListArray:
    """Pack a 2-D array into a parquet list column of ``arrow_type``."""
    n, k = arr_2d.shape
    flat = pa.array(np.ascontiguousarray(arr_2d).reshape(-1), type=arrow_type)
    offsets = pa.array(np.arange(0, n * k + 1, k, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, flat)


class TabularPreprocessor(NumCatPipeline):
    """Streaming tabular preprocessing: one machine, bounded memory.

    Emits ``cat_features`` (cumulative-offset ids, so all categorical columns
    share one embedding table) and ``num_features`` (standardised floats) — the
    two columns :class:`~fmlib.data.TabularDataset` expects.

    Produces the same artifact and the same output as the Spark backend.
    """

    def __init__(
        self,
        categorical_columns: Sequence[str] | None = None,
        numeric_columns: Sequence[str] | None = None,
        spec_tokens: dict | None = None,
        label_encoder=None,
        standard_scaler=None,
        label_encoder_kwargs=None,
        standard_scaler_kwargs=None,
        batch_rows: int = 250_000,
    ):
        if spec_tokens is None:
            spec_tokens = {"pad": 0}
        super().__init__(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            label_encoder=label_encoder,
            standard_scaler=standard_scaler,
            label_encoder_kwargs=label_encoder_kwargs,
            standard_scaler_kwargs=standard_scaler_kwargs,
            batch_rows=batch_rows,
        )
        self.spec_tokens = dict(spec_tokens) if spec_tokens is not None else {}
        self.offset_map: dict[str, int] = {}
        self.vocab_size = len(self.spec_tokens) if self.cat_cols is not None else 0

    # -- fit -------------------------------------------------------------
    def fit(
        self, source: Source, create_offset_only: bool = False
    ) -> TabularPreprocessor:
        if not create_offset_only:
            super().fit(source)
        self.offset_map, self.vocab_size = build_offset_map(
            self.cat_cols,
            self.label_encoder.values_to_id if self.cat_cols else {},
            self.spec_tokens,
        )
        return self

    def fit_transform(
        self, source, output_path=None, identity_cols=None, output="packed"
    ):
        self.fit(source)
        return self.transform(source, output_path, identity_cols, output)

    # -- transform -----------------------------------------------------------
    def _transform_batch(self, batch, identity_cols, output):
        identity_cols = identity_cols or []
        enc = self._encode_batch(batch)  # cat -> local int64 ids, num -> float64 scaled
        cat_set = set(self.cat_cols or [])
        num_set = set(self.num_cols or [])

        identity_raw = {
            c: batch.column(batch.schema.get_field_index(c)) for c in identity_cols
        }

        out: dict[str, pa.Array] = {}
        for name in batch.schema.names:
            if name in cat_set or name in num_set:
                continue
            out[name] = enc[name]

        if self.cat_cols:
            cat_2d = np.stack(
                [
                    np.asarray(enc[c].to_numpy(zero_copy_only=False), dtype=np.int64)
                    + self.offset_map[c]
                    for c in self.cat_cols
                ],
                axis=1,
            )
            if output == "packed":
                out["cat_features"] = _pack_list(cat_2d, pa.int64())
            else:
                for j, c in enumerate(self.cat_cols):
                    out[c] = pa.array(cat_2d[:, j], type=pa.int64())

        if self.num_cols:
            num_2d = np.stack(
                [
                    np.asarray(enc[c].to_numpy(zero_copy_only=False), dtype=np.float32)
                    for c in self.num_cols
                ],
                axis=1,
            )
            if output == "packed":
                out["num_features"] = _pack_list(num_2d, pa.float32())
            else:
                for j, c in enumerate(self.num_cols):
                    out[c] = pa.array(num_2d[:, j], type=pa.float32())

        for c, raw in identity_raw.items():
            out[c] = raw

        return pa.table(out)

    def transform(
        self, source: Source, output_path=None, identity_cols=None, output="packed"
    ):
        if output not in ("packed", "wide"):
            raise ValueError("output must be 'packed' or 'wide'")
        writer = None
        tables = []
        for batch in iter_record_batches(
            source, columns=None, batch_rows=self.batch_rows
        ):
            tbl = self._transform_batch(batch, identity_cols, output)
            if output_path is None:
                tables.append(tbl)
            else:
                if writer is None:
                    writer = pq.ParquetWriter(output_path, tbl.schema)
                writer.write_table(tbl)
        if output_path is None:
            return pa.concat_tables(tables) if tables else pa.table({})
        if writer is not None:
            writer.close()
        return output_path

    # -- (de)serialization -------------------------------------------------
    def dump(self) -> dict:
        state = super().dump()
        state.update({
            "spec_tokens": dict(self.spec_tokens),
            "offset_map": dict(self.offset_map),
            "vocab_size": self.vocab_size,
            "_backend": "local",
        })
        return deepcopy(state)

    @classmethod
    def load(cls, attr_dict: dict) -> TabularPreprocessor:
        attr_dict = deepcopy(attr_dict)
        attr_dict.pop("_backend", None)
        attr_dict.pop("_version", None)
        inst = super().load(attr_dict)
        inst.spec_tokens = attr_dict["spec_tokens"]
        inst.offset_map = attr_dict["offset_map"]
        inst.vocab_size = attr_dict["vocab_size"]
        return inst
