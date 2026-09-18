"""Local ``EventSequencePreprocessor`` -- artifact-compatible with the Spark one.

``fit`` reuses the streaming :class:`NumCatPipeline` pass (label-encode + scale).
``transform`` reproduces the Spark ``sort(id, time)`` + ``groupby(id, *groupby)``
+ ``collect_list`` step. To stay within a bounded memory budget on very large
inputs it shards rows into hash buckets on ``id_column`` (pass 1, streaming),
then sorts + groups each bucket independently (pass 2). ``n_buckets`` is chosen
from the row count; on small inputs it is 1 and the pass is a plain sort+group.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fmlib.preprocessing.base.io import Source, dataset_num_rows, iter_record_batches

from .base_pipe import NumCatPipeline

_CONVERSION_FACTORS = {"days": 86400, "weeks": 604800, "months": 2629800}
_ROWS_PER_BUCKET = 5_000_000


def date_to_scaled_unix(values: pa.Array, time_unit: str = "days") -> np.ndarray:
    """Epoch seconds (UTC) / conversion factor, as float32.

    Matches Spark ``(F.unix_timestamp(col) / factor).cast(FloatType())`` when the
    Spark session timezone is UTC.
    """
    factor = _CONVERSION_FACTORS.get(time_unit.lower())
    if factor is None:
        raise ValueError(f"Invalid time_unit: {time_unit}")
    secs = (
        values.cast(pa.timestamp("s")).cast(pa.int64()).to_numpy(zero_copy_only=False)
    )
    return (np.asarray(secs, dtype=np.float64) / factor).astype(np.float32)


class EventSequencePreprocessor(NumCatPipeline):
    """Streaming event-sequence preprocessing: one machine, bounded memory.

    Input is one row per event; output is one row per client, every attribute a
    time-ordered list. ``transform`` shards by a hash of the id column into
    bounded buckets, so a corpus larger than RAM still fits.

    Unlike the Spark backend, the ordering here is deterministic: events are
    sorted on the full-precision raw timestamp, and the output is always
    monotonic in time.
    """

    def __init__(
        self,
        categorical_columns: list[str],
        numeric_columns: list[str] | None,
        event_time_column: str = "evt_dttm",
        time_unit: str = "days",
        event_type_ids_column: str = "event_ids",
        id_column: str = "epk_id",
        groupby_columns: list[str] | None = None,
        label_encoder=None,
        standard_scaler=None,
        label_encoder_kwargs: dict[str, Any] | None = None,
        standard_scaler_kwargs: dict[str, Any] | None = None,
        batch_rows: int = 250_000,
    ):
        super().__init__(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            label_encoder=label_encoder,
            standard_scaler=standard_scaler,
            label_encoder_kwargs=label_encoder_kwargs,
            standard_scaler_kwargs=standard_scaler_kwargs,
            batch_rows=batch_rows,
        )
        if numeric_columns is not None:
            assert not set(categorical_columns).intersection(numeric_columns), (
                "Columns cannot be both categorical and numerical"
            )
        if time_unit not in _CONVERSION_FACTORS:
            raise ValueError(f"Invalid time_unit: {time_unit}")
        self.event_time_column = event_time_column
        self.id_column = id_column
        self.groupby_columns = list(groupby_columns) if groupby_columns else []
        self.event_type_ids_columns = event_type_ids_column
        self.time_unit = time_unit

    # -- meta ----------------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        return self.cat_cols + self.num_cols if self.num_cols else list(self.cat_cols)

    @property
    def sequence_columns(self) -> list[str]:
        return (
            [self.event_type_ids_columns, self.event_time_column]
            + (self.num_cols or [])
            + self.cat_cols
        )

    @property
    def columns_meta(self) -> dict[str, dict[str, Any]]:
        meta: dict[str, dict[str, Any]] = {}
        for column in self.columns:
            if self.num_cols and column in self.num_cols:
                meta[column] = {"type": "numeric", "n_classes": 1}
            else:
                meta[column] = {
                    "type": "categorical",
                    "n_classes": len(self.label_encoder.values_to_id[column]),
                }
        return meta

    # -- transform ---------------------------------------------------------
    def _prep_batch(self, batch: pa.RecordBatch) -> pd.DataFrame:
        """Encode + scale + fill numeric nulls + scale time -> a pandas frame."""
        enc = self._encode_batch(batch)
        data: dict[str, np.ndarray] = {}
        data[self.id_column] = enc[self.id_column].to_numpy(zero_copy_only=False)
        for gb in self.groupby_columns:
            data[gb] = np.asarray(enc[gb].to_pylist(), dtype=object)

        for c in self.num_cols or []:
            raw = batch.column(batch.schema.get_field_index(c))
            null_mask = np.asarray(raw.is_null().to_numpy(zero_copy_only=False), bool)
            vals = np.asarray(enc[c].to_numpy(zero_copy_only=False), dtype=np.float32)
            vals[null_mask] = 0.0
            data[c] = vals
        for c in self.cat_cols:
            data[c] = np.asarray(enc[c].to_numpy(zero_copy_only=False), dtype=np.int64)
        time_col = batch.column(batch.schema.get_field_index(self.event_time_column))
        data[self.event_time_column] = date_to_scaled_unix(time_col, self.time_unit)
        # Spark sorts on the raw timestamp *before* scaling; keep a full-precision
        # key so ties are not introduced by float32 rounding of the scaled value.
        data["__sort_key__"] = np.asarray(
            time_col.cast(pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        data[self.event_type_ids_columns] = np.asarray(
            batch.column(
                batch.schema.get_field_index(self.event_type_ids_columns)
            ).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        return pd.DataFrame(data)

    def _aggregate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = [self.id_column, *self.groupby_columns]
        df = df.sort_values([self.id_column, "__sort_key__"], kind="stable")
        list_cols = (
            (self.num_cols or [])
            + self.cat_cols
            + [self.event_time_column, self.event_type_ids_columns]
        )
        grouped = df.groupby(keys, sort=True, dropna=False)
        agg = grouped[list_cols].agg(list).reset_index()
        return agg[keys + list_cols]

    def _frame_to_table(self, agg: pd.DataFrame) -> pa.Table:
        arrays, names = [], []
        for k in [self.id_column, *self.groupby_columns]:
            vals = [
                None if (isinstance(v, float) and np.isnan(v)) else v
                for v in agg[k].tolist()
            ]
            arrays.append(pa.array(vals))
            names.append(k)
        for c in self.num_cols or []:
            arrays.append(pa.array(agg[c].tolist(), type=pa.list_(pa.float32())))
            names.append(c)
        for c in self.cat_cols:
            arrays.append(pa.array(agg[c].tolist(), type=pa.list_(pa.int64())))
            names.append(c)
        arrays.append(
            pa.array(agg[self.event_time_column].tolist(), type=pa.list_(pa.float32()))
        )
        names.append(self.event_time_column)
        arrays.append(
            pa.array(
                agg[self.event_type_ids_columns].tolist(), type=pa.list_(pa.int64())
            )
        )
        names.append(self.event_type_ids_columns)
        return pa.table(arrays, names=names)

    def transform(
        self,
        source: Source,
        output_path: str | None = None,
        n_buckets: int | None = None,
        tmp_dir: str | None = None,
    ):
        if n_buckets is None:
            n_buckets = max(1, math.ceil(dataset_num_rows(source) / _ROWS_PER_BUCKET))

        if n_buckets == 1:
            frames = [
                self._prep_batch(b)
                for b in iter_record_batches(
                    source, columns=None, batch_rows=self.batch_rows
                )
            ]
            agg = self._aggregate_frame(pd.concat(frames, ignore_index=True))
            table = self._frame_to_table(agg)
            if output_path is None:
                return table
            pq.write_table(table, output_path)
            return output_path

        work = tempfile.mkdtemp(prefix="fmlib_seq_", dir=tmp_dir)
        try:
            writers: dict[int, pq.ParquetWriter] = {}
            paths = {
                k: os.path.join(work, f"b{k:05d}.parquet") for k in range(n_buckets)
            }
            for batch in iter_record_batches(
                source, columns=None, batch_rows=self.batch_rows
            ):
                df = self._prep_batch(batch)
                bucket = (
                    pd.util.hash_pandas_object(
                        df[self.id_column], index=False
                    ).to_numpy()
                    % n_buckets
                )
                for k in np.unique(bucket):
                    part = pa.Table.from_pandas(
                        df.iloc[bucket == k], preserve_index=False
                    )
                    if k not in writers:
                        writers[k] = pq.ParquetWriter(paths[int(k)], part.schema)
                    writers[k].write_table(part)
            for w in writers.values():
                w.close()

            out_writer = None
            out_tables = []
            for k in range(n_buckets):
                if not os.path.exists(paths[k]):
                    continue
                agg = self._aggregate_frame(pd.read_parquet(paths[k]))
                table = self._frame_to_table(agg)
                if output_path is None:
                    out_tables.append(table)
                else:
                    if out_writer is None:
                        out_writer = pq.ParquetWriter(output_path, table.schema)
                    out_writer.write_table(table)
            if output_path is None:
                return pa.concat_tables(out_tables) if out_tables else pa.table({})
            if out_writer is not None:
                out_writer.close()
            return output_path
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def fit_transform(self, source: Source, output_path: str | None = None, **kw):
        self.fit(source)
        return self.transform(source, output_path, **kw)

    # -- (de)serialization -----------------------------------------------
    def dump(self) -> dict:
        state = super().dump()
        state.update({
            "groupby_columns": list(self.groupby_columns),
            "id_column": self.id_column,
            "event_time_column": self.event_time_column,
            "event_type_ids_columns": self.event_type_ids_columns,
            "time_unit": self.time_unit,
            "_backend": "local",
        })
        return deepcopy(state)

    @classmethod
    def load(cls, attr_dict: dict) -> EventSequencePreprocessor:
        attr_dict = deepcopy(attr_dict)
        attr_dict.pop("_backend", None)
        attr_dict.pop("_version", None)
        inst = super().load(attr_dict)
        inst.groupby_columns = attr_dict["groupby_columns"]
        inst.id_column = attr_dict["id_column"]
        inst.event_time_column = attr_dict["event_time_column"]
        inst.event_type_ids_columns = attr_dict["event_type_ids_columns"]
        inst.time_unit = attr_dict.get("time_unit", "days")
        return inst
