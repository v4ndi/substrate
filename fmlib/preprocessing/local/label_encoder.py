"""Local (pyarrow + numpy) label encoder, artifact-compatible with the Spark one.

``dump()`` / ``load()`` produce and consume the exact same dict as
:class:`fmlib.preprocessing.spark.LabelEncoder`, so an artifact fitted by either
backend can be loaded by the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import pyarrow as pa

from fmlib.preprocessing.base.accumulators import ValueCountAccumulator
from fmlib.preprocessing.base.encode import CategoricalMapper
from fmlib.preprocessing.base.io import Source, iter_record_batches


class LabelEncoder:
    """Per-column ``value -> id`` encoder.

    Args:
        columns: categorical column names.
        spec_tokens: reserved tokens, values must be ``0..k-1`` and include
            ``"unk": 0`` (same contract as the Spark encoder).
        frequency_encoder: order ids by descending value frequency.
        order: id assignment order when ``frequency_encoder`` is False --
            ``"sorted"`` (by value, default, deterministic), ``"count_desc"``,
            or ``"first_seen"``. The Spark encoder uses ``collect_set`` here,
            whose order is non-deterministic; the local backend is deterministic.
        max_cardinality / on_overflow: guard for id-like columns (see
            :class:`ValueCountAccumulator`).
    """

    def __init__(
        self,
        columns: Sequence[str],
        spec_tokens: Mapping[str, int] = {"unk": 0},
        frequency_encoder: bool = False,
        order: str = "sorted",
        max_cardinality: int = 1_000_000,
        on_overflow: str = "raise",
        batch_rows: int = 250_000,
    ):
        assert len(columns) > 0, "Expected list of columns"
        assert list(range(len(spec_tokens))) == sorted(spec_tokens.values()), (
            "spec_tokens must be in range 0..k"
        )
        assert "unk" in spec_tokens and spec_tokens["unk"] == 0

        self.columns = list(columns)
        self.spec_tokens = dict(spec_tokens)
        self.frequency_encoder = frequency_encoder
        self.order = order
        self.max_cardinality = max_cardinality
        self.on_overflow = on_overflow
        self.batch_rows = batch_rows
        self.values_to_id = {col: dict(spec_tokens) for col in self.columns}
        self._mappers: dict[str, CategoricalMapper] | None = None

    # -- fit -------------------------------------------------------------
    def fit(self, source: Source) -> LabelEncoder:
        assert all(
            len(self.values_to_id[c]) == len(self.spec_tokens) for c in self.columns
        ), "Your encoder is already fitted"
        acc = ValueCountAccumulator(
            self.columns, self.max_cardinality, self.on_overflow
        )
        for batch in iter_record_batches(
            source, columns=self.columns, batch_rows=self.batch_rows
        ):
            acc.update(batch)
        self.values_to_id = acc.finalize(
            spec_tokens=self.spec_tokens,
            frequency_encoder=self.frequency_encoder,
            order=self.order,
        )
        self._mappers = None
        return self

    # -- transform -----------------------------------------------------------
    def _get_mappers(self) -> dict[str, CategoricalMapper]:
        if self._mappers is None:
            self._mappers = {
                col: CategoricalMapper(
                    {
                        k: v
                        for k, v in self.values_to_id[col].items()
                        if k not in self.spec_tokens
                    },
                    unk_id=self.values_to_id[col]["unk"],
                )
                for col in self.columns
            }
        return self._mappers

    def transform_batch(self, batch: pa.RecordBatch) -> dict[str, pa.Array]:
        """Return ``{column: array}`` with categorical columns replaced by ids."""
        mappers = self._get_mappers()
        out: dict[str, pa.Array] = {}
        for name in batch.schema.names:
            col = batch.column(batch.schema.get_field_index(name))
            if name in self.columns:
                null_mask = col.is_null().to_numpy(zero_copy_only=False)
                ids = mappers[name](col.to_pylist(), null_mask=null_mask)
                out[name] = pa.array(ids, type=pa.int64())
            else:
                out[name] = col
        return out

    def transform(self, source: Source, output_path: str | None = None):
        """Transform ``source``; write to ``output_path`` or return a ``pa.Table``."""
        return _run_batches(source, self.transform_batch, output_path, self.batch_rows)

    def fit_transform(self, source: Source, output_path: str | None = None):
        self.fit(source)
        return self.transform(source, output_path)

    # -- (de)serialization -------------------------------------------------
    def dump(self) -> dict:
        state = {
            "columns": list(self.columns),
            "spec_tokens": dict(self.spec_tokens),
            "frequency_encoder": self.frequency_encoder,
            "values_to_id": deepcopy(self.values_to_id),
        }
        return deepcopy(state)

    @classmethod
    def load(cls, attr_dict: dict) -> LabelEncoder:
        attr_dict = deepcopy(attr_dict)
        inst = cls.__new__(cls)
        inst.columns = list(attr_dict["columns"])
        inst.spec_tokens = dict(attr_dict.get("spec_tokens", {"unk": 0}))
        inst.frequency_encoder = attr_dict.get("frequency_encoder", False)
        inst.order = attr_dict.get("order", "sorted")
        inst.max_cardinality = attr_dict.get("max_cardinality", 1_000_000)
        inst.on_overflow = attr_dict.get("on_overflow", "raise")
        inst.batch_rows = attr_dict.get("batch_rows", 250_000)
        inst.values_to_id = attr_dict["values_to_id"]
        inst._mappers = None
        return inst


def _run_batches(source, fn, output_path, batch_rows):
    """Apply ``fn(batch) -> dict[str, pa.Array]`` over ``source``, stream to parquet."""
    import pyarrow.parquet as pq

    writer = None
    tables = []
    for batch in iter_record_batches(source, columns=None, batch_rows=batch_rows):
        cols = fn(batch)
        tbl = pa.table(cols)
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
