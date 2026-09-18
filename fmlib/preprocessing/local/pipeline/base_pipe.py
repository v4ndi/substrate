"""Local ``NumCatPipeline`` -- categorical label-encoding + numeric standardization.

Single streaming pass for ``fit`` (both accumulators fed together) and a single
streaming pass for ``transform`` (categorical encoded, then numeric scaled, in
that order -- matching Spark ``NumCatPipeline.transform``).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any

import numpy as np
import pyarrow as pa

from fmlib.preprocessing.base.accumulators import (
    MeanStdAccumulator,
    ValueCountAccumulator,
)
from fmlib.preprocessing.base.encode import EPS, signed_log1p
from fmlib.preprocessing.base.io import (
    Source,
    iter_record_batches,
    shard_files,
    spawn_context,
)

from ..label_encoder import LabelEncoder, _run_batches
from ..standard_scaler import StandardScaler


def _fit_shard(
    source: Source,
    read_cols: list[str],
    batch_rows: int,
    num_cols: list[str] | None,
    to_log_columns: list[str],
    cat_cols: list[str] | None,
    max_cardinality: int,
    on_overflow: str,
) -> tuple[MeanStdAccumulator | None, ValueCountAccumulator | None]:
    """Fold one group of files into accumulators. Runs in a worker process.

    Both accumulators are fed from the same pass: they update from the same
    batches, so splitting them would mean reading the source twice.
    """
    ms = MeanStdAccumulator(num_cols, to_log_columns) if num_cols else None
    vc = (
        ValueCountAccumulator(cat_cols, max_cardinality, on_overflow)
        if cat_cols
        else None
    )
    for batch in iter_record_batches(source, columns=read_cols, batch_rows=batch_rows):
        if ms is not None:
            ms.update(batch)
        if vc is not None:
            vc.update(batch)
    return ms, vc


class NumCatPipeline:
    """Shared behaviour of the streaming numeric/categorical preprocessors.

    Owns the label encoder and standard scaler, the streaming ``fit`` pass and
    the ``dump`` / ``load`` artifact format that the Spark backend also reads.
    """

    def __init__(
        self,
        categorical_columns: Sequence[str] | None,
        numeric_columns: Sequence[str] | None,
        label_encoder: LabelEncoder | None = None,
        standard_scaler: StandardScaler | None = None,
        label_encoder_kwargs: dict[str, Any] | None = None,
        standard_scaler_kwargs: dict[str, Any] | None = None,
        batch_rows: int = 250_000,
    ):
        assert categorical_columns is not None or numeric_columns is not None, (
            "You should specify categorical_columns or numeric_columns"
        )
        self.cat_cols = list(categorical_columns) if categorical_columns else None
        self.num_cols = list(numeric_columns) if numeric_columns else None
        self.batch_rows = batch_rows
        le_kw = dict(label_encoder_kwargs or {})
        ss_kw = dict(standard_scaler_kwargs or {})

        if label_encoder is not None:
            self.label_encoder = label_encoder
        elif self.cat_cols is not None:
            self.label_encoder = LabelEncoder(
                self.cat_cols, spec_tokens={"unk": 0}, **le_kw
            )
        else:
            self.label_encoder = None

        if standard_scaler is not None:
            self.standard_scaler = standard_scaler
        elif self.num_cols is not None:
            self.standard_scaler = StandardScaler(columns=self.num_cols, **ss_kw)
        else:
            self.standard_scaler = None

    # -- fit -------------------------------------------------------------
    def fit(self, source: Source, num_workers: int = 1) -> NumCatPipeline:
        """Learn the vocabulary and the numeric statistics in one pass.

        Args:
            source: Anything the io helpers accept.
            num_workers: Processes folding disjoint groups of files. Above one,
                the parent merges worker states in file order.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``num_workers > 1`` with ``LabelEncoder.order`` set
                to ``"first_seen"``, the one order that depends on how the data
                was traversed.
        """
        if self.num_cols is not None:
            assert sorted(self.num_cols) == sorted(self.standard_scaler.columns), (
                "numeric_columns differ from StandardScaler.columns"
            )
        if self.cat_cols is not None:
            assert sorted(self.cat_cols) == sorted(self.label_encoder.columns), (
                "categorical_columns differ from LabelEncoder.columns"
            )

        if (
            num_workers > 1
            and self.cat_cols
            and self.label_encoder.order == "first_seen"
        ):
            msg = (
                "label encoder order='first_seen' cannot be fitted with "
                "num_workers > 1: it is the one order that depends on the "
                "traversal, which a process pool does not preserve. Use "
                "'sorted' (the default) or 'count_desc'."
            )
            raise ValueError(msg)

        read_cols = (self.cat_cols or []) + (self.num_cols or [])
        shard_args = (
            read_cols,
            self.batch_rows,
            self.num_cols,
            sorted(self.standard_scaler.to_log_columns) if self.num_cols else [],
            self.cat_cols,
            self.label_encoder.max_cardinality if self.cat_cols else 0,
            self.label_encoder.on_overflow if self.cat_cols else "raise",
        )
        groups: list[Source] = (
            list(shard_files(source, num_workers)) if num_workers > 1 else [source]
        )
        ms = vc = None
        if len(groups) == 1:
            ms, vc = _fit_shard(groups[0], *shard_args)
        else:
            with ProcessPoolExecutor(
                max_workers=len(groups), mp_context=spawn_context()
            ) as pool:
                futures = [
                    pool.submit(_fit_shard, group, *shard_args) for group in groups
                ]
                # Merge in submission order: the arithmetic is deterministic
                # only if the order the states are folded in is.
                for future in futures:
                    shard_ms, shard_vc = future.result()
                    ms = shard_ms if ms is None else ms.merge(shard_ms)
                    vc = shard_vc if vc is None else vc.merge(shard_vc)
        if ms is not None:
            self.standard_scaler.mean_std = ms.finalize()
        if vc is not None:
            self.label_encoder.values_to_id = vc.finalize(
                spec_tokens=self.label_encoder.spec_tokens,
                frequency_encoder=self.label_encoder.frequency_encoder,
                order=self.label_encoder.order,
            )
            self.label_encoder._mappers = None
        return self

    # -- transform -----------------------------------------------------------
    def _encode_batch(self, batch: pa.RecordBatch) -> dict[str, pa.Array]:
        """``{column: array}`` -- categorical -> int64 ids, numeric -> float64 scaled."""
        out: dict[str, pa.Array] = {}
        mappers = self.label_encoder._get_mappers() if self.cat_cols else {}
        cat_set = set(self.cat_cols or [])
        num_set = set(self.num_cols or [])
        log_set = set(self.standard_scaler.to_log_columns) if self.num_cols else set()

        for name in batch.schema.names:
            col = batch.column(batch.schema.get_field_index(name))
            if name in cat_set:
                null_mask = col.is_null().to_numpy(zero_copy_only=False)
                ids = mappers[name](col.to_pylist(), null_mask=null_mask)
                out[name] = pa.array(ids, type=pa.int64())
            elif name in num_set:
                null_mask = np.asarray(
                    col.is_null().to_numpy(zero_copy_only=False), bool
                )
                x = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
                if name in log_set:
                    x = signed_log1p(x)
                mean = self.standard_scaler.mean_std[name]["mean"]
                std = self.standard_scaler.mean_std[name]["std"]
                z = (x - mean) / (std + EPS)
                if self.standard_scaler.fillna:
                    z = np.where(null_mask, 0.0, z)
                out[name] = pa.array(z, type=pa.float64())
            else:
                out[name] = col
        return out

    def transform(self, source: Source, output_path: str | None = None):
        return _run_batches(source, self._encode_batch, output_path, self.batch_rows)

    def fit_transform(self, source: Source, output_path: str | None = None):
        self.fit(source)
        return self.transform(source, output_path)

    # -- (de)serialization -------------------------------------------------
    def dump(self) -> dict:
        state = deepcopy({
            k: v
            for k, v in self.__dict__.items()
            if k not in ("label_encoder", "standard_scaler")
        })
        state["label_encoder"] = (
            self.label_encoder.dump() if self.label_encoder is not None else None
        )
        state["standard_scaler"] = (
            self.standard_scaler.dump() if self.standard_scaler is not None else None
        )
        return deepcopy(state)

    @classmethod
    def load(cls, attr_dict: dict) -> NumCatPipeline:
        attr_dict = deepcopy(attr_dict)
        inst = cls.__new__(cls)
        le = attr_dict.pop("label_encoder", None)
        ss = attr_dict.pop("standard_scaler", None)
        inst.label_encoder = LabelEncoder.load(le) if le else None
        inst.standard_scaler = StandardScaler.load(ss) if ss else None
        inst.__dict__.update(attr_dict)
        inst.batch_rows = attr_dict.get("batch_rows", 250_000)
        return inst
