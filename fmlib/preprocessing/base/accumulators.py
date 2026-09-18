"""Single-pass streaming accumulators for ``fit``.

Both accumulators consume :class:`pyarrow.RecordBatch` chunks and hold memory
proportional to the number of feature columns (numeric stats) or the total number
of distinct categorical values, never to the number of rows.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import numpy as np
import pyarrow as pa

from .encode import signed_log1p


def _column_float(batch: pa.RecordBatch, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values_float64, null_mask_bool)`` for one column of a batch."""
    col = batch.column(batch.schema.get_field_index(name))
    null_mask = np.asarray(col.is_null(), dtype=bool)
    values = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
    return values, null_mask


class MeanStdAccumulator:
    """Streaming per-column mean and **sample** std (ddof=1), matching Spark.

    ``to_log_columns`` get :func:`signed_log1p` applied before accumulation, just
    like ``StandardScaler.fit``. Nulls are excluded; a genuine NaN in the data
    propagates to ``NaN`` stats exactly as ``F.mean`` / ``F.stddev`` do.

    Combination across batches uses Chan's parallel variance formula in float64.
    """

    def __init__(
        self,
        columns: Sequence[str],
        to_log_columns: Sequence[str] | None = None,
    ):
        self.columns = list(columns)
        self.to_log_columns = set(to_log_columns or [])
        self._n = {c: 0 for c in self.columns}
        self._mean = {c: 0.0 for c in self.columns}
        self._m2 = {c: 0.0 for c in self.columns}

    def update(self, batch: pa.RecordBatch) -> None:
        for col in self.columns:
            values, null_mask = _column_float(batch, col)
            valid = values[~null_mask]
            if valid.size == 0:
                continue
            if col in self.to_log_columns:
                valid = signed_log1p(valid)

            n_b = valid.size
            mean_b = float(valid.mean())
            m2_b = float(((valid - mean_b) ** 2).sum())

            n_a, mean_a, m2_a = self._n[col], self._mean[col], self._m2[col]
            n = n_a + n_b
            delta = mean_b - mean_a
            self._mean[col] = mean_a + delta * n_b / n
            self._m2[col] = m2_a + m2_b + delta * delta * n_a * n_b / n
            self._n[col] = n

    def merge(self, other: MeanStdAccumulator) -> MeanStdAccumulator:
        """Absorb the state of an accumulator fed a disjoint part of the data.

        This is the same Chan step :meth:`update` performs, with another
        accumulator's state in place of a batch's, which is what makes a
        process pool possible: every worker folds its own shards, the parent
        folds the workers.

        Floating-point addition is not associative, so combining in a different
        grouping than one sequential pass gives the same statistics to within
        rounding, not bit for bit. The categorical vocabulary, which is counted
        in integers, does combine exactly.

        Args:
            other: Accumulator over a disjoint part of the same columns.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If the two accumulators were configured differently.
        """
        if self.columns != other.columns:
            msg = f"Cannot merge accumulators over different columns: {self.columns} vs {other.columns}"
            raise ValueError(msg)
        if self.to_log_columns != other.to_log_columns:
            msg = "Cannot merge accumulators with different to_log_columns"
            raise ValueError(msg)
        for col in self.columns:
            n_b = other._n[col]
            if n_b == 0:
                continue
            n_a, mean_a, m2_a = self._n[col], self._mean[col], self._m2[col]
            mean_b, m2_b = other._mean[col], other._m2[col]
            n = n_a + n_b
            delta = mean_b - mean_a
            self._mean[col] = mean_a + delta * n_b / n
            self._m2[col] = m2_a + m2_b + delta * delta * n_a * n_b / n
            self._n[col] = n
        return self

    def finalize(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for col in self.columns:
            n = self._n[col]
            mean = self._mean[col]
            if n < 2:
                warnings.warn(
                    f"StandardScaler: column {col!r} has {n} non-null value(s); "
                    "sample std is undefined, emitting nan (Spark would emit null).",
                    stacklevel=2,
                )
                std = float("nan")
            else:
                std = float(np.sqrt(self._m2[col] / (n - 1)))
            out[col] = {"mean": float(mean), "std": std}
        return out


class CardinalityError(ValueError):
    """Raised when a categorical column exceeds ``max_cardinality`` during fit."""


class ValueCountAccumulator:
    """Streaming ``{column: {value: count}}`` for categorical columns.

    Nulls are excluded (they map to ``unk`` at transform time). ``max_cardinality``
    guards against id-like columns that do not belong in label encoding;
    ``on_overflow="topk"`` keeps the most frequent ``max_cardinality`` values
    instead of raising.
    """

    def __init__(
        self,
        columns: Sequence[str],
        max_cardinality: int = 1_000_000,
        on_overflow: str = "raise",
    ):
        if on_overflow not in ("raise", "topk"):
            raise ValueError("on_overflow must be 'raise' or 'topk'")
        self.columns = list(columns)
        self.max_cardinality = int(max_cardinality)
        self.on_overflow = on_overflow
        self._counts: dict[str, OrderedDict] = {c: OrderedDict() for c in self.columns}

    def update(self, batch: pa.RecordBatch) -> None:
        for col in self.columns:
            arr = batch.column(batch.schema.get_field_index(col))
            arr = arr.drop_null()
            if len(arr) == 0:
                continue
            vc = arr.value_counts()  # StructArray {values, counts}
            values = vc.field("values").to_pylist()
            counts = vc.field("counts").to_pylist()
            bucket = self._counts[col]
            for v, c in zip(values, counts, strict=False):
                if v is None:
                    continue
                v = _normalize_key(v)
                bucket[v] = bucket.get(v, 0) + int(c)
            if self.on_overflow == "raise" and len(bucket) > self.max_cardinality:
                raise CardinalityError(
                    f"Categorical column {col!r} exceeded max_cardinality="
                    f"{self.max_cardinality} during fit (seen {len(bucket)} "
                    "distinct values). Use a hash embedding for id-like columns, "
                    "or pass on_overflow='topk'."
                )

    def merge(self, other: ValueCountAccumulator) -> ValueCountAccumulator:
        """Absorb the counts of an accumulator fed a disjoint part of the data.

        Counts are integers, so this is exact: a parallel fit produces the same
        vocabulary as a sequential one, not an approximation of it. Order is
        another matter -- see the ``first_seen`` note in :meth:`finalize`.

        Args:
            other: Accumulator over a disjoint part of the same columns.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If the two accumulators were configured differently.
            CardinalityError: If the combined vocabulary exceeds
                ``max_cardinality`` under ``on_overflow='raise'``.
        """
        if self.columns != other.columns:
            msg = f"Cannot merge accumulators over different columns: {self.columns} vs {other.columns}"
            raise ValueError(msg)
        if (
            self.max_cardinality != other.max_cardinality
            or self.on_overflow != other.on_overflow
        ):
            msg = "Cannot merge accumulators with different cardinality policies"
            raise ValueError(msg)
        for col in self.columns:
            bucket = self._counts[col]
            for value, count in other._counts[col].items():
                bucket[value] = bucket.get(value, 0) + int(count)
            if self.on_overflow == "raise" and len(bucket) > self.max_cardinality:
                raise CardinalityError(
                    f"Categorical column {col!r} exceeded max_cardinality="
                    f"{self.max_cardinality} while merging parallel fit states "
                    f"(seen {len(bucket)} distinct values). Use a hash embedding "
                    "for id-like columns, or pass on_overflow='topk'."
                )
        return self

    def finalize(
        self,
        spec_tokens: Mapping[str, int],
        frequency_encoder: bool = False,
        order: str = "sorted",
    ) -> dict[str, dict]:
        """Build ``{column: {**spec_tokens, value: id, ...}}``.

        ``order`` (ignored when ``frequency_encoder`` is set, which forces
        count-descending): ``"sorted"`` by value, ``"count_desc"`` by frequency,
        ``"first_seen"`` by first appearance in the stream. Ties in the
        count-descending orders are broken by value for determinism.

        ``"first_seen"`` is the only order that depends on how the data was
        traversed, which is why a parallel fit refuses it.
        """
        start = len(spec_tokens)
        result: dict[str, dict] = {}
        for col in self.columns:
            bucket = self._counts[col]

            for tok in spec_tokens:
                if tok in bucket:
                    raise ValueError(
                        f"Column {col!r} contains a value equal to spec token "
                        f"{tok!r}; this collides with the reserved token."
                    )

            if self.on_overflow == "topk" and len(bucket) > self.max_cardinality:
                kept = sorted(
                    bucket.items(), key=lambda kv: (-kv[1], _sort_key(kv[0]))
                )[: self.max_cardinality]
                bucket = OrderedDict(kept)

            if frequency_encoder or order == "count_desc":
                values = [
                    v
                    for v, _ in sorted(
                        bucket.items(), key=lambda kv: (-kv[1], _sort_key(kv[0]))
                    )
                ]
            elif order == "first_seen":
                values = list(bucket.keys())
            elif order == "sorted":
                values = sorted(bucket.keys(), key=_sort_key)
            else:
                raise ValueError(f"unknown order {order!r}")

            mapping = dict(spec_tokens)
            for i, v in enumerate(values, start=start):
                mapping[v] = i
            result[col] = mapping
        return result


def _normalize_key(v):
    """Match how Spark ``collect_set`` hands values back to Python."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def _sort_key(v):
    return (0, v) if isinstance(v, int | float) else (1, str(v))
