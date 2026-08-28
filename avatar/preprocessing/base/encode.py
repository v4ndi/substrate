"""Pure-numpy transform math, kept bit-compatible with the Spark backend.

The Spark implementations these mirror:

* ``signed_log1p`` -> :func:`avatar.preprocessing.spark.standard_scaler.signed_log1p`
* ``standardize``  -> ``StandardScaler.transform``: ``(x - mean) / (std + 1e-8)``,
  then ``fillna(0.0)`` on null values only.
* ``CategoricalMapper`` -> ``LabelEncoder.transform``: known value -> id,
  unknown value or null -> ``unk`` id (0).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

EPS = 1e-8


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """``log1p(|x|) * sign(x)`` with ``sign(0) == 0``; NaN is preserved.

    Matches Spark ``F.log1p(F.abs(c)) * F.when(c == 0, 0).otherwise(F.signum(c))``.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.abs(x)) * np.sign(x)


def standardize(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    null_mask: np.ndarray | None = None,
    fillna: bool = True,
) -> np.ndarray:
    """Standardize a ``(n, n_features)`` float block column-wise.

    ``mean`` / ``std`` are 1-D of length ``n_features``. When ``fillna`` is set,
    positions flagged by ``null_mask`` (same shape as ``x``) are set to ``0.0``
    *after* scaling -- exactly like Spark's ``fillna({col: 0.0})`` which only
    touches nulls, not NaNs produced by the arithmetic.
    """
    x = np.asarray(x, dtype=np.float64)
    z = (x - mean) / (std + EPS)
    if fillna and null_mask is not None:
        z = np.where(null_mask, 0.0, z)
    return z


class CategoricalMapper:
    """Vectorized ``raw value -> id`` mapping for one categorical column.

    Mirrors ``LabelEncoder.transform``: any value not in ``mapping`` -- including
    null -- becomes ``unk_id``. Integer-keyed columns take a fast ``searchsorted``
    path; everything else falls back to a Python dict lookup per element.
    """

    def __init__(self, mapping: Mapping, unk_id: int = 0):
        self.unk_id = int(unk_id)
        self.mapping = dict(mapping)
        keys = list(self.mapping.keys())
        self._int_path = len(keys) > 0 and all(
            isinstance(k, (int, np.integer)) and not isinstance(k, bool) for k in keys
        )
        if self._int_path:
            order = np.argsort(np.array(keys, dtype=np.int64))
            self._sorted_keys = np.array(keys, dtype=np.int64)[order]
            self._sorted_ids = np.array(
                [self.mapping[keys[i]] for i in order], dtype=np.int64
            )

    def __call__(self, values, null_mask: np.ndarray | None = None) -> np.ndarray:
        n = len(values)
        out = np.full(n, self.unk_id, dtype=np.int64)
        if n == 0:
            return out
        if self._int_path:
            arr = np.asarray(
                [v if v is not None else np.iinfo(np.int64).min for v in values]
            )
            if arr.dtype.kind not in "iu":
                # object array with None already replaced; coerce
                arr = np.array(
                    [
                        np.iinfo(np.int64).min if v is None else int(v)
                        for v in values
                    ],
                    dtype=np.int64,
                )
            pos = np.searchsorted(self._sorted_keys, arr)
            pos_clipped = np.clip(pos, 0, len(self._sorted_keys) - 1)
            hit = self._sorted_keys[pos_clipped] == arr
            out = np.where(hit, self._sorted_ids[pos_clipped], self.unk_id)
        else:
            for i, v in enumerate(values):
                hit = self.mapping.get(v)
                if hit is not None:
                    out[i] = hit
        if null_mask is not None:
            out = np.asarray(out).copy()
            out[np.asarray(null_mask, dtype=bool)] = self.unk_id
        return out.astype(np.int64, copy=False)
