"""Cumulative vocabulary offsets for the single shared categorical embedding.

Identical logic to ``TabularPreprocessor.fit``: categorical columns are laid out
in disjoint contiguous id ranges inside ``[0, vocab_size)`` so one
``nn.Embedding`` table serves them all.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def build_offset_map(
    cat_cols: Sequence[str] | None,
    values_to_id: Mapping[str, Mapping],
    spec_tokens: Mapping[str, int] | None,
) -> tuple[dict[str, int], int]:
    """Return ``(offset_map, vocab_size)``.

    ``vocab_size`` starts at ``len(spec_tokens)`` (reserved low ids at the
    pipeline level) and grows by ``len(values_to_id[col])`` -- which already
    includes that column's per-column ``unk`` -- for every categorical column,
    in ``cat_cols`` order.
    """
    spec_tokens = spec_tokens or {}
    if not cat_cols:
        return {}, 0
    offset_map: dict[str, int] = {}
    vocab_size = len(spec_tokens)
    for col in cat_cols:
        offset_map[col] = vocab_size
        vocab_size += len(values_to_id[col])
    return offset_map, vocab_size
