"""Default hyperparameter space of the TabNN backend.

Every axis is categorical on purpose. That makes the space enumerable, which
is what lets the search run the whole grid when the grid is smaller than the
trial budget instead of sampling a small discrete set with replacement.

Key names are flat -- ``hidden_size``, ``lr`` -- exactly as the boosting space
uses ``depth`` and ``learning_rate``. The public AutoML config does not leak the
internal shape of the model; translating a flat name into config paths (and
``hidden_size`` reaches two of them) is the assembly layer's job.

Fixed and deliberately *not* searched: ``dropout_p``, ``batch_size``,
``num_heads``, ``attn_dropout``, ``out_head_hidden_dim``, ``aggregation``,
``weight_decay``, ``max_epochs``, ``patience`` and ``amp``. The epoch ceiling
buys time rather than metric while early stopping is live, and ``amp`` is a
launch parameter: trials run in different precision are not comparable, bf16
needs Ampere or newer, so the value is chosen from the hardware and recorded in
``backend.json`` rather than searched.

This module holds data, not torch: ``fmlib.automl`` stays importable without a
GPU stack, which is what keeps the boosting path light.
"""

from typing import Any

from fmlib.automl.exceptions import UnsupportedBackendError

#: 2 * 3 * 3 = 18 combinations against a default ``n_trials`` of 10, so the
#: default run samples 10 distinct points. ``num_heads=8`` divides both hidden
#: sizes (32/8, 64/8); a user-supplied space must keep that divisible.
_DEFAULT_SPACE: dict[str, dict[str, Any]] = {
    "tabular_transformer": {
        "hidden_size": {"type": "categorical", "choices": [32, 64]},
        "num_layers": {"type": "categorical", "choices": [2, 3, 4]},
        "lr": {"type": "categorical", "choices": [1e-4, 3e-4, 1e-3]},
    }
}


def default_search_space(engine: str) -> dict[str, dict[str, Any]]:
    """Return the packaged TabNN search space for one engine.

    Args:
        engine: TabNN engine name.

    Returns:
        A fresh copy of the default space, safe for the caller to mutate.

    Raises:
        UnsupportedBackendError: If the engine has no packaged space.
    """
    try:
        space = _DEFAULT_SPACE[engine]
    except KeyError:
        supported = ", ".join(sorted(_DEFAULT_SPACE))
        msg = f"tabnn backend has no default search space for engine={engine!r}; supported: {supported}"
        raise UnsupportedBackendError(msg) from None
    return {name: dict(definition) for name, definition in space.items()}
