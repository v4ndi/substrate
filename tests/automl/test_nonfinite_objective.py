"""A trial that scores `nan` must not win the search.

`nan` compares False against everything. The selection in both backend
families kept the first trial unconditionally — there was nothing better yet —
and then tested every later trial with `value > best_value`, which is False
when `best_value` is `nan`. So one degenerate validation split in the *first*
trial made every subsequent, better trial lose, and the search reported a
model whose score was not a number.

Nothing raised, nothing logged an error, and the artifact saved fine.

Found by writing the multiclass metric tests: `roc_auc_score(..., multi_class="ovr")`
does not raise when a class is absent from the split. It warns, averages a
`nan` into the macro, and returns it.

Two fixes, tested here: the metric refuses to return a non-score, and each
family's selection refuses to rank one — the second so that the guarantee does
not depend on every metric remembering.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fmlib.automl.exceptions import SchemaError
from fmlib.automl.metrics.multiclass import multiclass_objective


def _probabilities(rows: int, classes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(size=(rows, classes))
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        pytest.param(np.zeros(6, dtype=int), "'b', 'c'", id="one-class"),
        pytest.param(np.array([0, 0, 1, 1, 1, 0]), "'c'", id="two-of-three"),
    ],
)
def test_the_multiclass_objective_refuses_to_return_a_non_score(target, expected):
    """It used to return nan here, and nan is what ranks the trials."""
    with pytest.raises(SchemaError, match="undefined for the validation split"):
        multiclass_objective("roc_auc_ovr_macro", target, _probabilities(6, 3), "abc")

    # The message names what to change, not just that something is wrong.
    with pytest.raises(SchemaError, match=expected):
        multiclass_objective("roc_auc_ovr_macro", target, _probabilities(6, 3), "abc")


def test_a_complete_split_still_scores():
    value = multiclass_objective(
        "roc_auc_ovr_macro", np.array([0, 1, 2, 0, 1, 2]), _probabilities(6, 3), "abc"
    )
    assert math.isfinite(value)


# --------------------------------------------------------------------------- #
# The selection, driven through the real search in both families              #
# --------------------------------------------------------------------------- #
def _frame(rows: int, seed: int):
    import polars as pl

    rng = np.random.default_rng(seed)
    feature = rng.normal(size=rows)
    other = rng.normal(size=rows)
    logit = 1.6 * feature - 0.7 * other
    return pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 10_000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "segment": rng.choice(["a", "b"], rows),
        "feature": feature,
        "other": other,
        "target": (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(np.int8),
    })


class _MetricThatFailsFirst:
    """A real metric, except the first call returns something that is not one.

    This is how the bug arrives in production -- a degenerate validation split
    scoring `nan` on whichever trial happens to be first -- without needing a
    dataset degenerate enough to produce it.
    """

    def __init__(self, inner):
        self.inner = inner
        self.values: list[float] = []

    def __call__(self, target, scores):
        value = float("nan") if not self.values else float(self.inner(target, scores))
        self.values.append(value)
        return value


def test_a_first_trial_that_does_not_score_loses_the_boosting_search(
    tmp_path, monkeypatch
):
    """Before the fix this returned the nan trial and discarded every better one."""
    from fmlib.automl import BinaryTask, BinaryTaskConfig
    from fmlib.automl.tasks import binary as binary_module

    train = tmp_path / "train.parquet"
    valid = tmp_path / "valid.parquet"
    _frame(600, 1).write_parquet(train)
    _frame(300, 2).write_parquet(valid)

    inner = binary_module.BinaryTask._optimization_metric
    spy: dict[str, _MetricThatFailsFirst] = {}

    def patched(self, target, scores):
        metric = spy.setdefault(
            "m", _MetricThatFailsFirst(lambda t, s: inner(self, t, s))
        )
        return metric(target, scores)

    monkeypatch.setattr(binary_module.BinaryTask, "_optimization_metric", patched)

    task = BinaryTask(
        BinaryTaskConfig(
            output_dir=tmp_path / "out",
            env_type="local",
            device="cpu",
            backend="boosting",
            engine="catboost",
            target_column="target",
            client_id_column="epk_id",
            group_column=None,
            date_column="report_month",
            categorical_columns=["segment"],
            numerical_columns=["feature", "other"],
            hidden_state_columns=(),
            hyperopt=True,
            n_trials=3,
            verbose=False,
            random_state=42,
            search_space={
                "depth": {"type": "categorical", "choices": [2, 3, 4]},
            },
        )
    )
    result = task.train(train, valid)

    observed = spy["m"].values
    assert math.isnan(observed[0]), "the test did not reproduce the situation"
    assert len(observed) == 3

    selected = result.validation_metrics["global"]
    assert math.isfinite(selected), "the search selected a trial that did not score"
    assert selected == pytest.approx(max(observed[1:]))


@pytest.mark.slow
def test_a_first_trial_that_does_not_score_loses_the_tabnn_search(
    tmp_path, monkeypatch
):
    """Same rule, reached through `max` instead of a running comparison.

    The stub runs the real trials and only rewrites the first one's objective,
    so the losing candidate has genuine weights on disk. If the selection went
    back to preferring it, loading would succeed and the reported metric would
    be `nan` -- which is exactly how the bug looked.
    """
    from dataclasses import replace

    from fmlib.automl import BinaryTask, BinaryTaskConfig
    from fmlib.automl.backends.tabnn import fit as fit_module
    from fmlib.automl.backends.tabnn.runner import InProcessRunner

    train = tmp_path / "train.parquet"
    valid = tmp_path / "valid.parquet"
    _frame(600, 1).write_parquet(train)
    _frame(300, 2).write_parquet(valid)

    observed: list[float] = []

    class _FirstTrialDoesNotScore(InProcessRunner):
        def collect(self, handle):
            result = super().collect(handle)
            if not observed and result.objective is not None:
                observed.append(result.objective)
                return replace(result, objective=float("nan"))
            if result.objective is not None:
                observed.append(result.objective)
            return result

    monkeypatch.setattr(fit_module, "InProcessRunner", _FirstTrialDoesNotScore)

    task = BinaryTask(
        BinaryTaskConfig(
            output_dir=tmp_path / "out",
            env_type="local",
            device="cpu",
            backend="tabnn",
            engine="tabular_transformer",
            target_column="target",
            client_id_column="epk_id",
            group_column=None,
            date_column="report_month",
            categorical_columns=["segment"],
            numerical_columns=["feature", "other"],
            hidden_state_columns=(),
            hyperopt=True,
            n_trials=3,
            verbose=False,
            random_state=42,
            search_space={
                "hidden_size": {"type": "categorical", "choices": [16]},
                "num_layers": {"type": "categorical", "choices": [1, 2, 3]},
                "max_epochs": {"type": "categorical", "choices": [1]},
                "batch_size": {"type": "categorical", "choices": [128]},
                "num_heads": {"type": "categorical", "choices": [2]},
                "num_workers": {"type": "categorical", "choices": [0]},
                "evaluations_per_epoch": {"type": "categorical", "choices": [1]},
                "patience": {"type": "categorical", "choices": [1]},
            },
        )
    )
    result = task.train(train, valid)

    assert len(observed) == 3, "the search did not run three trials"
    selected = result.validation_metrics["global"]
    assert math.isfinite(selected), "the search selected a trial that did not score"
    assert selected == pytest.approx(max(observed[1:]))
