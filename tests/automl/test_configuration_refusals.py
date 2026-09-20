"""Small refusals, all of them messages a user will read.

Nothing here is deep. What these lines have in common is that each is the last
place a mistake can be named before it becomes something harder to read: a
metric that does not exist for a task, a metric list given as a bare string, a
template that was not packaged, an engine with no defaults. Left uncovered they
were also the places most likely to break silently under a refactor, because
nothing exercised them.

`resolve_amp` is here for a different reason: mixed precision is chosen from
the hardware rather than searched, so what it answers on a given machine is a
fact worth pinning rather than assuming.
"""

from __future__ import annotations

import pytest

from fmlib.automl.backends.tabnn.assembly import load_template, resolve_amp
from fmlib.automl.config.boosting import default_model_params, default_search_space
from fmlib.automl.exceptions import ConfigError
from fmlib.automl.metrics import (
    default_optimization_metric,
    resolve_evaluation_metrics,
    resolve_metric,
)

TASKS = ("binary", "multiclass", "regression", "response", "uplift")


# --------------------------------------------------------------------------- #
# Metric selection                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_a_default_optimization_metric(task):
    assert default_optimization_metric(task)


def test_a_task_with_no_default_metric_is_named():
    with pytest.raises(ConfigError, match="No default optimization metric"):
        default_optimization_metric("clustering")


@pytest.mark.parametrize("name", ["", None, 42, b"roc_auc"])
def test_a_metric_name_that_is_not_a_name_is_refused(name):
    with pytest.raises(ConfigError, match="non-empty string"):
        resolve_metric(name, "binary", "evaluation")


def test_a_metric_list_given_as_a_string_is_refused():
    """`metrics="roc_auc"` iterates into characters, so this must not be allowed."""
    with pytest.raises(ConfigError, match="not a string"):
        resolve_evaluation_metrics("roc_auc", "binary")


def test_an_empty_metric_list_is_refused_rather_than_treated_as_defaults():
    """Silence here would score a model on nothing and report success."""
    with pytest.raises(ConfigError, match="must not be empty"):
        resolve_evaluation_metrics([], "binary")


@pytest.mark.parametrize("names", [["roc_auc", ""], ["roc_auc", None], [1, 2]])
def test_a_metric_list_with_a_non_name_is_refused(names):
    with pytest.raises(ConfigError, match="non-empty string names"):
        resolve_evaluation_metrics(names, "binary")


def test_a_repeated_metric_is_refused():
    """Duplicates would be computed twice and collide in the result table."""
    with pytest.raises(ConfigError, match="unique names"):
        resolve_evaluation_metrics(["roc_auc", "roc_auc"], "binary")


@pytest.mark.parametrize("task", TASKS)
def test_the_task_defaults_resolve(task):
    assert resolve_evaluation_metrics(None, task)


# --------------------------------------------------------------------------- #
# TabNN assembly                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_a_packaged_template(task):
    assert load_template(task)["model"]


def test_a_task_without_a_template_lists_the_ones_that_exist():
    """The list is the point: it says what the caller could have asked for."""
    with pytest.raises(ConfigError) as error:
        load_template("clustering")
    message = str(error.value)
    assert "clustering" in message
    for task in TASKS:
        assert task in message


def test_amp_is_off_on_the_cpu():
    """Not hardware-dependent: no GPU means no mixed precision, always."""
    assert resolve_amp("cpu") == "no"
    assert resolve_amp("anything else") == "no"


def test_amp_follows_the_card_rather_than_the_config(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_amp("gpu") == "no"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_amp("gpu") == "no", "pre-Ampere must not be given bf16"

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_amp("gpu") == "bf16"


@pytest.mark.gpu
def test_amp_on_this_machine_is_what_the_card_supports():
    """The real answer here, not a mocked one."""
    import torch

    expected = "bf16" if torch.cuda.is_bf16_supported() else "no"
    assert resolve_amp("gpu") == expected


# --------------------------------------------------------------------------- #
# Boosting defaults                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine", ["catboost", "xgboost"])
def test_both_engines_have_packaged_defaults(engine):
    assert default_model_params(engine)
    assert default_search_space(engine, n_trials=10)


@pytest.mark.parametrize("engine", ["lightgbm", "", "CatBoost"])
def test_an_engine_without_defaults_is_named(engine):
    with pytest.raises(ValueError, match="engine="):
        default_model_params(engine)
    with pytest.raises(ValueError, match="engine="):
        default_search_space(engine, n_trials=10)


def test_catboost_defaults_adapt_to_the_row_count():
    """The adaptation exists; that it *is* an adaptation is what is asserted."""
    small = default_model_params("catboost", task="binary", train_rows=1_000)
    large = default_model_params("catboost", task="binary", train_rows=10_000_000)
    assert small != large


def test_defaults_are_not_adapted_without_both_task_and_rows():
    plain = default_model_params("catboost")
    assert default_model_params("catboost", task="binary") == plain
    assert default_model_params("catboost", train_rows=1_000) == plain
    assert default_model_params("xgboost", task="binary", train_rows=1_000) == (
        default_model_params("xgboost")
    )
