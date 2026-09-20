"""What a hand-written search space may say, and what it may not.

`search_space` is the one place a user hands AutoML a small program. Every
refusal here is the difference between a clear message at config time and a
trial that fails somewhere inside Optuna, or -- worse -- one that quietly
searches the wrong thing: a float range where an integer was meant, a step that
goes backwards, a `log` flag that is not a flag.

The space is validated against both backend families, because both accept one
and the rules are the same.
"""

from __future__ import annotations

import pytest

from fmlib.automl.backends.search import suggest_params
from fmlib.automl.exceptions import ConfigError


class _Trial:
    """Records what was asked of Optuna without running a study."""

    def __init__(self):
        self.asked: list[tuple[str, str, tuple]] = []

    def suggest_categorical(self, name, choices):
        self.asked.append((name, "categorical", tuple(choices)))
        return choices[0]

    def suggest_int(self, name, low, high, step=1, log=False):
        self.asked.append((name, "int", (low, high, step, log)))
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        self.asked.append((name, "float", (low, high, step, log)))
        return low


def _suggest(space, backend="boosting", engine="catboost"):
    return suggest_params(
        _Trial(),
        backend=backend,
        engine=engine,
        model_params={},
        search_space=space,
    )


BACKENDS = [("boosting", "catboost"), ("tabnn", "tabular_transformer")]


@pytest.mark.parametrize(("backend", "engine"), BACKENDS)
def test_a_categorical_sequence_is_offered_as_choices(backend, engine):
    trial = _Trial()
    params = suggest_params(
        trial,
        backend=backend,
        engine=engine,
        model_params={},
        search_space={"depth": [2, 3, 4]},
    )
    assert params["depth"] == 2
    assert trial.asked == [("depth", "categorical", (2, 3, 4))]


@pytest.mark.parametrize(("backend", "engine"), BACKENDS)
def test_an_empty_categorical_sequence_is_refused(backend, engine):
    """Nothing to choose from is not a search: it is a typo."""
    with pytest.raises(ConfigError, match="at least one choice"):
        _suggest({"depth": []}, backend, engine)


@pytest.mark.parametrize("definition", [7, "int", None, {"low": 1, "high": 3}])
def test_a_parameter_that_is_neither_a_range_nor_a_sequence_is_refused(definition):
    with pytest.raises(ConfigError, match="depth"):
        _suggest({"depth": definition})


@pytest.mark.parametrize("parameter_type", ["integer", "double", "", "INT"])
def test_an_unknown_range_type_is_refused(parameter_type):
    with pytest.raises(ConfigError, match="Unsupported search parameter"):
        _suggest({"depth": {"type": parameter_type, "low": 1, "high": 3}})


@pytest.mark.parametrize(
    ("low", "high"),
    [("1", 3), (1, "3"), (None, 3), (1, None), (True, 3)],
)
def test_non_numerical_bounds_are_refused(low, high):
    """`True` is an int in Python; as a bound it means something nobody wrote."""
    with pytest.raises(ConfigError, match="bounds must be numerical"):
        _suggest({"depth": {"type": "int", "low": low, "high": high}})


@pytest.mark.parametrize("log", ["yes", 1, 0, None])
def test_a_log_flag_that_is_not_a_flag_is_refused(log):
    """`log=1` would be truthy and silently switch the scale."""
    with pytest.raises(ConfigError, match="'log' must be boolean"):
        _suggest({"lr": {"type": "float", "low": 0.1, "high": 1.0, "log": log}})


@pytest.mark.parametrize("step", [0, -1, -0.5, "2", True])
def test_a_non_positive_step_is_refused(step):
    with pytest.raises(ConfigError, match="step must be positive"):
        _suggest({"depth": {"type": "int", "low": 1, "high": 9, "step": step}})


@pytest.mark.parametrize(
    "definition",
    [
        {"type": "int", "low": 1.5, "high": 9},
        {"type": "int", "low": 1, "high": 9.5},
        {"type": "int", "low": 1, "high": 9, "step": 1.5},
    ],
)
def test_an_integer_parameter_with_fractional_bounds_is_refused(definition):
    """Rounding silently is the alternative, and it changes the space searched."""
    with pytest.raises(ConfigError, match="requires integer bounds and step"):
        _suggest({"depth": definition})


def test_a_well_formed_range_reaches_optuna_unchanged():
    """The control: bounds, step and scale arrive as written."""
    trial = _Trial()
    suggest_params(
        trial,
        backend="boosting",
        engine="catboost",
        model_params={},
        search_space={
            "depth": {"type": "int", "low": 2, "high": 8, "step": 2},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True},
        },
    )
    assert trial.asked == [
        ("depth", "int", (2, 8, 2, False)),
        ("learning_rate", "float", (1e-3, 1e-1, None, True)),
    ]
