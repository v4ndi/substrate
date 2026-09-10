"""Narrowing the payload metrics get, before it crosses the wire.

In a distributed run the evaluation loop pickles ``(batch, output)`` for every
batch and sends it to rank 0. What is under test here is that a metric declaring
which fields it reads actually shrinks that payload, and that a metric declaring
nothing still receives everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from avatar.metrics import ScalarMetric
from avatar.train.utils import metric_field_selection, narrow_for_metrics


@dataclass
class Output:
    logits: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class Narrow(ScalarMetric):
    required_inputs = ("targets",)
    required_outputs = ("logits",)

    def update(self, inputs, outputs):
        pass

    def compute(self):
        return {}

    def reset(self):
        pass


class WantsEveryOutput(ScalarMetric):
    required_inputs = ()
    required_outputs = None

    def update(self, inputs, outputs):
        pass

    def compute(self):
        return {}

    def reset(self):
        pass


class WantsEverything(ScalarMetric):
    def update(self, inputs, outputs):
        pass

    def compute(self):
        return {}

    def reset(self):
        pass


def a_batch():
    return {
        "targets": torch.tensor([1.0]),
        "seq_features": torch.ones(1, 128),
        "epk_id": ["a"],
    }


def an_output():
    return Output(
        logits=torch.tensor([0.5]),
        hidden=torch.ones(1, 256),
        loss=torch.tensor(0.1),
    )


def test_selection_is_the_union_of_what_the_metrics_declare():
    inputs, outputs = metric_field_selection([Narrow(), Narrow()])
    assert inputs == frozenset({"targets"})
    assert outputs == frozenset({"logits"})


def test_one_undeclared_metric_widens_only_its_own_side():
    """Loss logging needs every output field but no input key."""
    inputs, outputs = metric_field_selection([Narrow(), WantsEveryOutput()])
    assert inputs == frozenset({"targets"})
    assert outputs is None


def test_a_metric_declaring_nothing_widens_both_sides():
    assert metric_field_selection([Narrow(), WantsEverything()]) == (None, None)


def test_no_metrics_means_no_narrowing():
    assert metric_field_selection([]) == (None, None)
    assert metric_field_selection(None) == (None, None)


def test_narrowing_drops_undeclared_input_keys():
    selection = metric_field_selection([Narrow()])
    batch, _ = narrow_for_metrics(a_batch(), an_output(), selection)
    assert set(batch) == {"targets"}


def test_narrowing_blanks_undeclared_output_fields_but_keeps_the_class():
    selection = metric_field_selection([Narrow()])
    _, output = narrow_for_metrics(a_batch(), an_output(), selection)
    assert isinstance(output, Output)
    assert output.logits is not None
    # Kept as attributes so that ``hasattr`` still answers True — a metric has
    # to test the value, not the attribute.
    assert output.hidden is None
    assert output.loss is None


def test_narrowing_leaves_the_original_untouched():
    original = an_output()
    selection = metric_field_selection([Narrow()])
    narrow_for_metrics(a_batch(), original, selection)
    assert original.hidden is not None


def test_no_selection_passes_the_payload_through_unchanged():
    batch, output = a_batch(), an_output()
    same_batch, same_output = narrow_for_metrics(batch, output, (None, None))
    assert same_batch is batch
    assert same_output is output


def test_a_wrapper_forwards_its_inner_metric_declaration():
    """GroupAverageMetricWrapper reads nothing itself; it must not widen the union."""
    from avatar.metrics.utils import GroupAverageMetricWrapper

    wrapper = GroupAverageMetricWrapper(Narrow())
    assert metric_field_selection([wrapper]) == (
        frozenset({"targets"}),
        frozenset({"logits"}),
    )
