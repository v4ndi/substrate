"""Golden values for the two loss-reporting metrics.

Both used to disagree with their own docstrings, and both disagreed by an amount
that looks like a plausible number in MLflow — which is why they need fixed
inputs with a hand-computed answer rather than a smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fmlib.metrics import MultiLossMetric, UniversalLossesMetric


@dataclass
class LossesOutput:
    losses: dict[str, torch.Tensor]
    num_items: dict[str, torch.Tensor]


@dataclass
class FieldsOutput:
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    not_a_loss_field: torch.Tensor | None = None


def feed_multi_loss(batches):
    metric = MultiLossMetric()
    for loss, num_items in batches:
        metric.update(
            None,
            LossesOutput(
                losses={"head": torch.tensor(loss)},
                num_items={"head": torch.tensor(num_items)},
            ),
        )
    return metric


def test_multi_loss_divides_summed_loss_by_summed_items():
    """Ratio of sums, not mean of ratios: the two differ on uneven batches."""
    metric = feed_multi_loss([(2.0, 4.0), (4.0, 2.0)])
    assert metric.compute() == {"head": (2.0 + 4.0) / (4.0 + 2.0)}


def test_multi_loss_compute_is_repeatable():
    """``compute`` used to consume its own accumulator; the second call was garbage."""
    metric = feed_multi_loss([(2.0, 4.0), (4.0, 2.0)])
    assert metric.compute() == metric.compute()


def test_multi_loss_reports_zero_for_a_head_with_no_items():
    metric = feed_multi_loss([(0.0, 0.0)])
    assert metric.compute() == {"head": 0.0}


def test_multi_loss_reset_forgets_everything():
    metric = feed_multi_loss([(2.0, 4.0)])
    metric.reset()
    assert metric.compute() == {}


def test_universal_losses_averages_a_field_over_the_batches_it_appeared_in():
    """A loss present in one batch out of four is not one quarter of itself."""
    metric = UniversalLossesMetric()
    for step in range(4):
        metric.update(
            None,
            FieldsOutput(
                loss=torch.tensor(1.0),
                aux_loss=torch.tensor(3.0) if step == 3 else None,
            ),
        )
    assert metric.compute() == {"basic_loss": 1.0, "aux_loss": 3.0}


def test_universal_losses_ignores_fields_that_are_not_losses():
    metric = UniversalLossesMetric()
    metric.update(
        None, FieldsOutput(loss=torch.tensor(2.0), not_a_loss_field=torch.tensor(9.0))
    )
    assert metric.compute() == {"basic_loss": 2.0}


def test_universal_losses_compute_is_repeatable_and_a_plain_dict():
    metric = UniversalLossesMetric()
    metric.update(None, FieldsOutput(loss=torch.tensor(2.0)))
    first = metric.compute()
    assert first == metric.compute()
    # A defaultdict handed to the caller would grow silently on a missing key.
    assert type(first) is dict
