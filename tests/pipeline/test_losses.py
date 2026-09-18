"""The loss contract: what modules return, and how pipelines consume it."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fmlib.losses import (
    ClassificationLoss,
    CompositeLoss,
    ContrastiveLoss,
    Loss,
    LossOutput,
    build_task_loss_fn,
)

# -- criterion selection -----------------------------------------------------


@pytest.mark.parametrize(
    ("num_classes", "task_type", "expected"),
    [
        (1, "regression", nn.MSELoss),
        (1, "classification", nn.BCEWithLogitsLoss),
        (5, "classification", nn.CrossEntropyLoss),
    ],
)
def test_criterion_follows_from_the_task(num_classes, task_type, expected):
    assert isinstance(build_task_loss_fn(num_classes, task_type), expected)


def test_regression_with_several_classes_is_rejected():
    with pytest.raises(ValueError, match="Invalid combination"):
        build_task_loss_fn(3, "regression")


# -- ClassificationLoss ------------------------------------------------------


def test_classification_loss_returns_a_loss_output():
    loss = ClassificationLoss(num_classes=3)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    result = loss(logits, targets)
    assert isinstance(result, LossOutput)
    assert result.components == {}
    assert result.num_items is None
    assert torch.isclose(result.loss, nn.CrossEntropyLoss()(logits, targets))


def test_an_injected_criterion_overrides_the_default():
    loss = ClassificationLoss(num_classes=3, loss_fn=nn.L1Loss())
    assert isinstance(loss.loss_fn, nn.L1Loss)


# -- a head of width one against a column of targets -------------------------
#
# The head emits (B, 1) and every collate function emits (B,). Left to the
# criterion, BCE refuses the pair outright and MSE broadcasts it into (B, B) —
# the error of every prediction against every other record's target.


def test_a_one_wide_head_scores_each_record_against_its_own_target():
    loss = ClassificationLoss(num_classes=1, task_type="regression")
    logits = torch.randn(6, 1)
    targets = torch.randn(6)

    result = loss(logits, targets)

    all_pairs = ((logits - targets.unsqueeze(0)) ** 2).mean()
    assert torch.isclose(result.loss, nn.MSELoss()(logits.squeeze(1), targets))
    assert not torch.isclose(result.loss, all_pairs)


def test_the_regression_gradient_reaches_each_prediction_separately():
    """Under broadcasting every prediction is pulled towards the batch mean."""
    loss = ClassificationLoss(num_classes=1, task_type="regression")
    logits = torch.randn(6, 1, requires_grad=True)
    targets = torch.randn(6)

    loss(logits, targets).loss.backward()

    expected = 2 * (logits.detach().squeeze(1) - targets) / targets.numel()
    assert torch.allclose(logits.grad.squeeze(1), expected, atol=1e-6)


def test_binary_targets_arrive_as_class_ids_and_are_still_scored():
    """``TabularCollateFn`` emits LongTensor for anything not marked regression."""
    loss = ClassificationLoss(num_classes=1, task_type="classification")
    logits = torch.randn(8, 1)
    targets = torch.randint(0, 2, (8,))

    result = loss(logits, targets)

    expected = nn.BCEWithLogitsLoss()(logits.squeeze(1), targets.float())
    assert torch.isclose(result.loss, expected)


def test_a_target_column_shaped_like_the_head_is_accepted_too():
    loss = ClassificationLoss(num_classes=1, task_type="regression")
    logits = torch.randn(5, 1)
    targets = torch.randn(5, 1)

    result = loss(logits, targets)

    assert torch.isclose(
        result.loss, nn.MSELoss()(logits.squeeze(1), targets.squeeze(1))
    )


def test_a_wide_head_still_gets_class_indices():
    """Cross-entropy wants (B, K) against (B,) integers; nothing to reconcile."""
    loss = ClassificationLoss(num_classes=4, task_type="classification")
    logits = torch.randn(7, 4)
    targets = torch.randint(0, 4, (7,))

    result = loss(logits, targets)

    assert torch.isclose(result.loss, nn.CrossEntropyLoss()(logits, targets))


def test_a_wide_head_accepts_indices_delivered_as_a_column():
    loss = ClassificationLoss(num_classes=4, task_type="classification")
    logits = torch.randn(7, 4)
    targets = torch.randint(0, 4, (7, 1))

    result = loss(logits, targets)

    assert torch.isclose(result.loss, nn.CrossEntropyLoss()(logits, targets.squeeze(1)))


def test_a_batch_that_cannot_line_up_is_an_explicit_error():
    loss = ClassificationLoss(num_classes=1, task_type="regression")
    with pytest.raises(ValueError, match="head of width 1"):
        loss(torch.randn(4, 1), torch.randn(3))


def test_l1_penalty_is_added_and_reported():
    model = nn.Sequential()
    model.add_module("embed", nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model.embed.weight.fill_(1.0)  # L1 norm == 4

    logits = torch.zeros(2)
    targets = torch.zeros(2)
    plain = ClassificationLoss(num_classes=1)(logits, targets).loss
    result = ClassificationLoss(num_classes=1, l1_weight=0.5)(
        logits, targets, model=model
    )
    assert float(result.components["l1"].detach()) == pytest.approx(4.0)
    assert float(result.loss.detach()) == pytest.approx(float(plain) + 0.5 * 4.0)


def test_l1_penalty_without_a_model_is_an_error():
    loss = ClassificationLoss(num_classes=1, l1_weight=0.1)
    with pytest.raises(ValueError, match="no model to regularise"):
        loss(torch.zeros(2), torch.zeros(2))


# -- CompositeLoss -----------------------------------------------------------


class ConstantLoss(Loss):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, *args, **kwargs) -> LossOutput:
        return LossOutput(loss=torch.tensor(self.value))


def test_composite_sums_its_parts_with_weights():
    composite = CompositeLoss(
        {"a": ConstantLoss(2.0), "b": ConstantLoss(3.0)}, weights={"b": 0.5}
    )
    result = composite()
    assert result.loss == pytest.approx(2.0 + 0.5 * 3.0)
    assert set(result.components) == {"a", "b"}


def test_composite_rejects_weights_for_unknown_losses():
    with pytest.raises(ValueError, match="unknown losses"):
        CompositeLoss({"a": ConstantLoss(1.0)}, weights={"b": 1.0})


# -- ContrastiveLoss ---------------------------------------------------------


def contrastive_batch():
    """Logits with both arms present and both classes inside each arm."""
    torch.manual_seed(0)
    return (
        torch.randn(8, 2, requires_grad=True),
        torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
        torch.tensor([1, 0, 1, 0, 1, 0, 1, 0]),
    )


def test_separate_heads_scores_both_arms():
    """The treated arm's cross-entropy used to be computed and discarded."""
    logits, is_treat, targets = contrastive_batch()
    loss = ContrastiveLoss(alpha=0.0, separate_heads=True)

    value = loss(logits=logits, is_treat=is_treat, targets=targets)

    criterion = nn.CrossEntropyLoss()
    control = criterion(logits[is_treat == 0], targets[is_treat == 0])
    treated = criterion(logits[is_treat == 1], targets[is_treat == 1])
    assert float(value) == pytest.approx(float(control + treated))


def test_separate_heads_sends_gradient_to_the_treatment_head():
    """With the term dropped, treated rows received no classification signal."""
    logits, is_treat, targets = contrastive_batch()
    loss = ContrastiveLoss(alpha=0.0, separate_heads=True)

    loss(logits=logits, is_treat=is_treat, targets=targets).backward()

    assert logits.grad[is_treat == 1].norm() > 0
