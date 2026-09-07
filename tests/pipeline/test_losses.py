"""The loss contract: what modules return, and how pipelines consume it."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from avatar.losses import (
    ClassificationLoss,
    CompositeLoss,
    Loss,
    LossOutput,
    build_task_loss_fn,
)
from avatar.losses.next_k_tokens import HeadPrediction, NextKTokensLoss

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


# -- NextKTokensLoss ---------------------------------------------------------


def numeric_head(**overrides) -> HeadPrediction:
    defaults = {
        "logits": torch.zeros(1, 4, 1),
        "input_ids": torch.ones(1, 4),
        "n_classes": 1,
        "attention_mask": torch.ones(1, 4),
    }
    defaults.update(overrides)
    return HeadPrediction(**defaults)


def test_heads_are_keyed_by_name_and_horizon():
    loss = NextKTokensLoss(horizon=2)
    result = loss([{"amount": numeric_head()}, {"amount": numeric_head()}])
    assert set(result.components) == {"amount_head_0", "amount_head_1"}
    assert set(result.num_items) == set(result.components)
    # The trainer does the cross-rank weighting, so there is no scalar here.
    assert result.loss is None


def test_feature_weights_apply_in_training_only():
    loss = NextKTokensLoss(horizon=1, feature_loss_weights={"amount": 0.5})
    head = numeric_head(logits=torch.full((1, 4, 1), 3.0))
    loss.train()
    trained = loss([{"amount": head}]).components["amount_head_0"]
    loss.eval()
    evaluated = loss([{"amount": head}]).components["amount_head_0"]
    assert trained == pytest.approx(float(evaluated) * 0.5)


def test_later_horizons_are_discounted():
    loss = NextKTokensLoss(horizon=2, horizion_loss_weight=1.0)
    loss.train()
    head = numeric_head(logits=torch.full((1, 4, 1), 3.0))
    result = loss([{"amount": head}, {"amount": head}])
    # coefs are 1 and 2, and horizon 1 shifts one position further.
    assert result.components["amount_head_1"] < result.components["amount_head_0"]


def test_timedelta_keeps_its_horizon_discount_in_eval():
    """Pre-existing asymmetry, preserved deliberately by the extraction."""
    loss = NextKTokensLoss(horizon=2, horizion_loss_weight=1.0)
    loss.eval()
    head = numeric_head(logits=torch.full((1, 4, 1), 3.0))
    discounted = numeric_head(
        logits=torch.full((1, 4, 1), 3.0), scale_by_coef_in_eval=True
    )
    plain_result = loss([{"x": head}, {"x": head}])
    delta_result = loss([{"x": discounted}, {"x": discounted}])
    # Ordinary heads skip the coefficient outside training; timedelta does not.
    assert plain_result.components["x_head_1"] == pytest.approx(
        float(delta_result.components["x_head_1"]) * 2
    )


def test_num_items_counts_unmasked_positions():
    loss = NextKTokensLoss(horizon=1)
    head = numeric_head(attention_mask=torch.tensor([[1, 1, 0, 0]]))
    result = loss([{"amount": head}])
    # Offset 1 drops the first position, leaving one unmasked.
    assert int(result.num_items["amount_head_0"]) == 1
