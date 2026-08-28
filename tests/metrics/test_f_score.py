import numpy as np
import pytest
import torch

from avatar.metrics import compute_f_score


@pytest.mark.hotpp
def test_perfect_match():
    preds = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    assert compute_f_score(preds, targets)["f_score"][0] == 1.0


@pytest.mark.hotpp
def test_partial_match():
    preds = torch.tensor([[1, 2, 4]], dtype=torch.int32)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    assert compute_f_score(preds, targets)["f_score"][0] == pytest.approx(2 / 3)


@pytest.mark.hotpp
def test_no_match():
    preds = torch.tensor([[4, 5, 6]], dtype=torch.int32)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    assert compute_f_score(preds, targets)["f_score"][0] == 0.0


@pytest.mark.hotpp
def test_empty_predictions():
    preds = torch.tensor([[]], dtype=torch.int32)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    assert compute_f_score(preds, targets)["f_score"][0] == 0.0


@pytest.mark.hotpp
def test_empty_targets():
    preds = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    targets = torch.tensor([[]], dtype=torch.int32)
    assert compute_f_score(preds, targets)["f_score"][0] == 0.0


@pytest.mark.hotpp
def test_duplicates_in_predictions():
    preds = torch.tensor([[1, 1, 2]], dtype=torch.int32)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    assert compute_f_score(preds, targets)["f_score"][0] == pytest.approx(2 / 3)


@pytest.mark.hotpp
def test_duplicates_in_targets():
    preds = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    targets = torch.tensor([[1, 1, 2]], dtype=torch.int32)
    assert compute_f_score(preds, targets)["f_score"][0] == pytest.approx(2 / 3)


@pytest.mark.hotpp
def test_macro_first():
    preds = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int32)
    targets = torch.tensor([[1, 1, 2], [1, 1, 2]], dtype=torch.int32)
    assert np.mean(compute_f_score(preds, targets)["f_score"]) == pytest.approx(2 / 3)


@pytest.mark.hotpp
def test_macro_second():
    preds = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int32)
    targets = torch.tensor([[1, 1, 2], [1, 2, 3]], dtype=torch.int32)
    assert np.mean(compute_f_score(preds, targets)["f_score"]) == pytest.approx(5 / 6)


@pytest.mark.hotpp
def test_empty_gt():
    preds = torch.tensor([[3, 4, 5], [4, 4, 0]], dtype=torch.int32)
    targets = torch.tensor([[2, 0, 0], [2, 0, 0]], dtype=torch.int32)
    assert np.mean(compute_f_score(preds, targets, empty_token=2)["f_score"]) == 0.0
    assert compute_f_score(preds, targets, empty_token=2)["tp"] == 0
    assert compute_f_score(preds, targets, empty_token=2)["fp"] == 5
    assert compute_f_score(preds, targets, empty_token=2)["fn"] == 0


@pytest.mark.hotpp
def test_empty_pred():
    preds = torch.tensor([[2], [2]], dtype=torch.int32)
    targets = torch.tensor([[3, 4], [5, 5]], dtype=torch.int32)
    assert np.mean(compute_f_score(preds, targets, empty_token=2)["f_score"]) == 0.0
    assert compute_f_score(preds, targets, empty_token=2)["tp"] == 0
    assert compute_f_score(preds, targets, empty_token=2)["fp"] == 0
    assert compute_f_score(preds, targets, empty_token=2)["fn"] == 4


@pytest.mark.hotpp
def test_empty_gt_pred_partial():
    preds = torch.tensor([[2, 0, 0], [3, 4, 5]], dtype=torch.int32)
    targets = torch.tensor([[3, 4, 0], [3, 5, 6]], dtype=torch.int32)
    assert np.mean(
        compute_f_score(preds, targets, empty_token=2)["f_score"]
    ) == pytest.approx(1 / 3)
    assert compute_f_score(preds, targets, empty_token=2)["tp"] == 2
    assert compute_f_score(preds, targets, empty_token=2)["fp"] == 1
    assert compute_f_score(preds, targets, empty_token=2)["fn"] == 3
