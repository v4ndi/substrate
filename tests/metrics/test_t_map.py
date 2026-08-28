import pytest
import torch

try:
    from hotpp.metrics import TMAPMetric

    _HAS_HOTPP = True
except ImportError:
    _HAS_HOTPP = False


@pytest.mark.hotpp
def test_map_metric():
    seq_target_mask = torch.tensor([[True, True], [False, True]])
    seq_target_times = torch.tensor([[-1, 2], [0, 8]])
    seq_target_labels = torch.tensor([[0, 1], [-1, 1]])
    seq_predicted_mask = torch.tensor([
        [True, True, False, True],
        [True, False, True, False],
    ])
    seq_predicted_times = torch.tensor([[2, 0, 3, 1], [4, 9, 8, 10]])
    seq_predicted_labels_logits = (
        torch.tensor([
            [
                [0, 0.9],
                [0.05, 0.05],
                [0, 0],
                [0, 1],
            ],
            [[0, 0.8], [0, 0], [0, 0.09], [0, 0]],
        ])
        .clip(min=1e-6)
        .log()
    )  # (1, 2, 4, 2).

    metric = TMAPMetric(time_delta_thresholds=[0, 1])
    metric.update(
        target_mask=seq_target_mask,
        target_times=seq_target_times,
        target_labels=seq_target_labels,
        predicted_mask=seq_predicted_mask,
        predicted_times=seq_predicted_times,
        predicted_labels_scores=seq_predicted_labels_logits,
    )
    # Matching (prediction -> target):
    # Batch 1: 0 -> 1 for horizon 1 and 3 -> 1, 1 -> 0 for horizon 2.
    # Batch 2: 2 -> 1.
    #
    # Scores horizon 1, batch 1:
    # class 0: Unmatched.
    # class 1: 0.9 (pos), 0, 1.
    #
    # Scores horizon 2, batch 1:
    # class 0: 0.0.
    # class 1: 0.9, 0, 1 (pos).
    #
    # Scores horizon 1, batch 2:
    # class 0: Empty.
    # class 1: 0.8, 0.09 (pos).
    #
    # Scores horizon 2, batch 2:
    # class 0: Empty.
    # class 1: 0.8, 0.09 (pos).
    #
    # All scores horizon 1:
    # class 0: Unmatched, recall is always 0.
    # class 1: 0, 0.09 (pos), 0.8, 0.9 (pos), 1.
    #
    # All scores horizon 2:
    # class 0: 0, 0.05 (pos), 0, 0, 0.
    # class 1: 0, 0.09 (pos), 0.8, 0.9, 1 (pos).
    ap_h1_c0 = 0
    ap_h1_c1 = 0.5
    ap_h2_c0 = 1
    ap_h2_c1 = 0.75
    map_gt = (ap_h1_c0 + ap_h1_c1 + ap_h2_c0 + ap_h2_c1) / 4
    assert metric.compute()["T-mAP"] == map_gt
