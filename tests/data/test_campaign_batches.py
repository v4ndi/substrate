"""The campaign sweep, which used to be an entrypoint and had no test.

It moved into the library when ``avatar/cam_inference.py`` was deleted, so the
behaviour the 23 campaign configs depend on is pinned here rather than in a
script nobody imports.
"""

from __future__ import annotations

import torch

from avatar.data.campaign import NO_CHANNEL, CampaignTaskChannelBatches
from avatar.data.tabular.batch import TabularBatch

RECORDS = 4


def batch() -> dict:
    return {
        "tab_features": TabularBatch(
            cat_features=torch.zeros(RECORDS, 3, dtype=torch.long),
            num_features=torch.zeros(RECORDS, 2),
        ),
        "epk_id": list(range(RECORDS)),
    }


def test_one_variant_per_task_channel_pair():
    sweep = CampaignTaskChannelBatches({
        "1": {"comm_type": [0, 1, 2]},
        "2": {"comm_type": [0]},
    })
    variants = list(sweep(batch()))
    assert len(variants) == 4
    assert [v["target_attr_2"][0] for v in variants] == [0, 1, 2, 0]
    assert [int(v["task_name"][0]) for v in variants] == [1, 1, 1, 2]


def test_numeric_task_names_become_a_tensor_column():
    sweep = CampaignTaskChannelBatches({"7": {"comm_type": [0]}})
    (variant,) = sweep(batch())
    assert torch.equal(variant["task_name"], torch.full((RECORDS,), 7))


def test_non_numeric_task_names_stay_strings():
    """Some campaigns key tasks by name; the metric writes either straight out."""
    sweep = CampaignTaskChannelBatches({"sa_response": {"comm_type": [0]}})
    (variant,) = sweep(batch())
    assert variant["task_name"] == ["sa_response"] * RECORDS


def test_a_task_without_channels_is_scored_once_with_no_group():
    sweep = CampaignTaskChannelBatches({"1": {"comm_type": [None]}})
    (variant,) = sweep(batch())
    assert variant["group"] is None
    assert variant["target_attr_2"] == [NO_CHANNEL] * RECORDS


def test_the_treated_arm_is_assumed_when_the_data_does_not_say():
    sweep = CampaignTaskChannelBatches({"1": {"comm_type": [0]}})
    (variant,) = sweep(batch())
    assert torch.equal(variant["is_treat"], torch.ones(RECORDS, dtype=torch.long))


def test_an_is_treat_column_already_in_the_batch_is_left_alone():
    original = batch()
    original["is_treat"] = torch.zeros(RECORDS, dtype=torch.long)
    sweep = CampaignTaskChannelBatches({"1": {"comm_type": [0]}})
    (variant,) = sweep(original)
    assert torch.equal(variant["is_treat"], torch.zeros(RECORDS, dtype=torch.long))


def test_the_source_batch_is_not_mutated():
    """Variants are shallow copies; the loop scores several from one batch."""
    original = batch()
    sweep = CampaignTaskChannelBatches({"1": {"comm_type": [0, 1]}})
    variants = list(sweep(original))
    assert "task_name" not in original
    assert variants[0]["tab_features"] is original["tab_features"]
