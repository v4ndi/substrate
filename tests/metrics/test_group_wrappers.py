"""Slicing a batch per group, and what the wrapper is allowed to forget.

The wrapper hands each group its own metric instance, so what matters is that
the slice it hands over is a real slice — every column cut to the same records,
nothing silently dropped and nothing silently left at full batch width — and
that ``reset`` leaves the wrapper as if it had just been built.
"""

import dataclasses
import logging
import types

import numpy as np
import pytest
import torch

from fmlib.metrics import ResponseMetrics
from fmlib.metrics.base import ScalarMetric
from fmlib.metrics.utils import (
    GroupAverageMetricWrapper,
    GroupDevidedMetricsWrapper,
)

#: The group key for channel "a" — a one-column group is keyed by a 1-tuple.
CHANNEL_A = ("a",)


class RecordingMetric(ScalarMetric):
    """Keeps whatever it was handed, so a test can look at the slice."""

    def __init__(self):
        self.seen: list[tuple] = []

    def update(self, inputs, outputs) -> None:
        self.seen.append((inputs, outputs))

    def compute(self) -> dict[str, float]:
        return {"batches": float(len(self.seen))}

    def reset(self) -> None:
        self.seen = []


@dataclasses.dataclass
class FakeOutput:
    """A model output with more than just logits on it."""

    logits: torch.Tensor
    conversion: torch.Tensor
    task_ids: np.ndarray
    names: list
    total_loss: torch.Tensor


def fake_output(size: int) -> FakeOutput:
    return FakeOutput(
        logits=torch.arange(size, dtype=torch.float32).unsqueeze(1),
        conversion=torch.arange(size, dtype=torch.float32),
        task_ids=np.arange(size),
        names=[f"r{i}" for i in range(size)],
        total_loss=torch.tensor(0.5),
    )


def fake_inputs(channels: list[str]) -> dict:
    size = len(channels)
    return {
        "channel": channels,
        "epk_id": np.arange(size),
        "targets": torch.arange(size, dtype=torch.float32),
        "split_type": np.array(["calib"] * size),
        "nested": {"extra": torch.arange(size)},
    }


@pytest.fixture
def wrapper_and_groups():
    """A wrapper over a recording metric, keyed by the ``channel`` column."""
    built: list[RecordingMetric] = []

    def factory():
        metric = RecordingMetric()
        built.append(metric)
        return metric

    wrapper = GroupDevidedMetricsWrapper(
        metric_class=factory, columns_to_devide=["channel"]
    )
    return wrapper, built


def test_every_output_field_is_cut_to_the_slice(wrapper_and_groups):
    """Not only ``logits``: a metric reading another field must get the slice.

    Only ``logits`` used to be replaced, so any other field arrived at full
    batch width and no longer lined up with it.
    """
    wrapper, _ = wrapper_and_groups
    channels = ["a", "b", "a", "b"]
    wrapper.update(fake_inputs(channels), fake_output(len(channels)))

    _, outputs = wrapper.group2metrics[CHANNEL_A].seen[0]
    assert outputs.logits.squeeze(1).tolist() == [0.0, 2.0]
    assert outputs.conversion.tolist() == [0.0, 2.0]
    assert outputs.task_ids.tolist() == [0, 2]
    assert outputs.names == ["r0", "r2"]


def test_a_field_that_is_not_per_record_passes_through(wrapper_and_groups):
    """A scalar loss has no batch dimension to cut."""
    wrapper, _ = wrapper_and_groups
    channels = ["a", "b"]
    wrapper.update(fake_inputs(channels), fake_output(len(channels)))

    _, outputs = wrapper.group2metrics[CHANNEL_A].seen[0]
    assert outputs.total_loss.item() == pytest.approx(0.5)


def test_numpy_columns_reach_the_inner_metric(wrapper_and_groups):
    """A numpy input column used to be dropped from the slice entirely."""
    wrapper, _ = wrapper_and_groups
    channels = ["a", "b", "a"]
    wrapper.update(fake_inputs(channels), fake_output(len(channels)))

    inputs, _ = wrapper.group2metrics[CHANNEL_A].seen[0]
    assert inputs["epk_id"].tolist() == [0, 2]
    assert inputs["split_type"].tolist() == ["calib", "calib"]
    assert inputs["targets"].tolist() == [0.0, 2.0]
    assert inputs["nested"]["extra"].tolist() == [0, 2]
    assert inputs["channel"] == ["a", "a"]


def test_the_original_batch_is_left_alone(wrapper_and_groups):
    """Slicing must not write back into the caller's output object."""
    wrapper, _ = wrapper_and_groups
    channels = ["a", "b"]
    inputs, outputs = fake_inputs(channels), fake_output(len(channels))
    wrapper.update(inputs, outputs)

    assert outputs.logits.shape[0] == 2
    assert outputs.names == ["r0", "r1"]
    assert inputs["epk_id"].tolist() == [0, 1]


def test_a_training_step_output_can_be_sliced(wrapper_and_groups):
    """Training outputs are not graph leaves, and deepcopy refuses them."""
    wrapper, _ = wrapper_and_groups
    weight = torch.ones(1, requires_grad=True)
    logits = (torch.arange(4, dtype=torch.float32).unsqueeze(1) * weight).sigmoid()
    outputs = types.SimpleNamespace(logits=logits)
    assert not logits.is_leaf

    wrapper.update(fake_inputs(["a", "b", "a", "b"]), outputs)

    _, seen = wrapper.group2metrics[CHANNEL_A].seen[0]
    assert seen.logits.shape[0] == 2


def test_reset_returns_a_freshly_built_wrapper(wrapper_and_groups):
    """The groups go with their metrics."""
    wrapper, _ = wrapper_and_groups
    wrapper.update(fake_inputs(["a", "b"]), fake_output(2))
    assert set(wrapper.group2metrics) == {("a",), ("b",)}

    wrapper.reset()
    assert wrapper.group2metrics == {}


def test_a_group_that_leaves_the_data_does_not_break_the_next_epoch():
    """The real symptom: an empty accumulator raises inside the inner metric.

    ``reset`` used to keep the group dictionary, so a channel present in the
    first epoch and absent from the second reached ``compute`` with nothing
    buffered, and ``ResponseMetrics`` raised ``IndexError`` on ``preds[0]``.
    """
    wrapper = GroupDevidedMetricsWrapper(
        metric_class=lambda: ResponseMetrics(main_metric="roc_auc_score"),
        columns_to_devide=["channel"],
    )

    def batch(channels, targets, logits):
        inputs = {
            "channel": channels,
            "targets": torch.tensor(targets, dtype=torch.float32),
            "split_type": np.array(["calib"] * len(targets)),
        }
        outputs = types.SimpleNamespace(
            logits=torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
        )
        return inputs, outputs

    wrapper.update(*batch(["a", "a", "b", "b"], [0, 1, 0, 1], [-1.0, 1.0, -1.0, 1.0]))
    first = wrapper.compute()
    assert any(name.startswith("channel_b_") for name in first)

    wrapper.reset()
    wrapper.update(*batch(["a", "a"], [0, 1], [-1.0, 1.0]))
    second = wrapper.compute()

    assert any(name.startswith("channel_a_") for name in second)
    assert not [name for name in second if name.startswith("channel_b_")]


class ScriptedMetric(ScalarMetric):
    """Returns the next canned result on every ``compute``."""

    def __init__(self, results):
        self.results = list(results)

    def update(self, inputs, outputs) -> None:
        """Nothing to accumulate."""

    def compute(self) -> dict[str, float]:
        return dict(self.results.pop(0))

    def reset(self) -> None:
        """Nothing to reset."""


def test_a_group_that_appears_later_joins_the_average():
    """The regular expression is resolved against every result, not the first.

    The inner wrapper discovers its groups from the data, so a channel that
    first shows up in the second epoch used to stay out of the average for the
    rest of the run — and nothing said so.
    """
    inner = ScriptedMetric([
        {"g_0_auc": 1.0},
        {"g_0_auc": 1.0, "g_1_auc": 0.0},
    ])
    wrapper = GroupAverageMetricWrapper(
        inner, avg_over_regulars={"avg_auc": r"g_\d+_auc"}
    )

    assert wrapper.compute()["avg_auc"] == pytest.approx(1.0)
    assert wrapper.compute()["avg_auc"] == pytest.approx(0.5)


def test_a_group_that_leaves_drops_out_of_the_average():
    """Resolution works in both directions."""
    inner = ScriptedMetric([
        {"g_0_auc": 1.0, "g_1_auc": 0.0},
        {"g_0_auc": 1.0},
    ])
    wrapper = GroupAverageMetricWrapper(
        inner, avg_over_regulars={"avg_auc": r"g_\d+_auc"}
    )

    assert wrapper.compute()["avg_auc"] == pytest.approx(0.5)
    assert wrapper.compute()["avg_auc"] == pytest.approx(1.0)


def test_an_average_over_nothing_is_not_reported(caplog):
    """It used to be reported as ``0``, which reads as a collapse."""
    inner = ScriptedMetric([{"unrelated": 1.0}])
    wrapper = GroupAverageMetricWrapper(
        inner, avg_over_regulars={"avg_auc": r"g_\d+_auc"}
    )

    with caplog.at_level(
        logging.WARNING, logger="fmlib.metrics.utils.group_average_wrap"
    ):
        result = wrapper.compute()

    assert "avg_auc" not in result
    assert any("matched nothing" in record.message for record in caplog.records)


def test_an_explicit_group_survives_a_missing_member(caplog):
    """A metric may now omit a name it cannot compute; the average goes on."""
    inner = ScriptedMetric([{"a": 1.0}])
    wrapper = GroupAverageMetricWrapper(inner, groups={"avg": ["a", "b"]})

    with caplog.at_level(
        logging.WARNING, logger="fmlib.metrics.utils.group_average_wrap"
    ):
        result = wrapper.compute()

    assert result["avg"] == pytest.approx(1.0)
    assert any("did not produce" in record.message for record in caplog.records)
