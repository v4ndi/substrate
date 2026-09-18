"""The contract every metric in the package has to keep.

These are deliberately parametrised over the whole public surface rather than
written one class at a time: the defects this file exists to catch — a
``compute`` that returns ``None``, a ``compute`` that eats its own accumulator —
were each present in several classes at once, and a per-class test would have
been written for the classes that were already fine.
"""

from __future__ import annotations

import inspect

import pytest

from fmlib import metrics as metrics_module
from fmlib.metrics import ArtifactMetric, BaseMetric, ScalarMetric

PUBLIC_METRICS = [
    getattr(metrics_module, name)
    for name in metrics_module.__all__
    if inspect.isclass(getattr(metrics_module, name))
    and issubclass(getattr(metrics_module, name), BaseMetric)
    and getattr(metrics_module, name) not in (BaseMetric, ScalarMetric, ArtifactMetric)
]


@pytest.mark.parametrize("metric_class", PUBLIC_METRICS, ids=lambda c: c.__name__)
def test_every_metric_is_scalar_or_artifact(metric_class):
    """Subclassing BaseMetric directly leaves what ``compute`` returns undecided."""
    assert issubclass(metric_class, ScalarMetric | ArtifactMetric), (
        f"{metric_class.__name__} inherits BaseMetric directly; pick ScalarMetric "
        "or ArtifactMetric so that the return type of compute() is fixed"
    )


@pytest.mark.parametrize("metric_class", PUBLIC_METRICS, ids=lambda c: c.__name__)
def test_declared_fields_are_names_or_none(metric_class):
    """``required_*`` is a tuple of field names, or None for "everything"."""
    for attribute in ("required_inputs", "required_outputs"):
        declared = getattr(metric_class, attribute)
        if isinstance(declared, property):
            continue  # a wrapper forwarding its inner metric's declaration
        assert declared is None or (
            isinstance(declared, tuple)
            and all(isinstance(name, str) for name in declared)
        ), f"{metric_class.__name__}.{attribute} is {declared!r}"


@pytest.mark.parametrize("metric_class", PUBLIC_METRICS, ids=lambda c: c.__name__)
def test_artifact_metrics_implement_flush_and_not_compute(metric_class):
    """``compute`` belongs to ArtifactMetric; a subclass overriding it can return None."""
    if not issubclass(metric_class, ArtifactMetric):
        return
    assert metric_class.compute is ArtifactMetric.compute, (
        f"{metric_class.__name__} overrides compute(); implement flush() instead "
        "so the empty-dict return stays in one place"
    )
    assert metric_class.flush is not ArtifactMetric.flush


def test_artifact_compute_returns_an_empty_dict(tmp_path):
    """The whole point of the split: a collector can no longer poison the scores."""

    class Collector(ArtifactMetric):
        def __init__(self, path_to_save):
            super().__init__(path_to_save=path_to_save)
            self.flushed = 0

        def update(self, inputs, outputs):
            pass

        def flush(self):
            self.flushed += 1

        def reset(self):
            pass

    metric = Collector(str(tmp_path / "out"))
    scores: dict[str, float] = {}
    scores.update(metric.compute())
    assert scores == {}
    assert metric.flushed == 1


def test_artifact_metric_accepts_an_existing_directory(tmp_path):
    """A directory left by an earlier run must not kill the job before it starts."""

    class Collector(ArtifactMetric):
        def update(self, inputs, outputs):
            pass

        def flush(self):
            pass

        def reset(self):
            pass

    target = tmp_path / "out"
    target.mkdir()
    (target / "leftover.parquet").write_text("from an earlier run")
    Collector(str(target))  # must not raise


def test_artifact_metric_rejects_an_unknown_format(tmp_path):
    class Collector(ArtifactMetric):
        def update(self, inputs, outputs):
            pass

        def flush(self):
            pass

        def reset(self):
            pass

    with pytest.raises(ValueError, match="output_format"):
        Collector(str(tmp_path / "out"), output_format="feather")


def test_part_names_are_monotone_within_a_second(tmp_path):
    """Two flushes inside one second used to land on the same name."""
    import pandas as pd

    class Collector(ArtifactMetric):
        def update(self, inputs, outputs):
            pass

        def flush(self):
            self.save_dataframe(pd.DataFrame({"a": [1]}))

        def reset(self):
            pass

    metric = Collector(str(tmp_path / "out"), prefix="run")
    for _ in range(3):
        metric.flush()

    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert len(written) == 3, written
    assert [name.split("_part-")[1] for name in written] == [
        "00000_run.parquet",
        "00001_run.parquet",
        "00002_run.parquet",
    ]
