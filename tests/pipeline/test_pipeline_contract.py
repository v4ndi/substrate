"""The contract every pipeline in the package has to keep.

Parametrised over the whole public surface rather than written one class at a
time, because the defects it exists to catch were never about one class: a
``forward`` that made ``targets`` mandatory killed inference for
``SupervisedLearner``, and the same mistake was one line away in every other
pipeline. The rules themselves are stated on
:class:`~avatar.pipeline.base.BasePipeline`.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from avatar import pipeline as pipeline_module
from avatar.data.tabular.batch import TabularBatch
from avatar.nn.embedding import TabularEmbedding
from avatar.nn.tabular import TabularTransformer
from avatar.pipeline import BasePipeline

PIPELINES = [
    getattr(pipeline_module, name)
    for name in pipeline_module.__all__
    if inspect.isclass(getattr(pipeline_module, name))
    and issubclass(getattr(pipeline_module, name), BasePipeline)
    and getattr(pipeline_module, name) is not BasePipeline
]

N_CAT, N_NUM, WIDTH, VOCAB, RECORDS = 3, 5, 16, 40, 8


def test_the_scan_actually_finds_pipelines():
    """Guard against the parametrisation silently collecting nothing."""
    assert len(PIPELINES) >= 2


@pytest.mark.parametrize("pipeline_class", PIPELINES, ids=lambda c: c.__name__)
def test_forward_accepts_unknown_batch_keys(pipeline_class):
    """The trainer calls ``model(**batch)``, so a spare column is not an error."""
    parameters = inspect.signature(pipeline_class.forward).parameters.values()
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters), (
        f"{pipeline_class.__name__}.forward has no **kwargs; a batch carrying "
        "one column it does not read would raise TypeError"
    )


@pytest.mark.parametrize("pipeline_class", PIPELINES, ids=lambda c: c.__name__)
def test_targets_are_optional(pipeline_class):
    """Inference has no labels."""
    targets = inspect.signature(pipeline_class.forward).parameters.get("targets")
    assert targets is not None, f"{pipeline_class.__name__}.forward takes no targets"
    assert targets.default is None, (
        f"{pipeline_class.__name__}.forward makes targets mandatory; inference "
        "passes none"
    )


@pytest.mark.parametrize("pipeline_class", PIPELINES, ids=lambda c: c.__name__)
def test_required_inputs_are_named_parameters(pipeline_class):
    """A declared input the signature does not accept could never arrive."""
    declared = pipeline_class.required_inputs
    assert isinstance(declared, tuple) and all(isinstance(n, str) for n in declared), (
        f"{pipeline_class.__name__}.required_inputs must be a tuple of names"
    )
    accepted = inspect.signature(pipeline_class.forward).parameters
    missing = [name for name in declared if name not in accepted]
    assert not missing, (
        f"{pipeline_class.__name__} requires {missing}, which forward does not name"
    )


@pytest.mark.parametrize("pipeline_class", PIPELINES, ids=lambda c: c.__name__)
def test_required_inputs_have_no_default(pipeline_class):
    """Anything with a default is optional by definition, so it is not required."""
    accepted = inspect.signature(pipeline_class.forward).parameters
    with_defaults = [
        name
        for name in pipeline_class.required_inputs
        if accepted[name].default is not inspect.Parameter.empty
    ]
    assert not with_defaults, (
        f"{pipeline_class.__name__} declares {with_defaults} required, but "
        "forward gives them defaults; one of the two is wrong"
    )


def batch() -> TabularBatch:
    generator = torch.Generator().manual_seed(0)
    return TabularBatch(
        cat_features=torch.randint(0, VOCAB, (RECORDS, N_CAT), generator=generator),
        num_features=torch.randn(RECORDS, N_NUM, generator=generator),
    )


def build(pipeline_class, extra_tokens: int):
    return pipeline_class(
        embedding=TabularEmbedding(
            num_numerical_features=N_NUM, hidden_size=WIDTH, vocab_size=VOCAB
        ),
        tabular_encoder=TabularTransformer(
            hidden_size=WIDTH, num_heads=2, num_layers=1
        ),
        aggregation_config={
            "name": "linear",
            "num_features": N_CAT + N_NUM + extra_tokens,
            "emb_dim": WIDTH,
        },
    )


@pytest.mark.parametrize("pipeline_class", PIPELINES, ids=lambda c: c.__name__)
def test_scores_a_batch_with_no_targets(pipeline_class):
    """The rule that broke inference, checked by actually running it."""
    extra = {"is_treat": torch.zeros(RECORDS, dtype=torch.long)}
    inputs = {
        name: extra[name] for name in pipeline_class.required_inputs if name in extra
    }
    model = build(pipeline_class, extra_tokens=len(inputs))
    model.eval()

    with torch.no_grad():
        output = model(tab_features=batch(), **inputs)

    assert hasattr(output, "loss"), "the trainer reads .loss and nothing else"


@pytest.mark.parametrize("pipeline_class", PIPELINES, ids=lambda c: c.__name__)
def test_missing_inputs_are_reported_by_name(pipeline_class):
    model = build(pipeline_class, extra_tokens=len(pipeline_class.required_inputs) - 1)
    assert model.missing_inputs({}) == list(pipeline_class.required_inputs)
    supplied = dict.fromkeys(pipeline_class.required_inputs, object())
    assert model.missing_inputs(supplied) == []


# -- the deprecated import path ----------------------------------------------


def test_the_old_uplift_path_still_resolves():
    """Forty-eight configs name it, several of them records of production runs."""
    import avatar.pipeline.uplift as old

    with pytest.warns(DeprecationWarning, match="has moved to"):
        assert old.SLearner is not None

    from avatar.pipeline.tabular import SLearner as moved

    with pytest.warns(DeprecationWarning):
        assert old.SLearner is moved


def test_the_old_interaction_path_still_resolves():
    import avatar.pipeline.uplift.treatment_interaction as old
    from avatar.pipeline.tabular.interaction import IgnoreTreatmentInteraction

    with pytest.warns(DeprecationWarning, match="has moved to"):
        assert old.IgnoreTreatmentInteraction is IgnoreTreatmentInteraction


def test_the_old_path_does_not_invent_names():
    import avatar.pipeline.uplift as old

    with pytest.raises(AttributeError):
        old.__getattr__("SomethingThatNeverExisted")
