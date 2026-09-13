"""Uplift pipelines: model the effect of treatment, not the outcome.

The S-Learner approach — one model, with the treatment flag fed in as a
feature. How that flag reaches the representation is pluggable
(:mod:`~avatar.pipeline.uplift.treatment_interaction`), which is what lets the
same backbone serve concat, sum and elementwise variants.
"""

from avatar.pipeline.uplift.treatment_interaction import IgnoreTreatmentInteraction

from .s_learner import SLearner

__all__ = [
    "IgnoreTreatmentInteraction",
    "SLearner",
]
