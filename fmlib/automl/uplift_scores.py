"""Which uplift score columns a backend family produces, and what they mean.

The boosting backend fits S-, T- and X-learners in one go and reports all ten
columns. TabNN fits one S-Learner, and the constant that used to be a single
tuple would have forced it to invent seven columns it does not have.

So the shape is a function of the backend family. The column *names* and their
order are frozen: they are in saved predictions and in reports.
"""

from __future__ import annotations

__all__ = [
    "ALL_UPLIFT_SCORE_COLUMNS",
    "UPLIFT_LEARNERS",
    "uplift_learners",
    "uplift_score_columns",
]

#: ``learner -> (effect, control, treatment)``. Calibration recalibrates the
#: two arms and recomputes the effect as their difference, which is why the
#: triple travels together.
UPLIFT_LEARNERS: dict[str, tuple[str, str, str]] = {
    "s": ("score_s", "score_s_control", "score_s_treatment"),
    "t": ("score_t", "score_t_control", "score_t_treatment"),
    "x": ("score_x", "score_x_control", "score_x_treatment"),
}

#: Every column the boosting backend emits, in the order predictions store
#: them. ``score_x_propensity`` belongs to no triple: it is the propensity
#: model's own output.
ALL_UPLIFT_SCORE_COLUMNS: tuple[str, ...] = (
    *UPLIFT_LEARNERS["s"],
    *UPLIFT_LEARNERS["t"],
    *UPLIFT_LEARNERS["x"],
    "score_x_propensity",
)

_BY_BACKEND: dict[str, tuple[str, ...]] = {
    "tabnn": UPLIFT_LEARNERS["s"],
}


def uplift_score_columns(backend: str) -> tuple[str, ...]:
    """Return the score columns one backend family produces.

    Args:
        backend: Backend family name.

    Returns:
        The columns, in storage order. Anything other than TabNN gets the full
        S/T/X set, which is what boosting has always produced.
    """
    return _BY_BACKEND.get(backend, ALL_UPLIFT_SCORE_COLUMNS)


def uplift_learners(backend: str) -> dict[str, tuple[str, str, str]]:
    """Return the learner triples one backend family produces."""
    columns = set(uplift_score_columns(backend))
    return {
        name: triple
        for name, triple in UPLIFT_LEARNERS.items()
        if set(triple) <= columns
    }
