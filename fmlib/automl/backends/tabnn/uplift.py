"""The TabNN uplift adapter: one S-Learner, scored twice.

This is a class of its own while the four supervised tasks share one, because
what differs is a **protocol**, not a value: one training pass with each
record's real treatment, two scoring passes with everything forced to treated
and then to control, and a matrix out rather than a vector.

Only the S-Learner. T- and X-metalearners are out of scope, so an uplift TabNN
model reports three columns where boosting reports ten -- which is why the
score columns became a function of the backend family.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fmlib.automl.data.schema import FeatureSchema
from fmlib.automl.uplift_scores import uplift_score_columns

from .base import TabNNBackend, _stack

__all__ = ["UpliftTabNNBackend"]


class UpliftTabNNBackend(TabNNBackend):
    """A fitted S-Learner, and the two passes that turn it into an uplift."""

    @property
    def learner_params(self) -> dict[str, dict]:
        """The per-learner parameter map the uplift task reports.

        Boosting fills it with three learners; there is one here, and calling
        it ``s`` is what makes the report read the same either way.
        """
        return {"s": dict(self.params)}

    @property
    def learner_metrics(self) -> dict[str, float]:
        """The per-learner validation metric, for the same reason."""
        return {"s": float(self.validation_metric)}

    def predict_score(self, frame: pl.DataFrame, schema: FeatureSchema) -> np.ndarray:
        """Score a frame and return ``(n, 3)``: effect, control, treated.

        The model runs both arms itself in evaluation mode, so the difference
        is its own rather than one reconstructed here -- which is what keeps
        the effect exactly equal to treated minus control, as the task's
        validation requires.

        Args:
            frame: Rows to score, canonical names, embeddings still vectors.
            schema: The fitted schema, for the contract.

        Returns:
            One row per input row, columns in
            :func:`~fmlib.automl.uplift_scores.uplift_score_columns` order.
        """
        import pyarrow.dataset as ds
        import torch

        from fmlib.data.tabular.batch import TabularBatch

        width = len(uplift_score_columns("tabnn"))
        if frame.height == 0:
            return np.zeros((0, width), dtype=np.float64)

        encoded = self._preprocessor().transform(
            ds.dataset(self._physical(frame).to_arrow()),
            None,
            identity_cols=list(self.hidden_states),
            output="packed",
        )
        table = pl.from_arrow(encoded)
        model = self._model()

        effects: list[np.ndarray] = []
        controls: list[np.ndarray] = []
        treated: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, table.height, self.batch_size):
                chunk = table.slice(start, self.batch_size)
                batch = TabularBatch(
                    cat_features=_stack(chunk, "cat_features", torch.long),
                    num_features=_stack(chunk, "num_features", torch.float32),
                    hidden_states={
                        name: _stack(chunk, name, torch.float32)
                        for name in self.hidden_states
                    }
                    or None,
                )
                if batch.hidden_states is not None:
                    batch._hidden_states = {
                        name: torch.where(value.isnan(), 0, value)
                        for name, value in batch.hidden_states.items()
                    }
                # The treatment values passed here are ignored by the two
                # scoring passes -- the model forces each arm itself -- but the
                # input is part of the signature, so it is supplied.
                is_treat = torch.zeros(chunk.height, dtype=torch.long)
                output = model(tab_features=batch, is_treat=is_treat)
                effects.append(_host(output.uplift))
                controls.append(_host(output.control_probs))
                treated.append(_host(output.treatment_probs))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.column_stack([
            np.concatenate(effects),
            np.concatenate(controls),
            np.concatenate(treated),
        ])


def _host(value) -> np.ndarray:
    """One arm of a scoring pass, as a flat float64 array on the host."""
    import torch

    if value is None:
        msg = (
            "The S-Learner returned no scoring passes; an uplift model has to be "
            "scored in evaluation mode"
        )
        raise ValueError(msg)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value).reshape(-1).astype(np.float64)
