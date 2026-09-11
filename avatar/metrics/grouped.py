"""The shape shared by every metric that scores a population group by group.

Uplift, response and regression metrics are the same program with two holes in
it: buffer a batch of per-record columns, concatenate them at the end of the
epoch, split the population by ``group`` and by ``split_type`` (``calib``
against ``test``), score each slice and name the numbers
``{split}_group_{group}_{metric}``. What differs is only which columns a metric
keeps and how it scores a slice — the two hooks below.
"""

import abc
import logging
import os

import numpy as np
import pandas as pd

from avatar.metrics.base import ScalarMetric

logger = logging.getLogger(__name__)

#: The name of the slice the calibrators are fitted on, and the fallback when a
#: batch carries no ``split_type`` at all.
CALIB_SPLIT = "calib"

#: The held-out slice. When it is present its numbers are the ones that count.
TEST_SPLIT = "test"


def group_prefixed(scores: dict, split: str, group) -> dict:
    """Name a slice's scores after the split and group they came from."""
    return {f"{split}_group_{group}_{key}": value for key, value in scores.items()}


class GroupedPredictionMetric(ScalarMetric):
    """Buffer per-record predictions, score each ``group`` separately.

    Subclasses fill in two hooks:

    * :meth:`collect` turns one batch into a dict of numpy columns. ``y_true``
      must be among them — it is what the population size is read from. A
      column whose value is ``None`` in the first batch is dropped, which is
      how the optional ones (``group``, ``split_type``, ``epk_id``) disappear
      when the data does not carry them.
    * :meth:`score_group` scores one group and returns ``(named scores, the
      value that feeds ``mean_{main_metric}``)``. Returning ``None`` for the
      second means this group has nothing to contribute to the mean.

    Args:
        save_submit_path: Directory to write the merged per-record frame into.
            ``None`` writes nothing.
        main_metric: Which of the produced numbers is averaged over groups into
            ``mean_{main_metric}``.
    """

    def __init__(
        self,
        save_submit_path: str | None = None,
        main_metric: str | None = None,
    ):
        self.preds: list[dict] = []
        self.save_submit = save_submit_path
        self.main_metric = main_metric

    @abc.abstractmethod
    def collect(self, inputs, outputs) -> dict:
        """Turn one batch into the columns this metric scores."""

    @abc.abstractmethod
    def score_group(
        self, merged: dict, group, calib_mask: np.ndarray, test_mask: np.ndarray
    ) -> tuple[dict, float | None]:
        """Score one group; return its named scores and its main-metric value."""

    def prepare(self, merged: dict) -> None:
        """Hook: add derived columns before the group loop. Default: nothing."""

    def aggregate(self, result: dict, main_values: list[float]) -> None:
        """Hook: fold the per-group main values into summary numbers.

        Groups that produced nothing leave ``main_values`` empty, and then no
        mean is reported at all — a missing key says "no data" far more clearly
        than the ``nan`` that averaging an empty list would give.
        """
        if main_values:
            result[f"mean_{self.main_metric}"] = float(np.mean(main_values))

    def update(self, inputs, outputs) -> None:
        """Buffer one batch."""
        self.preds.append(self.collect(inputs, outputs))

    def compute(self) -> dict[str, float]:
        """Score every group and return the flat name → number mapping."""
        merged = self.merge()
        self.prepare(merged)

        result: dict[str, float] = {}
        main_values: list[float] = []
        for group in np.unique(merged["group"]):
            in_group = merged["group"] == group
            scores, main_value = self.score_group(
                merged,
                group,
                in_group & (merged["split_type"] == CALIB_SPLIT),
                in_group & (merged["split_type"] == TEST_SPLIT),
            )
            result.update(scores)
            if main_value is not None:
                main_values.append(main_value)

        self.aggregate(result, main_values)
        self.write_submit(merged)
        return result

    def reset(self) -> None:
        """Drop the buffered batches."""
        self.preds = []

    def merge(self) -> dict:
        """Concatenate the buffered batches and fill in the optional columns."""
        merged = {
            key: np.concatenate([batch[key] for batch in self.preds])
            for key in self.preds[0]
            if self.preds[0][key] is not None
        }
        population = merged["y_true"].shape[0]
        if "group" not in merged:
            merged["group"] = np.full(population, -1, dtype=np.int32)
        if "split_type" not in merged:
            merged["split_type"] = np.full(population, CALIB_SPLIT)
        return merged

    def write_submit(self, merged: dict) -> None:
        """Write the merged frame to ``save_submit_path``, if one was given."""
        if self.save_submit is None:
            return
        os.makedirs(self.save_submit, exist_ok=True)
        submit_path = os.path.join(self.save_submit, "predict.parquet")
        pd.DataFrame(merged).to_parquet(submit_path, index=False)
        logger.info("submit written to %s", submit_path)
