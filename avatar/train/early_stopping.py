"""Metric-plateau detection."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Implements early stopping mechanism for training processes.

    Monitors a specified metric and stops training when the metric hasn't improved
    for a given number of consecutive evaluations (patience). Supports both
    maximization and minimization objectives.

    An evaluation that does not produce a comparable number — the metric is
    missing, ``None`` or ``nan`` — counts as **no improvement**: the record
    stands and patience ticks. A ``nan`` used to be treated as an improvement,
    because every comparison against it is false: it overwrote the best score,
    reset the counter and, since the checkpoint callback saves whenever the
    counter is zero, also wrote a checkpoint labelled best. One undefined epoch
    was enough for a later collapse to go unnoticed.

    Attributes:
        main_metric (str): Name of the metric to monitor for early stopping.
        patience (int): Number of evaluations to wait before stopping when no improvement.
        delta (float): Minimum change in monitored metric to qualify as improvement.
        counter (int): Current count of evaluations without improvement.
        best_score (float): Best score observed so far.
        early_stop (bool): Flag indicating whether to stop training.
        strategy (str): Optimization strategy - "max" to maximize or "min" to minimize.

    Example:
        >>> early_stopping = EarlyStopping(main_metric="val_loss", patience=5, strategy="min")
        >>> for epoch in range(100):
        ...     # Training happens here
        ...     val_scores = {"val_loss": 0.2, "val_acc": 0.95}
        ...     early_stopping(val_scores)
        ...     if early_stopping.early_stop:
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(
        self,
        main_metric: str,
        patience: int = 10,
        delta: int = 0,
        strategy: str = "max",
    ):
        """Initializes the EarlyStopping instance.

        Args:
            main_metric (str): Name of the metric to monitor.
            patience (int, optional): Number of evaluations to wait before stopping when
                no improvement occurs. Defaults to 10.
            delta (float, optional): Minimum change in monitored metric to qualify as
                improvement. Defaults to 0.
            strategy (str, optional): Optimization strategy - "max" to maximize or
                "min" to minimize the metric. Defaults to "max".

        Raises:
            AssertionError: If strategy is neither "max" nor "min".
        """
        assert strategy in ["min", "max"], (
            f"Unsupported value for strategy: {strategy=}"
        )
        self.main_metric = main_metric
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.strategy = strategy

    def read_score(self, scores) -> float | None:
        """The monitored value, or ``None`` when this evaluation has none.

        A metric that cannot be computed omits its key rather than reporting
        ``nan``, so both cases arrive here and both are reported in the log —
        a monitored name that never appears is usually a typo in the config.
        """
        if self.main_metric not in scores:
            logger.warning(
                "%s is not among the reported metrics (%s); this evaluation counts "
                "as no improvement",
                self.main_metric,
                ", ".join(sorted(scores)[:5]) or "none at all",
            )
            return None

        value = scores[self.main_metric]
        try:
            comparable = math.isfinite(value)
        except (TypeError, ValueError):
            comparable = False
        if not comparable:
            logger.warning(
                "%s is %r and cannot be compared; this evaluation counts as no "
                "improvement and the best score of %r stands",
                self.main_metric,
                value,
                self.best_score,
            )
            return None
        return float(value)

    def register_no_improvement(self) -> None:
        """Tick patience, and stop the run once it runs out."""
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

    def __call__(self, scores):
        """Evaluates whether to stop training based on current metric scores.

        Args:
            scores (dict): Dictionary containing metric values from current evaluation.

        Returns:
            None: Updates internal state (counter, best_score, early_stop).
        """
        score = self.read_score(scores)
        if score is None:
            self.register_no_improvement()
            return

        if self.strategy == "min":
            score *= -1
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.register_no_improvement()
        else:
            self.best_score = score
            self.counter = 0
