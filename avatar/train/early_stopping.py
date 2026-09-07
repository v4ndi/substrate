"""Metric-plateau detection, unchanged in behaviour from ``avatar.train_utils``."""

from __future__ import annotations


class EarlyStopping:
    """Implements early stopping mechanism for training processes.

    Monitors a specified metric and stops training when the metric hasn't improved
    for a given number of consecutive evaluations (patience). Supports both
    maximization and minimization objectives.

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
            ValueError: If strategy is neither "max" nor "min".
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

    def __call__(self, scores):
        """Evaluates whether to stop training based on current metric scores.

        Args:
            scores (dict): Dictionary containing metric values from current evaluation.

        Returns:
            None: Updates internal state (counter, best_score, early_stop).

        Raises:
            AssertionError: If main_metric is not found in scores dictionary.
        """
        assert self.main_metric in scores, (
            f"Not found metric: {self.main_metric} in scores"
        )
        score = scores[self.main_metric]
        if self.strategy == "min":
            score *= -1
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
