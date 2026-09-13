"""The metric contract: ``update`` / ``compute`` / ``reset``.

Metrics come in two kinds, and the difference is in the type, not in a comment:

* :class:`ScalarMetric` produces numbers. ``compute`` returns
  ``{name: value}``, which the trainer logs and feeds to early stopping.
* :class:`ArtifactMetric` produces a file. Subclasses implement ``flush``;
  ``compute`` is written once here, in the base class, and always returns an
  empty dict.

The split exists because the single contract did not hold: four collectors used
to return ``None`` from ``compute``, and the evaluation loop merges that result
into a dict. Now there is nowhere for a ``None`` to come from.

Metrics may also declare **which fields they read** through
:attr:`BaseMetric.required_inputs` and :attr:`BaseMetric.required_outputs`. In a
distributed run the evaluation loop sends ``(batch, output)`` to rank 0, and the
declaration is what lets it send two tensors instead of the whole batch. ``None``
— the default — means "everything on that side", which is always correct and
always the expensive answer.
"""

from __future__ import annotations

import abc
import datetime
import logging
import os

logger = logging.getLogger(__name__)


class BaseMetric(abc.ABC):
    """Abstract base class defining the interface for all metric computations.

    Provides the fundamental structure for metrics that need to:

    1. Accumulate statistics across multiple batches (``update``)
    2. Compute final metric values (``compute``)
    3. Reset internal state between evaluations (``reset``)

    Typical usage pattern::

        metric = ConcreteMetric()
        for batch in dataset:
            predictions = model(batch)
            metric.update(batch, predictions)
        results = metric.compute()
        metric.reset()

    Prefer subclassing :class:`ScalarMetric` or :class:`ArtifactMetric` over this
    class directly — they fix what ``compute`` returns.

    Attributes:
        required_inputs: Keys of ``inputs`` this metric reads, or ``None`` when
            it reads whatever the batch happens to carry. Declaring the keys
            lets a distributed run gather only those, instead of the whole
            batch.
        required_outputs: The same for fields of the model's output dataclass.
    """

    required_inputs: tuple[str, ...] | None = None
    required_outputs: tuple[str, ...] | None = None

    @abc.abstractmethod
    def update(self, inputs, outputs) -> None:
        """Accumulate metric statistics from a single batch.

        Args:
            inputs: Raw input data for the batch (typically unused, but provided
                for reference if the metric needs input features).
            outputs: Model predictions or raw outputs for the batch.

        Note:
            Implementations modify internal state and return nothing. All
            computation is deferred until :meth:`compute`.
        """
        raise NotImplementedError("Method must be implemented by child classes")

    @abc.abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute and return all metrics using accumulated state.

        Returns:
            Dictionary of metric names to their computed values. Never
            ``None`` — a metric with nothing to report returns ``{}``.

        Note:
            Must not modify internal state: calling it twice in a row has to
            give the same answer. Use :meth:`reset` to start over.
        """
        raise NotImplementedError("Method must be implemented by child classes")

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all internal state variables.

        Note:
            Returns the metric to its initial state, as if newly instantiated.
        """
        raise NotImplementedError("Method must be implemented by child classes")


class ScalarMetric(BaseMetric):
    """A metric whose product is numbers.

    Adds nothing to :class:`BaseMetric` but the promise in its name: whatever
    ``compute`` returns goes straight into the log and into early stopping, so
    every value has to be a plain number under a stable name.
    """


class ArtifactMetric(BaseMetric):
    """A metric whose product is a file rather than a number.

    Subclasses implement :meth:`flush` — accumulate, write, forget — and get
    :meth:`compute` for free. ``compute`` writes the tail and returns ``{}``,
    which is what keeps a collector from poisoning the score dict.

    Args:
        path_to_save: Directory for the output files; created if missing, and an
            existing one is not an error.
        prefix: Optional tag in the filename, to tell runs apart.
        output_format: ``parquet`` or ``csv``.

    Raises:
        ValueError: ``output_format`` is neither ``parquet`` nor ``csv``.
    """

    #: Formats :meth:`save_dataframe` knows how to write.
    FORMATS = ("parquet", "csv")

    def __init__(
        self,
        path_to_save: str,
        prefix: str | None = None,
        output_format: str = "parquet",
    ):
        if output_format not in self.FORMATS:
            raise ValueError(
                f"output_format must be one of {self.FORMATS}, got {output_format!r}"
            )
        self.path_to_save = path_to_save
        self.prefix = prefix
        self.output_format = output_format
        # A directory left behind by an earlier run is not a reason to refuse to
        # start: the run stamp below already keeps the two runs' files apart,
        # and failing here killed the job before training even began.
        os.makedirs(self.path_to_save, exist_ok=True)
        self._run_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._part = 0

    def next_path(self) -> str:
        """Path for the next part file.

        Collectors flush every ``save_steps`` batches. Naming the parts by
        timestamp alone meant two flushes inside the same second landed on the
        same name and the second silently overwrote the first; the part counter
        makes the name monotone however fast the flushes come.
        """
        name = [self._run_stamp, f"part-{self._part:05d}"]
        if self.prefix is not None:
            name.append(self.prefix)
        self._part += 1
        return os.path.join(
            self.path_to_save, "_".join(name) + f".{self.output_format}"
        )

    def save_dataframe(self, df) -> str:
        """Write one part file and return its path."""
        path = self.next_path()
        if self.output_format == "csv":
            df.to_csv(path, index=False)
        else:
            # Spark reads datetime64[us], pandas writes datetime64[ns].
            for column in df.select_dtypes(include=["datetime64[ns]"]).columns:
                df[column] = df[column].astype(str)
            df.to_parquet(path, index=False)
        logger.info("predictions written to %s", path)
        return path

    @abc.abstractmethod
    def flush(self) -> None:
        """Write everything accumulated so far and reset.

        Called both from :meth:`compute` and, for collectors that stream, every
        ``save_steps`` batches from ``update``.
        """
        raise NotImplementedError("Method must be implemented by child classes")

    def compute(self) -> dict[str, float]:
        """Write the tail of the collection.

        Returns:
            An empty dict. The product of this metric is the file.
        """
        self.flush()
        return {}
