"""The metric contract: ``update`` / ``compute`` / ``reset``.

:class:`BaseMetric` is what every metric in this package implements and what
the evaluation loop calls. :class:`BaseInferenceMetric` specialises it for
metrics whose product is a file rather than a number.
"""

import abc
import datetime
import os

import pandas as pd
import torch


class BaseMetric(abc.ABC):
    """Abstract base class defining the interface for all metric computations.

    Provides the fundamental structure for metrics that need to:
    1. Accumulate statistics across multiple batches (update)
    2. Compute final metric values (compute)
    3. Reset internal state between evaluations (reset)

    Child classes must implement all abstract methods to support stateful metric
    computation in iterative/streaming scenarios.

    Typical usage pattern:
        metric = ConcreteMetric()
        for batch in dataset:
            predictions = model(batch)
            metric.update(batch, predictions)
        results = metric.compute()
        metric.reset()

    Methods:
        update: Process a single batch of inputs/outputs to update internal state
        compute: Calculate final metrics from accumulated state
        reset: Clear all accumulated state for fresh evaluation
    """

    @abc.abstractmethod
    def update(self, inputs, outputs) -> None:
        """Accumulate metric statistics from a single batch.

        Args:
            inputs: Raw input data for the batch (typically unused, but provided
                   for reference if metric needs input features)
            outputs: Model predictions or raw outputs for the batch

        Note:
            Implementation should modify internal state but not return values.
            All computation should be deferred until compute() is called.
        """
        raise NotImplementedError("Method must be implemented by child classes")

    @abc.abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute and return all metrics using accumulated state.

        Returns:
            Dictionary of metric names to their computed values. All returned
            values should be scalar floats suitable for logging/aggregation.

        Note:
            Should not modify internal state. For stateful operations between
            computations, use reset() explicitly.
        """
        raise NotImplementedError("Method must be implemented by child classes")

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all internal state variables.

        Note:
            Should return the metric to its initial state, as if newly instantiated.
            Called automatically at the start of update() in some implementations.
        """
        raise NotImplementedError("Method must be implemented by child classes")


class BaseInferenceMetric(BaseMetric):
    """Base class for metrics whose product is a file, not a number.

    Args:
        path_to_save: Path to save dataframe
        prefix: Prefix for saving file
            prefix = "first" -> path_to_save/<datetime.now()>_first.csv
            Default: None -> path_to_save/<datetime.now()>.csv
        output_format: str: Output format
            available: "csv", "parquet"
    """

    def __init__(
        self,
        path_to_save: str,
        prefix: str | None = None,
        output_format: str = "parquet",
    ):
        self.path_to_save = path_to_save
        assert output_format in ["csv", "parquet"], "Wrong output format"
        self.prefix = prefix
        self.output_format = output_format
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save, exist_ok=False)

    def save_dataframe(self, df):
        time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        if self.prefix is not None:
            cur_path = (
                os.path.join(self.path_to_save, time_now)
                + f"_{self.prefix}.{self.output_format}"
            )
        else:
            cur_path = (
                os.path.join(self.path_to_save, time_now) + f".{self.output_format}"
            )
        if self.output_format == "csv":
            df.to_csv(cur_path, index=False)

        elif self.output_format == "parquet":
            df.to_parquet(cur_path, index=False)

        self.reset()
        print(f"predict_saved: {cur_path}")


class ClassificationInferenceMetrics(BaseInferenceMetric):
    """Save per-record classification probabilities during inference.

    Args:
        input_columns_to_save: Columns of ``inputs`` to carry into the output
            alongside the prediction — typically an id and the target.
        path_to_save: Directory for the output file; created if missing.
        classification_type: Currently only ``binary`` is implemented.
        prefix: Optional suffix in the filename, to tell runs apart.
        output_format: ``parquet`` or ``csv``.

    Note:
        Everything is held in memory until :meth:`compute`, unlike the
        campaign collectors, which flush every ``save_steps`` batches.
    """

    def __init__(
        self,
        input_columns_to_save: list[str],
        path_to_save: str,
        classification_type: str = "binary",
        prefix: str | None = None,
        output_format: str = "parquet",
    ):
        super().__init__(
            path_to_save=path_to_save, prefix=prefix, output_format=output_format
        )
        self.input_columns_to_save = input_columns_to_save
        self.preds = []
        self.classification_type = classification_type

    def update(self, inputs, outputs):
        self.preds.append({
            k: v.detach().contiguous().cpu().tolist()
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
            if k in self.input_columns_to_save
        })
        if self.classification_type == "binary":
            if outputs.logits.dim() == 1:
                predicted = (
                    torch.nn.functional.sigmoid(outputs.logits)
                    .detach()
                    .contiguous()
                    .cpu()
                    .tolist()
                )
            else:
                predicted = (
                    torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]
                    .detach()
                    .contiguous()
                    .cpu()
                    .tolist()
                )

        self.preds[-1]["predicted"] = predicted

    def compute(self):
        merged_preds = {key: [] for key in self.preds[0].keys()}
        for pred in self.preds:
            for key, value in pred.items():
                merged_preds[key].extend(value)

        self.save_dataframe(pd.DataFrame(merged_preds))

    def reset(self):
        self.preds = []
