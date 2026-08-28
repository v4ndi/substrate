import abc
import datetime
import os


class BaseInference(abc.ABC):
    """Base module for inference

    Args:
        path_to_save: Path to save dataframe
        prefix: Prefix for saving file
            prefix = "first" -> path_to_save/<datetime.now()>_first.csv
            Default: None -> path_to_save/<datetime.now()>.csv
        output_format: str: Output format
            available: "csv", "parquet"
    """

    def __init__(
        self, path_to_save: str, prefix: str = None, output_format: str = "parquet"
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

    @abc.abstractmethod
    def update(self, inputs, outputs) -> None:
        """Accumulate inputs and outputs for inference

        Args:
            inputs: Raw input data for the batch (typically unused, but provided
                   for reference if inference class needs input features)
            outputs: Model predictions or raw outputs for the batch

        Note:
            Implementation should modify internal state but not return values.
            All computation should be deferred until compute() is called.
        """
        raise NotImplementedError("Method must be implemented by child classes")

    @abc.abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute and return all results.


        Note:
            Should not modify internal state. For stateful operations between
            computations, use reset() explicitly.
        """
        raise NotImplementedError("Method must be implemented by child classes")

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all internal state variables.

        Note:
            Should return the class to its initial state, as if newly instantiated.
            Called automatically at the start of update() in some implementations.
        """
        raise NotImplementedError("Method must be implemented by child classes")
