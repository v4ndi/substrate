from datetime import timedelta
from typing import Any

import accelerate
import hydra
import mlflow
from accelerate import InitProcessGroupKwargs


def init_accelerate(
    accelerate_arguments: dict[str, Any], mlflow_arguments: dict[str, Any] = None
) -> accelerate.Accelerator:
    """Initialize and configure the Hugging Face Accelerator with optional MLflow tracking.

    This function sets up the `accelerate.Accelerator` with the provided arguments and
    optionally logs the configuration to MLflow if MLflow arguments are provided.

    Args:
        accelerate_arguments (Dict[str, Any]): A dictionary containing configuration parameters
            for the `accelerate.Accelerator`.

        mlflow_arguments (Dict[str, Any]): A dictionary containing MLflow tracking parameters.
            If provided, the Accelerator config will be logged to MLflow. Common keys include:
            - `experiment_name` (str): Name of the MLflow experiment.
            - `run_name` (str): Name of the MLflow run.
            - Other supported MLflow parameters (e.g., `tracking_uri`).

    Returns:
        accelerate.Accelerator: An initialized Accelerator object configured with given arguments.

    """
    if mlflow_arguments is not None and "tracking_uri" in mlflow_arguments:
        mlflow.set_tracking_uri(mlflow_arguments["tracking_uri"])
        mlflow_arguments = dict(mlflow_arguments)
        mlflow_arguments.pop("tracking_uri")

    if mlflow_arguments is not None:
        tracker = accelerate.tracking.MLflowTracker(**mlflow_arguments)
    else:
        tracker = None
    if "kwargs_handlers" in accelerate_arguments:
        kwargs_handlers = [
            hydra.utils.instantiate(kwarg)
            for kwarg in accelerate_arguments["kwargs_handlers"]
        ]
        del accelerate_arguments.kwargs_handlers
    else:
        kwargs_handlers = []
    kwargs_handlers.append(InitProcessGroupKwargs(timeout=timedelta(seconds=36000000)))

    accelerator = hydra.utils.instantiate(accelerate_arguments)(
        log_with=tracker,
        kwargs_handlers=kwargs_handlers,
    )
    if mlflow_arguments is not None:
        accelerator.init_trackers(mlflow_arguments["run_name"])

    return accelerator
