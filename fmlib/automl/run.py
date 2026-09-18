"""Remote worker entrypoint for every fmlib AutoML action."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from fmlib.automl.config import (
    BinaryTaskConfig,
    MulticlassTaskConfig,
    RegressionTaskConfig,
    ResponseTaskConfig,
    UpliftTaskConfig,
)
from fmlib.automl.execution import ExecutionContext
from fmlib.automl.result_io import evaluation_payload
from fmlib.automl.tasks import (
    BinaryTask,
    MulticlassTask,
    RegressionTask,
    ResponseTask,
    UpliftTask,
)

_TASK_TYPES = {
    "binary": (BinaryTaskConfig, BinaryTask),
    "response": (ResponseTaskConfig, ResponseTask),
    "regression": (RegressionTaskConfig, RegressionTask),
    "multiclass": (MulticlassTaskConfig, MulticlassTask),
    "uplift": (UpliftTaskConfig, UpliftTask),
}


def _configure_logging(log_path: Path, run_id: str) -> logging.Logger:
    """Configure one stdout and shared-filesystem log carrying the run ID."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s run_id={run_id} %(name)s: %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return logging.getLogger(__name__)


def execute_spec(spec_path: str | Path) -> dict[str, Any]:
    """Execute a resolved train, predict or evaluate run spec."""
    action_started = perf_counter()
    payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    result_path = Path(payload["result_path"])
    logger = _configure_logging(Path(payload["log_path"]), payload["run_id"])
    task_name = payload.get("task", "binary")
    if task_name not in _TASK_TYPES:
        msg = f"Unsupported AutoML task: {task_name!r}"
        raise ValueError(msg)
    config_class, task_class = _TASK_TYPES[task_name]
    config = config_class.from_mapping(payload["config"])
    requested_config = payload.get("requested_config", payload["config"])
    action = payload["action"]
    arguments = payload["payload"]
    logger.info(
        "Starting action=%s task=%s backend=%s engine=%s env_type=%s device=%s "
        "original_pool=%r effective_pool=%r profile=%s num_gpus=%d num_nodes=%d",
        action,
        task_name,
        config.backend,
        config.engine,
        requested_config["env_type"],
        requested_config["device"],
        config.environment.pool,
        config.environment.effective_pool,
        config.environment.resource_profile,
        config.environment.resolved_num_gpus,
        config.environment.resolved_num_nodes,
    )
    try:
        if action == "train":
            artifact_root = Path(arguments["artifact_path"])
            entity_config = config_class.from_mapping(
                arguments.get("entity_config", requested_config)
            )
            task = task_class(entity_config, _entity_path=artifact_root)
            execution = task._execution_view(ExecutionContext.from_config(config))
            training = execution._execute_train(
                arguments["train_path"],
                arguments["valid_path"],
                remote_layout=arguments.get("remote_layout"),
                remote_group_value=arguments.get("remote_group_value"),
                trial=arguments.get("trial"),
            )
            task._adopt(execution)
            artifact = task.save(artifact_root)
            result = {
                "status": "succeeded",
                "artifact_path": str(artifact),
                "best_params": dict(training.best_params),
                "validation_metrics": dict(training.validation_metrics),
            }
            if arguments.get("trial"):
                result["trial"] = dict(arguments["trial"])
        elif action == "predict":
            artifact_root = Path(arguments["artifact_path"])
            parent_layout = config.resolved_model_layout
            remote_layout = arguments["remote_layout"]
            remote_group_value = arguments.get("remote_group_value")
            runtime_config = (
                config
                if config.group_column is None
                else replace(config, model_layout=remote_layout)
            )
            task = task_class(runtime_config, _entity_path=artifact_root)
            task._restore_artifact(artifact_root)
            task._select_remote_prediction_part(remote_layout, remote_group_value)
            execution = task._execution_view(
                ExecutionContext.from_config(runtime_config), prepare_backends=True
            )
            execution._validate_prediction_layout()
            prediction = execution._execute_predict(
                arguments["test_path"],
                remote_group_value=remote_group_value,
                include_row_id=True,
                include_group=parent_layout == "global_and_per_group",
            )
            scores_path = Path(arguments["scores_path"])
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            scores = task._prediction_storage_frame(prediction)
            scores.write_parquet(scores_path)
            result = {
                "status": "succeeded",
                "scores_path": str(scores_path),
                "class_order": prediction.class_order,
            }
        elif action == "evaluate":
            artifact_root = Path(arguments["artifact_path"])
            task = task_class(config, _entity_path=artifact_root)
            task._restore_artifact(artifact_root)
            execution = task._execution_view(
                ExecutionContext.from_config(config), prepare_backends=False
            )
            evaluation = execution._execute_evaluate(
                arguments["test_path"],
                arguments["scores_path"],
                arguments["evaluation_kind"],
                tuple(arguments["metrics"]),
            )
            result = {
                **evaluation_payload(evaluation),
                "status": "succeeded",
                "figure_paths_raw": {
                    name: str(Path(config.output_dir) / f"{name}.png")
                    for name in evaluation.figures_raw
                },
                "figure_paths_calibrated": {
                    name: str(Path(config.output_dir) / f"{name}.png")
                    for name in evaluation.figures_calibrated
                },
            }
        elif action == "calibrate":
            task = task_class(config, _entity_path=Path(config.output_dir))
            output = task._execute_calibrate(
                arguments["test_scores_path"],
                arguments["calibration_scores_path"],
                arguments["calibration_path"],
                arguments["calibration_strategy"],
                remote_layout=arguments["remote_layout"],
                include_row_id=True,
            )
            scores_path = Path(arguments["output_path"])
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            output.result.scores.write_parquet(scores_path)
            result = {
                "status": "succeeded",
                "scores_path": str(scores_path),
                "calibration_strategy": output.result.calibration_strategy,
                "state": output.state,
            }
        else:
            msg = f"Unsupported AutoML action: {action!r}"
            raise ValueError(msg)
    except Exception as exc:
        logger.exception(
            "AutoML action failed at worker stage duration_seconds=%.3f",
            perf_counter() - action_started,
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Finished action=%s duration_seconds=%.3f",
        action,
        perf_counter() - action_started,
    )
    return result


def main() -> None:
    """Parse the one public worker argument and execute its resolved spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Absolute path to run_spec.json")
    args = parser.parse_args()
    execute_spec(args.spec)


if __name__ == "__main__":
    main()
