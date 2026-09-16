"""Path-based calibration lifecycle shared by Response and Uplift tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from avatar.automl.backends.boosting.uplift import UPLIFT_SCORE_COLUMNS
from avatar.automl.calibrators import CalibrationStrategy, calibrator_class
from avatar.automl.config.base import EnvironmentConfig
from avatar.automl.data import ParquetSource
from avatar.automl.exceptions import ConfigError, SchemaError
from avatar.automl.tasks.evaluation import align_prediction_scores, align_scalar_scores
from avatar.automl.types import CalibrationResult, ParquetPath

if TYPE_CHECKING:
    from .base import BaseBoostingTask

_ROW_ID = "__fmlib_calibration_row_id"
_UPLIFT_LEARNERS = {
    "s": ("score_s", "score_s_control", "score_s_treatment"),
    "t": ("score_t", "score_t_control", "score_t_treatment"),
    "x": ("score_x", "score_x_control", "score_x_treatment"),
}


def _required(frame: pl.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        msg = f"{source} is missing required calibration columns: {missing}"
        raise SchemaError(msg)


def _branches(
    frame: pl.DataFrame, source: str, default_layout: str
) -> dict[str, pl.DataFrame]:
    if "model_layout" not in frame.columns:
        if default_layout == "global_and_per_group":
            msg = f"{source} must contain model_layout for global_and_per_group calibration"
            raise SchemaError(msg)
        return {default_layout: frame}
    values = set(frame["model_layout"].unique().to_list())
    if None in values:
        msg = f"{source} contains null model_layout values"
        raise SchemaError(msg)
    unknown = sorted(values - {"global", "per_group"}, key=str)
    if unknown:
        msg = f"{source} contains unsupported model_layout values: {unknown}"
        raise SchemaError(msg)
    return {
        str(key[0] if isinstance(key, tuple) else key): branch
        for key, branch in frame.partition_by(
            "model_layout", as_dict=True, maintain_order=True
        ).items()
    }


@dataclass(frozen=True)
class CalibrationOutput:
    """A fitted calibration plus the state the task commits for it."""

    result: CalibrationResult
    state: Mapping[str, Any]


def execute_calibration(
    task: BaseBoostingTask,
    test_scores_path: ParquetPath,
    calibration_scores_path: ParquetPath,
    calibration_path: ParquetPath,
    strategy: CalibrationStrategy,
    *,
    remote_layout: str | None = None,
    include_row_id: bool = False,
) -> CalibrationOutput:
    """Fit mappings from a stored calibration prediction and transform a stored test prediction."""
    calibrator = calibrator_class(strategy)
    scores = ParquetSource.resolve(test_scores_path).read().with_row_index(_ROW_ID)
    calibration_scores = ParquetSource.resolve(calibration_scores_path).read()
    default_layout = task.config.resolved_model_layout
    if task._task_name == "response":
        truth = task._read(calibration_path)
        task._target(truth)
        calibration = align_scalar_scores(
            calibration_scores,
            truth,
            task._internal_config,
            task._column_mapper,
            operation_name="Calibration",
        )
    else:
        truth = task._normalize_treatment(task._read(calibration_path), required=True)
        task._target(truth)
        task._treatment(truth, require_both=True)
        treatment_column = task._internal_config.treatment_column
        calibration = align_prediction_scores(
            calibration_scores,
            truth,
            task._internal_config,
            task._column_mapper,
            score_columns=UPLIFT_SCORE_COLUMNS,
            truth_columns=(treatment_column,),
            operation_name="Calibration",
        )
    score_branches = _branches(scores, "test prediction", default_layout)
    calibration_branches = _branches(
        calibration, "calibration prediction", default_layout
    )
    if remote_layout is not None:
        if (
            remote_layout not in score_branches
            or remote_layout not in calibration_branches
        ):
            msg = f"Calibration branch {remote_layout!r} is missing from one of the input paths"
            raise SchemaError(msg)
        score_branches = {remote_layout: score_branches[remote_layout]}
        calibration_branches = {remote_layout: calibration_branches[remote_layout]}
    missing_branches = sorted(set(score_branches) - set(calibration_branches))
    if missing_branches:
        msg = f"Calibration prediction is missing model branches: {missing_branches}"
        raise SchemaError(msg)

    target_column = task._internal_config.target_column
    result_parts: list[pl.DataFrame] = []
    states: dict[str, Any] = {}
    for layout, score_frame in score_branches.items():
        reference = calibration_branches[layout]
        if task._task_name == "response":
            _required(score_frame, {"score"}, "test prediction")
            _required(reference, {"score", target_column}, "calibration prediction")
            state = calibrator.fit(
                reference["score"].to_numpy(), reference[target_column].to_numpy()
            )
            result_parts.append(
                score_frame.with_columns(
                    pl.Series("score", state.predict(score_frame["score"].to_numpy()))
                )
            )
            states[layout] = {"score": state.to_dict()}
        else:
            treatment_column = task._internal_config.treatment_column
            _required(score_frame, set(UPLIFT_SCORE_COLUMNS), "test prediction")
            _required(
                reference,
                {*UPLIFT_SCORE_COLUMNS, target_column, treatment_column},
                "calibration prediction",
            )
            treatment = task._binary_values(
                reference[treatment_column], treatment_column
            )
            target = task._binary_values(reference[target_column], target_column)
            output = score_frame.clone()
            branch_state: dict[str, Any] = {}
            for learner, (effect, control, treated) in _UPLIFT_LEARNERS.items():
                branch_state[learner] = {}
                for arm, arm_name, column in (
                    (0, "control", control),
                    (1, "treatment", treated),
                ):
                    mask = treatment == arm
                    mapping = calibrator.fit(
                        reference[column].to_numpy()[mask], target[mask]
                    )
                    output = output.with_columns(
                        pl.Series(column, mapping.predict(output[column].to_numpy()))
                    )
                    branch_state[learner][arm_name] = mapping.to_dict()
                output = output.with_columns(
                    (pl.col(treated) - pl.col(control)).alias(effect)
                )
            task._validate_score_matrix(
                output.select(UPLIFT_SCORE_COLUMNS).to_numpy(), require_x_effect=True
            )
            result_parts.append(output)
            states[layout] = branch_state
    result = pl.concat(result_parts, how="diagonal_relaxed").sort(_ROW_ID)
    if not include_row_id:
        result = result.drop(_ROW_ID)
    return CalibrationOutput(
        CalibrationResult(result, strategy),
        {"task": task._task_name, "calibration_strategy": strategy, "branches": states},
    )


class CalibratableTask:
    """Expose calibration only on task facades that own this lifecycle."""

    def calibrate(
        self,
        test_path: ParquetPath,
        calibration_path: ParquetPath,
        *,
        calibration_strategy: CalibrationStrategy = "beta_calibration",
        env_type: Literal["local", "osiris"] | None = None,
        environment: EnvironmentConfig | Mapping[str, Any] | None = None,
    ) -> CalibrationResult | None:
        """Fit on the stored calibration prediction and transform the stored test prediction.

        Both dataset paths must already have a succeeded :meth:`predict`
        operation. Calibration never invokes prediction itself.
        """
        for name, value in (
            ("test_path", test_path),
            ("calibration_path", calibration_path),
        ):
            if not isinstance(value, str | Path):
                msg = f"{name} must be a parquet path, not {type(value).__name__}"
                raise ConfigError(msg)
        calibrator_class(calibration_strategy)
        return self._operation_runner().calibrate(
            test_path,
            calibration_path,
            calibration_strategy=calibration_strategy,
            env_type=env_type,
            environment=environment,
        )

    def load_calibration(self, test_path: ParquetPath) -> CalibrationResult:
        """Load the calibration result associated exactly with ``test_path``."""
        return self._store.load_calibration(test_path)

    def _execute_calibrate(
        self,
        test_scores_path: ParquetPath,
        calibration_scores_path: ParquetPath,
        calibration_path: ParquetPath,
        calibration_strategy: CalibrationStrategy,
        *,
        remote_layout: str | None = None,
        include_row_id: bool = False,
    ) -> CalibrationOutput:
        return execute_calibration(
            self,
            test_scores_path,
            calibration_scores_path,
            calibration_path,
            calibration_strategy,
            remote_layout=remote_layout,
            include_row_id=include_row_id,
        )

    def _remote_calibration_parts(
        self, test_scores_path: ParquetPath
    ) -> tuple[str, ...]:
        return tuple(
            _branches(
                ParquetSource.resolve(test_scores_path).read(),
                "test prediction",
                self.config.resolved_model_layout,
            )
        )
