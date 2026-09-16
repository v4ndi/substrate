"""One prediction sequence for all boosting task formulations."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from avatar.automl.progress import log_progress
from avatar.automl.types import ParquetPath, PredictionResult

if TYPE_CHECKING:
    from .base import BaseBoostingTask


def execute_prediction(
    task: BaseBoostingTask,
    test_path: ParquetPath,
    *,
    remote_group_value: Any | None = None,
    include_row_id: bool = False,
    include_group: bool = False,
) -> PredictionResult:
    """Score ``test_path`` with every model the task's layout plans for."""
    task._require_fitted()
    total_started = perf_counter()
    stage_started = perf_counter()
    log_progress("[predict 1/4] loading test parquet")
    frame = task._prediction_input_frame(
        task._normalize_prediction_frame(task._read(test_path)),
        remote_group_value=remote_group_value,
        include_row_id=include_row_id,
    )
    log_progress(
        "[predict 1/4] completed duration_seconds=%.3f test_rows=%d",
        perf_counter() - stage_started,
        frame.height,
    )

    stage_started = perf_counter()
    log_progress("[predict 2/4] validating routing and building prediction branches")
    return_combined = (
        task._internal_config.resolved_model_layout == "global_and_per_group"
    )
    branches: list[tuple[str, PredictionResult]] = []
    layout_frames = task._prediction_layout_frames(frame)
    log_progress(
        "[predict 2/4] completed duration_seconds=%.3f model_layout=%s branches=%d",
        perf_counter() - stage_started,
        task._internal_config.resolved_model_layout,
        len(layout_frames),
    )

    stage_started = perf_counter()
    log_progress(
        "[predict 3/4] scoring prediction branches branches_total=%d inference_device=%s",
        len(layout_frames),
        task._prediction_device,
    )
    for branch_index, (layout, scope_frame) in enumerate(layout_frames, start=1):
        branch_started = perf_counter()
        branch = task._with_model_layout(layout)
        raw_started = perf_counter()
        log_progress(
            "[predict 3/4][branch %d/%d] layout=%s raw scoring started rows=%d",
            branch_index,
            len(layout_frames),
            layout,
            scope_frame.height,
        )
        raw = branch._score_prediction_branch(scope_frame)
        log_progress(
            "[predict 3/4][branch %d/%d] layout=%s raw scoring completed duration_seconds=%.3f prediction_rows=%d",
            branch_index,
            len(layout_frames),
            layout,
            perf_counter() - raw_started,
            scope_frame.height,
        )
        prediction = branch._prediction_result(scope_frame, raw)
        if return_combined or include_group:
            prediction = branch._with_prediction_group(prediction, scope_frame)
        if include_row_id:
            prediction = branch._with_prediction_row_id(prediction, scope_frame)
        branches.append((layout, prediction))
        log_progress(
            "[predict 3/4][branch %d/%d] layout=%s completed duration_seconds=%.3f",
            branch_index,
            len(layout_frames),
            layout,
            perf_counter() - branch_started,
        )
    log_progress(
        "[predict 3/4] completed duration_seconds=%.3f", perf_counter() - stage_started
    )

    stage_started = perf_counter()
    log_progress("[predict 4/4] assembling prediction result")
    result = task._combine_layout_predictions(branches)
    log_progress(
        "[predict 4/4] completed duration_seconds=%.3f total_duration_seconds=%.3f output_rows=%d",
        perf_counter() - stage_started,
        perf_counter() - total_started,
        result.scores.height,
    )
    return result
