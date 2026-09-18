"""Binary response-modelling task facade."""

from pathlib import Path

from fmlib.automl.config import ResponseTaskConfig

from .binary import BinaryTask
from .calibration import CalibratableTask


class ResponseTask(CalibratableTask, BinaryTask):
    """Train and evaluate binary response models with optional treatment."""

    _config_class = ResponseTaskConfig
    _task_name = "response"
    _artifact_directory = "response_model"

    def __init__(self, config: ResponseTaskConfig, *, _entity_path: Path | None = None):
        super().__init__(config, _entity_path=_entity_path)
