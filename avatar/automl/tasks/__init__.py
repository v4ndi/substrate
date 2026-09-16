"""User-facing AutoML task facades."""

from .binary import BinaryTask
from .multiclass import MulticlassTask
from .regression import RegressionTask
from .response import ResponseTask
from .uplift import UpliftTask

__all__ = ["BinaryTask", "MulticlassTask", "RegressionTask", "ResponseTask", "UpliftTask"]
