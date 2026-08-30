"""Multi-task pipelines over a shared set of experts.

MMoE gives every task its own gate over one pool of experts; PLE splits that
pool into shared and task-specific experts. Both are the same pipeline shape —
embedding, backbone of experts, one head per task, one composite loss — and
differ only in the backbone, which is why ``PLE`` is ``MMoE`` with a different
backbone class.
"""

from .mmoe import PLE, MMoE, MMoEBackbone, PLEBackbone, TaskHead
from .response import MultiTaskResponse

__all__ = [
    "PLE",
    "MMoE",
    "MMoEBackbone",
    "MultiTaskResponse",
    "PLEBackbone",
    "TaskHead",
]
