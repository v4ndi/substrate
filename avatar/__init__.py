"""Foundation models for tabular and event-sequence customer data.

Start at ``docs/getting_started.md``; the config schema every run is described
by is in ``docs/configuration/schema.md``.
"""

from . import losses
from ._version import __version__
from .training_arguments import TrainingArguments

__all__ = ["TrainingArguments", "__version__", "losses"]
