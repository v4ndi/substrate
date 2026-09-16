"""Immutable operation settings, separate from persistent task configuration."""

from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Mapping

from avatar.automl.config.base import BaseTaskConfig
from avatar.automl.data import CanonicalColumnMapper


class _FrozenDict(dict):
    """Keep config mappings immutable while retaining JSON/asdict support."""

    def _immutable(self, *args, **kwargs):
        msg = "Execution settings are immutable"
        raise TypeError(msg)

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self


def _freeze(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        result = copy(value)
        for field in fields(value):
            object.__setattr__(result, field.name, _freeze(getattr(value, field.name)))
        return result
    if isinstance(value, Mapping):
        return _FrozenDict((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ExecutionContext:
    """Snapshot effective public and canonical settings for one operation."""

    config: BaseTaskConfig
    internal_config: BaseTaskConfig

    @classmethod
    def from_config(cls, config: BaseTaskConfig) -> "ExecutionContext":
        mapper = CanonicalColumnMapper.from_config(config)
        return cls(_freeze(config), _freeze(mapper.normalize_config(config)))

    def derive(self, **updates: Any) -> "ExecutionContext":
        """Create validated settings for a branch without changing its parent."""
        return type(self).from_config(replace(self.config, **updates))
