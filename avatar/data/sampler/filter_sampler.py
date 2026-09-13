"""Samplers that filter records by column values."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from avatar.data.sampler.base_sampler import BaseSampler


class ColumnFilterSampler(BaseSampler):
    """Filter samples by a column with optional min/max inclusive bounds.

    This sampler yields only items whose `column` value satisfies the given
    lower/upper bound condition. Bounds are inclusive, and either bound may be
    omitted by passing `None`. The values can be of any comparable type as
    long as the column values support the comparison operators with the given
    bounds.

    Parameters
    ----------
    column : str
        The key or field name in the dataset sample used for filtering.
    min_value : Any or None
        Inclusive lower bound. Use `None` to disable the lower bound.
    max_value : Any or None
        Inclusive upper bound. Use `None` to disable the upper bound.
    allowed_values: tuple[any]
        List of allowed values

    Raises:
    ------
    AssertionError
        If both `min_value`, `max_value`, `allowed_values` are `None`.
    AssertionError
        If both bounds are provided and `min_value` is not less than or equal to `max_value`.
    ValueError
        If both bounds are `None` when evaluating the condition.

    Notes:
    -----
    - This sampler assumes each yielded item from `self.dataset_iterator` is a
      mapping or object accessible via `sample[column]`. Adjust access if your
      dataset returns tuples or objects.
    - If integrating with PyTorch `DataLoader`, typical samplers yield indices
      rather than sample payloads; adapt `__iter__` accordingly in that case.

    Examples:
    --------
    Filter by an integer range:
        min_value=10, max_value=20 includes values in [10, 20].

    Filter with only upper bound:
        min_value=None, max_value='m' includes values <= 'm' for comparable strings.
    """

    def __init__(
        self, column: str, min_value=None, max_value=None, allowed_values=None
    ):
        super().__init__()
        if min_value is not None and max_value is not None:
            assert min_value <= max_value, (
                "min_value must be less or equal than max_value"
            )
        assert (
            min_value is not None or max_value is not None or allowed_values is not None
        ), "min_value, max_value and allowed_values cannot be None simultaneously"
        assert (min_value is not None or max_value is not None) ^ (
            allowed_values is not None
        ), "Cannot specify both range (min/max_value) and explicit allowed_values"

        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = (
            tuple(allowed_values) if allowed_values is not None else None
        )
        self.column = column

    def _check_condition(self, data):
        if self.allowed_values is not None:
            return data in self.allowed_values
        elif self.min_value is not None and self.max_value is not None:
            return self.min_value <= data <= self.max_value
        elif self.min_value is not None:
            return data >= self.min_value
        elif self.max_value is not None:
            return data <= self.max_value
        else:
            # This should never happen due to assertions in __init__
            raise RuntimeError("Invalid validator state: all constraints are None")

    def __iter__(self):
        for data in self.dataset_iterator:
            if self._check_condition(data[self.column]):
                yield data
            else:
                continue


class MultiTaskColumnsFilterSampler(BaseSampler):
    """Filter on several columns at once, for multi-task datasets.

    Special-cased by the datasets: its decisions define which records exist at
    all, so the scan applies it once and it does not run again during iteration.
    """

    def __init__(
        self,
        task_name_column: str,
        filters: dict[str, dict[str, Any]],
    ):
        """Filter samples using task-specific column rules.

        Rules support exact equality (``eq``), membership (``in``), and
        inclusive lower/upper bounds (``min`` and ``max``). Two-element
        ``[min, max]`` rules remain supported for backwards compatibility.

        Args:
            task_name_column: Column naming the task each record belongs to;
                its value selects which entry of ``filters`` applies.
            filters: dict where
                key = task name (str)
                    key = column name (str)
                    value = a rule containing eq, in, min, and/or max

        Examples:
            MultiTaskColumnsFilterSampler(
                task_name_column="task",
                filters={
                    "classification": {
                        "age": {"min": 18, "max": 65},
                        "country": {"in": ["DE", "FR"]},
                    }
                },
            )
        """
        super().__init__()
        self.task_name_column = task_name_column
        self.filters = filters

    @property
    def filter_columns(self) -> set[str]:
        """Columns required to evaluate :meth:`accepts`."""
        columns = {self.task_name_column}
        if self.filters is not None:
            for task_filters in self.filters.values():
                if task_filters is not None:
                    columns.update(task_filters.keys())
        return columns

    @staticmethod
    def _parse_rule(rule) -> tuple[str, Any]:
        """Normalize and validate a rule for scalar and vectorized evaluation."""
        if isinstance(rule, Mapping) or OmegaConf.is_dict(rule):
            if "eq" in rule:
                return "eq", rule["eq"]

            if "in" in rule:
                return "in", rule["in"]

            min_value = rule.get("min")
            max_value = rule.get("max")
        elif (isinstance(rule, Sequence) or OmegaConf.is_list(rule)) and not isinstance(
            rule, str | bytes
        ):
            if len(rule) != 2:
                raise ValueError("legacy range filters must contain [min, max]")
            min_value, max_value = rule
        else:
            raise TypeError(f"Unsupported filter rule: {rule!r}")

        if min_value is None and max_value is None:
            raise ValueError(f"Filter must specify 'eq', 'in', 'min', or 'max': {rule}")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError(f"min ({min_value!r}) must be <= max ({max_value!r})")
        return "range", (min_value, max_value)

    @classmethod
    def _check_condition(cls, data: Any, rule) -> bool:
        kind, value = cls._parse_rule(rule)
        if kind == "eq":
            return data == value
        if kind == "in":
            return data in value

        min_value, max_value = value
        if min_value is not None and data < min_value:
            return False
        if max_value is not None and data > max_value:
            return False
        return True

    @classmethod
    def _condition_mask(cls, data: np.ndarray, rule) -> np.ndarray:
        """Vectorized equivalent of :meth:`_check_condition`."""
        kind, value = cls._parse_rule(rule)
        if kind == "eq":
            return np.asarray(data == value, dtype=bool)
        if kind == "in":
            return np.asarray(np.isin(data, list(value)), dtype=bool)

        min_value, max_value = value
        mask = np.ones(len(data), dtype=bool)
        if min_value is not None:
            mask &= data >= min_value
        if max_value is not None:
            mask &= data <= max_value
        return mask

    def build_mask(self, data) -> np.ndarray:
        """Build a vectorized mask using the same task rules as :meth:`accepts`.

        ``data`` may be a mapping of column names to arrays or a PyArrow table.
        Tasks without configured rules remain accepted.
        """

        def column_values(column: str) -> np.ndarray:
            values = data[column]
            if hasattr(values, "to_numpy"):
                try:
                    values = values.to_numpy(zero_copy_only=False)
                except TypeError:
                    values = values.to_numpy()
            return np.asarray(values)

        tasks = column_values(self.task_name_column)
        accepted = np.ones(len(tasks), dtype=bool)
        if self.filters is None:
            return accepted

        for task_name, task_filters in self.filters.items():
            if task_filters is None:
                continue
            task_rows = np.asarray(tasks == task_name, dtype=bool)
            if not task_rows.any():
                continue
            task_accepted = np.ones(len(tasks), dtype=bool)
            for column, rule in task_filters.items():
                task_accepted &= self._condition_mask(column_values(column), rule)
            accepted[task_rows] = task_accepted[task_rows]
        return accepted

    def accepts(self, data: dict[str, Any]) -> bool:
        """Return whether a row belongs to the sampler's logical dataset."""
        task_filters = (
            None
            if self.filters is None
            else self.filters.get(data[self.task_name_column])
        )
        if task_filters is None:
            return True
        return all(
            self._check_condition(data[col], rule) for col, rule in task_filters.items()
        )

    def __iter__(self):
        for data in self.dataset_iterator:
            if self.accepts(data):
                yield data


OmegaConf.register_new_resolver(
    "to_datetime64",
    lambda date_str: np.datetime64(date_str),
)
