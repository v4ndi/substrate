"""The collate contract shared by every modality."""

from abc import ABC, abstractmethod
from typing import Any


class BaseCollateFn(ABC):
    """Abstract base class for collate functions.

    A collate function is responsible for combining a list of samples into a batch.
    This class defines the interface that all collate functions must implement.
    """

    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Combines a list of samples into a batch.

        Args:
            batch (List[Dict[str, Any]]): A list of samples, where each sample is a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representing the batched data.
        """
        pass

    @staticmethod
    def extract_values(batch: list[dict[str, Any]], column_name: str):
        return [item[column_name] for item in batch]
