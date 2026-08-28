import math
import os
import warnings
from glob import glob
from typing import Any, Iterator, Optional

import numpy as np
import torch

from avatar.data.parquet import parquet_num_rows, read_parquet_file
from avatar.data.sampler import BaseSampler


class BaseIterDataset(torch.utils.data.IterableDataset):
    """
    A base iterable dataset class for reading and processing parquet files.

    Args:
        path (str):
            The directory path containing the parquet files.
        read_columns (Optional[list[str]]):
            list of columns to read from the parquet files. If None, all columns are read.
        shuffle_files (bool):
            Whether to shuffle the list of parquet files. Default is True.
        shuffle_pq (bool):
            Whether to shuffle the rows within each parquet file. Default is True.

    Raises:
        AssertionError: If the provided path does not exist or if the folder is empty.
    """

    def __init__(
        self,
        path: str | list,
        read_columns: Optional[list[str]] = None,
        shuffle_files: bool = True,
        shuffle_pq: bool = True,
    ):
        super().__init__()
        self.files = self.collate_files(path)
        if shuffle_files:
            self.shuffle_files()
        assert len(self.files) != 0, f"Find empty folder: {path}"
        self.total_length = sum(map(parquet_num_rows, self.files))
        self.read_columns = (
            list(read_columns) if read_columns is not None else read_columns
        )
        self.shuffle_pq = shuffle_pq

    def collate_files(self, path: str | list) -> list[str]:
        def find_files(path: str):
            assert os.path.exists(path), f"Directory {path} doesn't exist"
            return glob(os.path.join(path, "**/*.parquet"), recursive=True)

        files = []
        if isinstance(path, str):
            files = find_files(path)
        else:
            for p in path:
                files.extend(find_files(p))
        return files

    def shuffle_files(self) -> None:
        """Shuffles the list of parquet files."""
        np.random.shuffle(self.files)

    def files_per_worker(self) -> tuple[int, int]:
        """
        Distributes the files among workers for parallel processing.

        Returns:
            tuple[int, int]: The start and end indices of files for the current worker.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if worker_info.num_workers > len(self.files):
                warnings.warn(
                    f"Number of workers ({worker_info.num_workers}) is greater than "
                    f"number of files ({len(self.files)}). Some workers will not "
                    "receive any work.",
                    RuntimeWarning,
                )

        if worker_info is None:
            iter_start = 0
            iter_end = len(self.files)
        else:
            per_worker = int(
                math.ceil((len(self.files) - 0) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = 0 + worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))

        return iter_start, iter_end

    def __len__(self) -> int:
        """Returns the total number of rows across all parquet files."""
        return self.total_length

    def process(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Processes a single record from the parquet file.

        Args:
            features (dict[str, Any]):
                A dictionary representing a single record.

        Returns:
            dict[str, Any]: The processed record.
        """
        return record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterates over the dataset, yielding processed records.

        Yields:
            dict[str, Any]: A processed record from the dataset.
        """
        pass


class IterDataset(BaseIterDataset):
    """
    A concrete implementation of BaseIterDataset for iterating over parquet files.

    Args:
        path (str): The directory path containing the parquet files.
        read_columns (Optional[list[str]]): list of columns to read from the parquet files.
                                            If None, all columns are read.
        shuffle_files (bool): Whether to shuffle the list of parquet files. Default is False.
        shuffle_pq (bool): Whether to shuffle the rows within each parquet file. Default is True.
        sampler (Optional[BaseSampler]): An optional sampler to use for sampling records.
    """

    def __init__(
        self,
        path: str | list,
        read_columns: Optional[list[str]] = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        sampler: BaseSampler = None,
    ):
        super().__init__(
            path=path,
            read_columns=read_columns,
            shuffle_files=shuffle_files,
            shuffle_pq=shuffle_pq,
        )
        self.sampler = sampler

    def _iter_impl(self) -> Iterator[dict[str, Any]]:
        """
        Iterates over the dataset, yielding processed records.

        Yields:
            dict[str, Any]: A processed record from the dataset.
        """
        iter_start, iter_end = self.files_per_worker()
        for i in range(iter_start, iter_end):
            records = read_parquet_file(
                file=self.files[i], columns=self.read_columns, shuffle=self.shuffle_pq
            )
            for record in records:
                processed_record = self.process(record)
                if processed_record is not None:
                    yield processed_record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.sampler is None:
            return self._iter_impl()
        self.sampler.set_dataset_iterator(self._iter_impl())
        return self.sampler.__iter__()
