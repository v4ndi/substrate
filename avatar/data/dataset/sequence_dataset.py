import concurrent
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from tqdm import tqdm

from avatar.data.dataset.parquet_dataset import IterDataset
from avatar.data.dataset.tabular_dataset import TabularDataset


class EventSequenceDataset(IterDataset):
    """A dataset for processing and filtering event sequence data with optional tabular features.

    This dataset handles:
    - Sequence length filtering (min/max length constraints)
    - Event modality filtering (selecting specific event types)
    - Random or fixed-length sequence slicing
    - Conversion of numpy arrays to PyTorch tensors
    - Optional tabular feature processing

    Args:
        path (str): Path to the parquet dataset
        sequence_columns (list[str]): list of column names containing sequence data
        event_time_column (Optional[str]): Name of column containing event timestamps
        event_ids_column (Optional[str]): Name of column containing event type identifiers
        selected_event_ids (Optional[list[int]]): list of event IDs to keep (None keeps all)
        min_length (int): Minimum sequence length (records shorter than this are filtered out)
        max_length (int): Maximum sequence length (longer sequences are sliced)
        random_slicing (bool): Whether to use random slicing for long sequences
        has_tabular (bool): Whether the dataset contains tabular features
        read_columns (Optional[list[str]]): Specific columns to read from parquet files
        shuffle_files (bool): Whether to shuffle the input files
        shuffle_pq (bool): Whether to shuffle rows within each parquet file
        lazy_process (bool): If True, returns processing function instead of processed data

    Raises:
        AssertionError: If invalid length constraints or column configurations are provided
    """

    def __init__(
        self,
        path: str,
        sequence_columns: list[str],
        event_time_column: str | None = None,
        event_ids_column: str | None = None,
        selected_event_ids: list[int] | None = None,
        min_length: int = 1,
        max_length: int = 512,
        random_slicing: bool = False,
        has_tabular: bool = False,
        read_columns: list[str] | None = None,
        shuffle_files=False,
        shuffle_pq=True,
        lazy_process: bool = False,
    ):
        super().__init__(
            path=path,
            read_columns=read_columns,
            shuffle_files=shuffle_files,
            shuffle_pq=shuffle_pq,
        )
        assert min_length >= 0, "min_length must be greater than or equal to 0"
        assert event_time_column not in sequence_columns, (
            "event_time_column in the sequence_columns"
        )
        if event_ids_column is not None:
            assert event_ids_column not in sequence_columns, (
                "event_ids_column in the sequence_columns"
            )

        self.min_length = min_length
        self.max_length = max_length
        self.random_slicing = random_slicing
        self.sequence_columns = sequence_columns
        self.has_tabular = has_tabular
        self.event_ids_column = event_ids_column
        self.event_time_column = event_time_column
        self.lazy_process = lazy_process
        if selected_event_ids is not None:
            self.selected_event_ids = torch.tensor(selected_event_ids).long()
        else:
            self.selected_event_ids = None

    def _filter_by_length(self, record: dict):
        """Filters and slices records based on length constraints and configuration.

        Args:
            record (dict): Input record containing sequence data. Must contain:
                - All columns specified in self.sequence_columns
                - Optional: self.event_time_column and self.event_ids_column

        Returns:
            Optional[tuple]: Either:
                - (start, end) indices for slicing the sequence (end may be None)
                - None if record should be filtered out (too short or invalid)

        Note:
            Records are filtered out if:
            1. Sequence length < min_length
            2. Event/time columns have mismatched lengths (when both present)
        """

        # Filter by min/max lenght and optionaly random slicing
        seq_len = len(record[self.sequence_columns[-1]])
        if seq_len < self.min_length:
            return None

        if seq_len > self.max_length and self.random_slicing:
            max_start = seq_len - self.max_length
            start = random.randint(0, max_start)
            end = start + self.max_length
        else:
            start = -self.max_length
            end = None

        return start, end

    def _modalities_mask(
        self, record: dict, start: int, end: int
    ) -> torch.Tensor | None:
        """Creates and applies a modality mask based on selected event IDs.

        Processes event IDs by:
        1. Converting to PyTorch tensor
        2. Applying modality filtering if selected_event_ids is specified
        3. Slicing according to start/end indices

        Args:
            record (dict): The record to process (modified in-place)
            start (int): Start index for sequence slicing
            end (Optional[int]): End index for sequence slicing (None for open-end)

        Returns:
            Optional[torch.Tensor]: Returns:
                - Modality mask tensor if event filtering was applied
                - None if no filtering was done or no event_ids_column configured

        Modifies:
            - Adds '_event_ids' tensor to record
            - Removes original event_ids_column
            - Filters and slices event IDs according to configuration
        """

        if self.event_ids_column is None:
            return None

        event_data = record[self.event_ids_column]
        record["_event_ids"] = torch.from_numpy(np.array(event_data)).long()
        del record[self.event_ids_column]

        if self.selected_event_ids is None:
            record["_event_ids"] = record["_event_ids"][start:end]
            return None

        modalities_mask = torch.isin(record["_event_ids"], self.selected_event_ids)

        filtered_events = record["_event_ids"][modalities_mask]
        sliced_events = filtered_events[start:end]

        record["_event_ids"] = sliced_events
        return modalities_mask

    def _prepare_record(
        self, record: np.ndarray, start: int, end: int, modalities_mask=None
    ):
        """Converts and slices a single record column to PyTorch tensor.

        Args:
            record (np.ndarray): Input array-like data to convert
            start (int): Start index for slicing
            end (Optional[int]): End index for slicing
            modalities_mask (Optional[torch.Tensor]): Mask for event filtering

        Returns:
            torch.Tensor: Processed tensor with optional filtering and slicing
        """
        record = torch.from_numpy(np.array(record))
        if len(record.shape) == 0:
            return record
        if modalities_mask is not None:
            record = record[modalities_mask]
        record = record[start:end]
        return record

    def _process_record(self, record: dict, start: int, end: int, modalities_mask):
        """Processes all record components according to dataset configuration.

        Handles:
        - Sequence column conversion and type casting
        - Timestamp processing (if configured)
        - Tabular feature processing (if configured)

        Args:
            record (dict): Input record to process
            start (int): Start index for sequence slicing
            end (Optional[int]): End index for sequence slicing
            modalities_mask (Optional[torch.Tensor]): Event modality mask

        Returns:
            dict: Processed record with all components converted to tensors
        """

        partial_prepare = partial(
            self._prepare_record,
            start=start,
            end=end,
            modalities_mask=modalities_mask,
        )
        # Preprocessing sequence features
        for column in self.sequence_columns:
            record[column] = partial_prepare(record[column])

            if record[column].dtype in (torch.int16, torch.int32, torch.int64):
                record[column] = record[column].long()
            elif record[column].dtype in (torch.float16, torch.float32, torch.float64):
                record[column] = record[column].float()

        # Preprocessing timestamps feature
        if self.event_time_column is not None:
            # expected that self.event_time_columns contains float values
            record["_timestamps"] = partial_prepare(
                record[self.event_time_column]
            ).float()
            del record[self.event_time_column]

        if self.has_tabular:
            TabularDataset.process_tabular(record)
        return record

    def process(self, record: dict):
        """Main processing pipeline for a single record.

        Applies the full processing sequence:
        1. Length filtering
        2. Modality filtering (if configured)
        3. Data conversion and type casting
        4. Tabular processing (if configured)

        Args:
            record (dict): Input record to process

        Returns:
            Optional[Union[dict, Callable]]: Depending on configuration:
                - Processed record dict (if lazy_process=False)
                - Processing function (if lazy_process=True)
                - None if record was filtered out
        """

        filter_result = self._filter_by_length(record=record)
        if filter_result is None:
            return filter_result
        else:
            start, end = filter_result

        modalities_mask = self._modalities_mask(
            record=record,
            start=start,
            end=end,
        )
        if modalities_mask is not None and len(record["_event_ids"]) < self.min_length:
            return None

        procces_record_func = partial(
            self._process_record,
            record=record,
            start=start,
            end=end,
            modalities_mask=modalities_mask,
        )
        if self.lazy_process:
            return procces_record_func
        else:
            return procces_record_func()


# TODO add calling shuffle files when after ending iterator
class ShardEventSequenceDataset(EventSequenceDataset):
    """
    A dataset for distributed, sharded processing of event sequences from Parquet files.

    This class extends `EventSequenceDataset` to support:
    - Sharding of files and records across distributed workers (e.g., DDP, DataLoader workers)
    - Efficient filtering of files and records based on sequence length
    - Parallel statistics computation for dataset preparation

    Limitations:
        - Does not support event modality filtering via `selected_event_ids`

    Args:
        path (str): Path to the Parquet dataset directory.
        sequence_columns (list[str]): list of column names containing sequence data.
        event_time_column (Optional[str]): Name of column containing event timestamps.
        event_ids_column (Optional[str]): Name of column containing event type identifiers.
        selected_event_ids (Optional[list[int]]): Not supported; will raise NotImplementedError if set.
        min_length (int): Minimum sequence length (records shorter than this are filtered out).
        max_length (int): Maximum sequence length (longer sequences are sliced).
        random_slicing (bool): Whether to use random slicing for long sequences.
        has_tabular (bool): Whether the dataset contains tabular features.
        read_columns (Optional[list[str]]): Specific columns to read from parquet files.
        shuffle_files (bool): Whether to shuffle the input files.
        shuffle_pq (bool): Whether to shuffle rows within each parquet file.
        lazy_process (bool): If True, returns processing function instead of processed data.

    Raises:
        NotImplementedError: If `selected_event_ids` is provided.
    """

    def __init__(
        self,
        path: str,
        sequence_columns: list[str],
        event_time_column: str | None = None,
        event_ids_column: str | None = None,
        selected_event_ids: list[int] | None = None,
        min_length: int = 1,
        max_length: int = 512,
        random_slicing: bool = False,
        has_tabular: bool = False,
        read_columns: list[str] | None = None,
        shuffle_files: bool = False,
        shuffle_pq: bool = True,
        lazy_process: bool = False,
    ):
        """
        Initialize the sharded event sequence dataset.

        Raises:
            NotImplementedError: If `selected_event_ids` is provided.
        """
        self.shard_by_rank = True
        super().__init__(
            path=path,
            sequence_columns=sequence_columns,
            event_time_column=event_time_column,
            event_ids_column=event_ids_column,
            selected_event_ids=selected_event_ids,
            min_length=min_length,
            max_length=max_length,
            random_slicing=random_slicing,
            has_tabular=has_tabular,
            read_columns=read_columns,
            shuffle_files=shuffle_files,
            shuffle_pq=shuffle_pq,
            lazy_process=lazy_process,
        )
        if selected_event_ids is not None:
            raise NotImplementedError(
                "This implementation doesn't support selected_event_ids"
            )

        self._max_records_per_file: int = None
        self._max_records_per_worker: int = None
        self._prepare_files()

    def _prepare_files(self):
        """
        Analyze and filter Parquet files based on sequence length requirements.

        Filters out files that do not contain enough valid records and sets
        the maximum number of records per file for sharding.
        """
        source_len_files = len(self.files)
        self.files, self._max_records_per_file = self._analyze_parquet_directory(
            files=self.files,
            column_name=self.sequence_columns[0],
            min_length=self.min_length,
        )
        if len(self.files) != source_len_files:
            warnings.warn(
                f"Filtered {source_len_files - len(self.files)} files from {source_len_files}.",
                stacklevel=2,
            )

    def __len__(self):
        """
        Return the number of records available to the current process (shard).

        Returns:
            int: Number of records for this process.
        """
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return self._max_records_per_file * len(self.files) // world_size

    def _calculate_parquet_statistic(
        self, file_path: str, column_name: str, min_length: int
    ) -> dict:
        """
        Calculate statistics about list lengths in a specific column of a Parquet file.

        Args:
            file_path (str): Path to the Parquet file.
            column_name (str): Name of the column to analyze.
            min_length (int): Minimum length threshold for lists.

        Returns:
            dict: Contains 'min_length_count' and 'total_rows' for the file.
        """
        table = pq.read_table(
            file_path,
            columns=[column_name],
            use_threads=False,
            memory_map=True,
        )
        array = table[column_name]
        lengths = pa.compute.list_value_length(array)
        min_length_count = (lengths.to_numpy() < min_length).sum()
        return {
            "min_length_count": min_length_count,
            "total_rows": table.shape[0],
        }

    def _analyze_parquet_directory(
        self, files: list, column_name: str, min_length: int
    ) -> tuple:
        """
        Process multiple Parquet files to analyze list lengths in a specific column.

        Args:
            files (list): list of Parquet file paths to process.
            column_name (str): Name of the column to analyze.
            min_length (int): Minimum length threshold for lists.

        Returns:
            tuple: (list of file paths that meet the criteria, minimum valid row count).
        """
        stats = []
        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(
                    self._calculate_parquet_statistic,
                    file_path,
                    column_name,
                    min_length,
                ): file_path
                for file_path in files
            }
            progress_bar = tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(files),
                desc="Scanning parquet files, preparing dataset",
            )
            for future in progress_bar:
                file_path = future_to_file[future]
                file_stats = {"file_path": file_path, **future.result()}
                stats.append(file_stats)

        stats_df = pd.DataFrame(stats)
        stats_df["valid_rows"] = stats_df["total_rows"] - stats_df["min_length_count"]
        mean_rows = stats_df["valid_rows"].mean()
        std_rows = stats_df["valid_rows"].std()
        min_rows = stats_df["valid_rows"].min()
        max_rows = stats_df["valid_rows"].max()

        warnings.warn(
            f"\nmean_rows: {mean_rows}, std_rows: {std_rows}, min_rows: {min_rows}, max_rows: {max_rows}\n",
            stacklevel=2,
        )

        return stats_df["file_path"].tolist(), stats_df["valid_rows"].min()

    def _get_enviroment_info(self):
        """
        Retrieve environment information for distributed and multi-worker processing.

        Returns:
            tuple: (num_workers, worker_id, world_size, rank)
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
        return num_workers, worker_id, world_size, rank

    def files_per_worker(self):
        """
        Distribute files among workers and processes for parallel processing.

        Returns:
            tuple[int, int]: The start and end indices of files for the current worker.
        """
        num_workers, worker_id, world_size, rank = self._get_enviroment_info()
        if num_workers == 1 and world_size == 1:
            iter_start = 0
            iter_end = len(self.files)
            self._max_records_per_worker = (
                self._max_records_per_file * len(self.files) * 2
            )
        else:
            files_len = len(self.files)
            per_process = files_len // world_size
            if world_size > files_len:
                raise ValueError(
                    "Total number of files must be greater than world_size"
                )
            self.files = self.files[: per_process * world_size]
            per_worker = int((per_process) / float(num_workers))
            remainder = per_process % num_workers
            process_start = rank * per_process
            process_end = min(process_start + per_process, len(self.files))
            iter_start = rank * per_process
            if worker_id > 0:
                iter_start += per_worker * worker_id + remainder
            if worker_id == 0:
                iter_end = min(iter_start + per_worker + remainder, process_end)
            else:
                iter_end = min(iter_start + per_worker, process_end)
            if worker_id == 0:
                self._max_records_per_worker = self._max_records_per_file * (
                    per_worker + remainder
                )
            else:
                self._max_records_per_worker = self._max_records_per_file * per_worker
        return iter_start, iter_end

    def process(self, record):
        """
        Process a single record, tracking the maximum records per worker.

        Args:
            record (dict): Input record to process.

        Returns:
            Optional[dict]: Processed record or None if quota exceeded or record is invalid.
        """
        process_result = super().process(record=record)
        if process_result is not None:
            self._max_records_per_worker -= 1
            if self._max_records_per_worker < 0:
                return None
            else:
                return process_result
        return None
