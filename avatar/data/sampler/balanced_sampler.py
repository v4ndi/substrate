import numpy as np
import torch.multiprocessing as mp

from avatar.data.sampler.base_sampler import BaseSampler


class DatePrioritazeSampler(BaseSampler):
    """
    A custom data sampler that prioritizes recent data based on a date threshold,
    gradually increasing the proportion of recent samples over training epochs.

    The sampler dynamically adjusts the ratio of 'recent' (post-threshold) and
    'default' (pre-threshold) data in each batch, according to a specified schedule.

    Args:
        batch_size: The size of each batch.
        date_column: The name of the column in the dataset containing the date.
        date_treshold: The threshold date for considering data as 'recent'.
        epochs: A list containing the start and end epochs for the sampler.
        last_months_ratios: A list containing the ratios of 'recent' data at the start and at the end.
        accumulated_batches: The number of batches to accumulate before resampling.
    """

    def __init__(
        self,
        batch_size: int,
        date_column: str,
        date_treshold: str = "2025-01-31",
        epochs: tuple[int, int] = (0, 0),
        last_months_ratios: tuple[int, int] = (0.35, 0.9),
        accumulated_batches: int = 2,
        num_workers: int = 8,
    ):
        self.batch_size = batch_size
        self.date_column = date_column
        self.date_treshold = np.datetime64(f"{date_treshold}")
        self.start_epoch, self.end_epoch = epochs
        self.last_months_ratio_start, self.last_months_ratio_end = last_months_ratios
        self.accumulated_batches = accumulated_batches
        self.num_workers = num_workers

        self.scale_diff = self.last_months_ratio_end - self.last_months_ratio_start
        self.continue_iter_flag = True
        self._current_step = mp.Value("i", 0)

    def _collect_dates(self):
        self.buff_last_ids = []
        self.buff_last = []

        self.buff_default = []
        self.buff_default_ids = []

        id_last, id_default = 0, 0

        self.continue_iter_flag = False
        for data in self.dataset_iterator:
            if data[self.date_column] >= self.date_treshold:
                self.buff_last.append(data)
                self.buff_last_ids.append(id_last)
                id_last += 1

            else:
                self.buff_default.append(data)
                self.buff_default_ids.append(id_default)
                id_default += 1

            if len(self.buff_last) == self.num_of_last:
                self.continue_iter_flag = True
                break

    def _collect_fianl_buff(self):
        buff_last_ids = np.array(self.buff_last_ids)
        buff_default_ids = np.array(self.buff_default_ids)

        np.random.shuffle(buff_last_ids)
        np.random.shuffle(buff_default_ids)

        buff_default_ids = buff_default_ids[: self.num_of_default]

        self.buff_last = [self.buff_last[i] for i in buff_last_ids]
        self.buff_default = [self.buff_default[i] for i in buff_default_ids]

        final_buff = np.concatenate([self.buff_last, self.buff_default])
        np.random.shuffle(final_buff)
        return final_buff

    def __iter__(self):
        assert self.dataset_iterator is not None, (
            "set dataset iterator, using set_dataset_iterator() method"
        )
        self.continue_iter_flag = True
        current_epoch = self._current_step.value // self.num_workers + 1
        if current_epoch >= self.start_epoch:
            self.scale_ratio = (
                self.last_months_ratio_start
                + (self.scale_diff)
                / (self.end_epoch - self.start_epoch)
                * current_epoch
            )
            self.num_of_last = int(self.batch_size * self.scale_ratio)
            self.num_of_default = self.batch_size - self.num_of_last

            while self.continue_iter_flag:
                self._collect_dates()
                final_buff = self._collect_fianl_buff()

                for data in final_buff:
                    yield data
        else:
            for data in self.dataset_iterator:
                yield data

        self._current_step.value += 1

    def __len__(self):
        assert self.dataset_iterator is not None, (
            "set dataset iterator, using set_dataset_iterator() method"
        )
        return len(self.dataset_iterator)


class StreamingBalancedSampler(BaseSampler):
    """
    Sampler for iterable dataset

    Args:
        resample_column: column with target values, that should be resampled
        conversion: conversion of positives, that must be in result samples
        batch_size: batch size (you should copy from dataloader)
        accumulated_batches: number of batches to accumulate before resampling
    """

    def __init__(
        self,
        resample_column: str,
        conversion: float,
        batch_size: int,
        accumulated_batches: int = 2,
    ):
        self.resample_column = resample_column
        self.conversion = conversion
        self.batch_size = batch_size
        self.accumulated_batches = accumulated_batches

        self._calc_num_pos_neg()

        self.continue_iter_flag = True
        self.dataset_iterator = None

    def _calc_num_pos_neg(self):
        self.num_pos = int(self.conversion * self.batch_size * self.accumulated_batches)
        self.num_neg = self.batch_size * self.accumulated_batches - self.num_pos

    def _fill_buffers(self):
        self.pos_ids_buff = []
        self.neg_ids_buff = []
        self.buff = []
        idx = 0
        self.continue_iter_flag = False
        for data in self.dataset_iterator:
            target = data[self.resample_column]
            if target == 1:
                self.pos_ids_buff.append(idx)
            else:
                self.neg_ids_buff.append(idx)
            self.buff.append(data)
            idx += 1
            if len(self.buff) >= self.batch_size * self.accumulated_batches:
                self.continue_iter_flag = True
                break

    def __iter__(self):
        self.continue_iter_flag = True
        assert self.dataset_iterator is not None, (
            "set dataset iterator, using set_dataset_iterator() method"
        )
        self._calc_num_pos_neg()

        while self.continue_iter_flag:
            self._fill_buffers()
            num_pos = self.num_pos
            num_neg = self.num_neg

            if (
                len(self.buff) < self.batch_size * self.accumulated_batches
            ):  # to save last batch_sizes
                num_pos = (
                    self.num_pos
                    * len(self.buff)
                    // (self.batch_size * self.accumulated_batches)
                )
                num_neg = len(self.buff) - num_pos

            pos_ids = (
                np.random.choice(self.pos_ids_buff, size=num_pos, replace=True)
                if len(self.pos_ids_buff) > 0
                else np.array([], dtype=int)
            )
            neg_ids = (
                np.random.choice(self.neg_ids_buff, size=num_neg, replace=True)
                if len(self.neg_ids_buff) > 0
                else np.array([], dtype=int)
            )
            accum_batches_ids = np.concatenate([pos_ids, neg_ids])
            np.random.shuffle(accum_batches_ids)
            accum_batches_ids = accum_batches_ids[: len(self.buff)]
            for i in accum_batches_ids:
                yield self.buff[i]

    def __len__(self):
        assert self.dataset_iterator is not None, (
            "set dataset iterator, using set_dataset_iterator() method"
        )
        return len(self.dataset_iterator)


class StreamingBalancedUnderSampler(StreamingBalancedSampler):
    """
    Sampler for iterable dataset, that undersamples negatives
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        self.continue_iter_flag = True
        assert self.dataset_iterator is not None, (
            "set dataset iterator, using set_dataset_iterator() method"
        )

        while self.continue_iter_flag:
            self._fill_buffers()

            pos_ids = (
                np.array(self.pos_ids_buff)
                if len(self.pos_ids_buff) > 0
                else np.array([], dtype=int)
            )
            num_neg = int((1 / self.conversion - 1) * len(pos_ids))
            neg_ids = (
                np.random.choice(
                    self.neg_ids_buff,
                    size=min(num_neg, len(self.neg_ids_buff)),
                    replace=False,
                )
                if len(self.neg_ids_buff) > 0
                else np.array([], dtype=int)
            )
            accum_batches_ids = np.concatenate([pos_ids, neg_ids])
            np.random.shuffle(accum_batches_ids)
            accum_batches_ids = accum_batches_ids[: len(self.buff)]
            for i in accum_batches_ids:
                yield self.buff[i]


class ScheduleConversionSampler(StreamingBalancedSampler):
    """
    Sampler for iterable dataset

    works as a sampler with scheduled conversion from initial to final conversion linearly
    """

    def __init__(
        self,
        initial_conversion: float,
        final_conversion: float,
        num_scheduling_steps: int,
        resample_column: str,
        batch_size: int,
        num_workers: int,
        accumulated_batches: int = 4,
    ):
        super().__init__(
            conversion=initial_conversion,
            resample_column=resample_column,
            batch_size=batch_size,
            accumulated_batches=accumulated_batches,
        )

        self.initial_conversion = initial_conversion
        self.final_conversion = final_conversion
        self.num_scheduling_steps = num_scheduling_steps
        self._current_step = mp.Value("i", 0)
        self.num_workers = num_workers

    def _set_conversion(self):
        num_epochs = self._current_step.value // self.num_workers
        train_ratio = num_epochs / self.num_scheduling_steps
        if num_epochs >= self.num_scheduling_steps:
            self.conversion = self.final_conversion
        else:
            self.conversion = (
                self.initial_conversion
                + (self.final_conversion - self.initial_conversion) * train_ratio
            )

    def _get_default_iter(self):
        for data in self.dataset_iterator:
            yield data

    def __iter__(self):
        self._set_conversion()
        if self.conversion == self.final_conversion:
            return self._get_default_iter()
        iter = super().__iter__()
        self._current_step.value += 1
        return iter
