from functools import partial
from typing import Any

import torch

from avatar.data.dataset.collate_fn.base_collate_fn import BaseCollateFn
from avatar.data.tabular_batch import TabularBatch


def check_columns_to_tensor(columns_to_tensor: dict[str, str] | None):
    """Check columns types for conversion to tensor"""
    if columns_to_tensor is None:
        return True
    for column, dtype in columns_to_tensor.items():
        if dtype not in ["long", "float"]:
            raise ValueError(f"Invalid dtype for column {column}: {dtype}")


class TabularCollateFn(BaseCollateFn):
    """
    Initializes the TabularCollateFn class.

    Args:
        target_column (Optional[str], optional): The column name to use as the
            target labels. Defaults to None.
        is_regression (bool, optional): If True, processes the target as regression labels.
            Defaults to False (classification).
    """

    def __init__(
        self,
        target_column: str | None = None,
        is_regression: bool = False,
    ):
        self.target_column = target_column
        self.is_regression = is_regression

    @staticmethod
    def collate_tabular(batch):
        """
        Collates tabular features (both categorical and numerical).

        Args:
            tab_features: A batch of tabular features (list of dictionaries).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of collated categorical and numerical features.
        """
        tab_features = [item["tab_features"] for item in batch]
        if tab_features[0]["cat_features"] is not None:
            cat_features = torch.stack([item["cat_features"] for item in tab_features])
        else:
            cat_features = None

        if tab_features[0]["num_features"] is not None:
            num_features = torch.stack([item["num_features"] for item in tab_features])
        else:
            num_features = None

        if (
            "_hidden_states" in tab_features[0]
            and tab_features[0]["_hidden_states"] is not None
        ):
            hidden_states = {}
            for key in tab_features[0]["_hidden_states"].keys():
                hidden_states[key] = torch.stack([
                    item["_hidden_states"][key] for item in tab_features
                ])
        else:
            hidden_states = None

        return TabularBatch(
            cat_features=cat_features,
            num_features=num_features,
            hidden_states=hidden_states,
        )

    def __call__(self, batch: list[dict[str, Any]]):
        """
        Processes the batch and converts it into a suitable format for model input.

        Args:
            batch (List[Dict[str, Any]]): A list of samples
            (each sample is a dictionary of features).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of processed features
                and target labels.
        """
        if isinstance(batch[0], partial):
            batch = [process_func() for process_func in batch]

        processed_batch = {}
        processed_batch["tab_features"] = TabularCollateFn.collate_tabular(batch=batch)

        for key in batch[0].keys():
            if self.target_column is not None and key == self.target_column:
                target_values = self.extract_values(batch, key)
                if self.is_regression:
                    processed_batch["targets"] = torch.FloatTensor(target_values)
                else:
                    processed_batch["targets"] = torch.LongTensor(target_values)
            elif key not in ["tab_features"]:
                processed_batch[key] = self.extract_values(batch, key)

        hidden_states = (
            {
                key: torch.where(value.isnan(), 0, value)
                for key, value in processed_batch["tab_features"].hidden_states.items()
            }
            if processed_batch["tab_features"].hidden_states is not None
            else None
        )
        processed_batch["tab_features"]._hidden_states = hidden_states
        return processed_batch


class UpliftCollateFn(TabularCollateFn):
    def __init__(
        self,
        treatment_column: str,
        target_column: str | None = None,
        group_column: str | None = None,
        inverse_treatment: bool = False,
    ):
        """
        Initializes the UpliftCollateFn class.

        Args:
            treatment_column: str; name of column that contains the binary values:
                1 - treatment group; 0 - control
            target_column (Optional[str], optional): The column name to use as the
                target labels. Defaults to None.
        """
        super().__init__(target_column=target_column, is_regression=False)
        self.treatment_column = treatment_column
        self.group_column = group_column
        self.inverse_treatment = inverse_treatment

    def __call__(self, batch: list[dict[str, Any]]):
        """
        Processes the batch and converts it into a suitable format for model input.

        Args:
            batch (List[Dict[str, Any]]): A list of samples
            (each sample is a dictionary of features).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of processed features and target labels.
        """
        processed_batch = super().__call__(batch=batch)
        if self.inverse_treatment:
            processed_batch["is_treat"] = 1 - torch.LongTensor(
                processed_batch[self.treatment_column]
            )
        else:
            processed_batch["is_treat"] = torch.LongTensor(
                processed_batch[self.treatment_column]
            )
        del processed_batch[self.treatment_column]
        if self.group_column is not None:
            processed_batch["group"] = torch.LongTensor(
                processed_batch[self.group_column]
            )
            del processed_batch[self.group_column]

        return processed_batch


class MultiTaskUpliftCollateFn(UpliftCollateFn):
    def __init__(
        self,
        treatment_column: str,
        task_name_column: str,
        task_mapping: dict[str, int],
        target_column: str | None = None,
        group_column: str | None = None,
        inverse_treatment: bool = False,
    ):
        """
        Initializes the UpliftCollateFn class.

        Args:
            treatment_column: str; name of column that contains the binary values:
                1 - treatment group; 0 - control
            target_column (Optional[str], optional): The column name to use as the
                target labels. Defaults to None.
        """
        super().__init__(
            target_column=target_column,
            treatment_column=treatment_column,
            group_column=group_column,
            inverse_treatment=inverse_treatment,
        )
        self.task_name_column = task_name_column
        self.task_mapping = task_mapping

    def __call__(self, batch: list[dict[str, Any]]):
        """
        Processes the batch and converts it into a suitable format for model input.

        Args:
            batch (List[Dict[str, Any]]): A list of samples
            (each sample is a dictionary of features).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of processed features and target labels.
        """
        processed_batch = super().__call__(batch=batch)
        processed_batch["task_name"] = torch.LongTensor([
            self.task_mapping.get(x) for x in processed_batch[self.task_name_column]
        ])
        del processed_batch[self.task_name_column]
        return processed_batch


class SupervisedCollateFn(TabularCollateFn):
    def __init__(
        self,
        target_column: str | None = None,
        is_regression: bool = False,
        add_extra_columns: dict[str, str] | None = None,
    ):
        if add_extra_columns is None:
            add_extra_columns = {}
        super().__init__(target_column=target_column, is_regression=is_regression)
        self.add_extra_columns = add_extra_columns

    def __call__(self, batch: list[dict[str, Any]]):
        processed_batch = super().__call__(batch=batch)
        if self.add_extra_columns is None:
            return processed_batch
        for col_rename, col_name in self.add_extra_columns.items():
            if col_name in processed_batch:
                processed_batch[col_rename] = torch.tensor(processed_batch[col_name])
                del processed_batch[col_name]
        return processed_batch


class MultiTaskSupervisedCollateFn(SupervisedCollateFn):
    def __init__(
        self,
        task_name_column: str,
        task_mapping: dict[str, int],
        target_column: str | None = None,
        is_regression: bool = False,
        add_extra_columns: dict[str, str] | None = None,
    ):
        """
        Initializes the UpliftCollateFn class.

        Args:
            treatment_column: str; name of column that contains the binary values:
                1 - treatment group; 0 - control
            target_column (Optional[str], optional): The column name to use as the
                target labels. Defaults to None.
        """
        if add_extra_columns is None:
            add_extra_columns = {}
        super().__init__(
            target_column=target_column,
            is_regression=is_regression,
            add_extra_columns=add_extra_columns,
        )
        self.task_name_column = task_name_column
        self.task_mapping = task_mapping

    def __call__(self, batch: list[dict[str, Any]]):
        """
        Processes the batch and converts it into a suitable format for model input.

        Args:
            batch (List[Dict[str, Any]]): A list of samples
            (each sample is a dictionary of features).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of processed features and target labels.
        """
        processed_batch = super().__call__(batch=batch)
        processed_batch["task_name"] = torch.LongTensor([
            self.task_mapping.get(x) for x in processed_batch[self.task_name_column]
        ])
        del processed_batch[self.task_name_column]
        return processed_batch
