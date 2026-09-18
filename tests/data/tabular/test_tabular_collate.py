import os
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fmlib.data.tabular import (
    SupervisedCollateFn,
    TabularCollateFn,
    UpliftCollateFn,
)


@pytest.fixture
def create_parquet_file():
    """
    Creates a temporary Parquet file with required columns and returns the file path.
    """
    with TemporaryDirectory() as temp_dir:
        data = {
            "id": [1, 2, 3],
            "target": [0, 1, 0],
            "cat_features": [[0, 1], [2, 3], [4, 5]],  # Categorical features (lists)
            "num_features": [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],  # Numerical features (lists)
        }

        # Create a Pandas DataFrame
        df = pd.DataFrame(data)

        # Convert to Arrow Table
        table = pa.Table.from_pandas(df)

        # Write the Arrow Table to a Parquet file
        parquet_path = os.path.join(temp_dir, "test_data.parquet")
        pq.write_table(table, parquet_path)

        yield parquet_path


def test_collate_tabular(create_parquet_file):
    """
    Test the collate_tabular method.
    """
    # _ = TabularDataset(path=os.path.dirname(create_parquet_file))
    collate_fn = TabularCollateFn(target_column="target")

    # Simulate a batch of tabular data
    batch = [
        {
            "id": 1,
            "target": 0,
            "cat_features": torch.tensor([0, 1]),
            "num_features": torch.tensor([0.1, 0.2]),
            "tab_features": {
                "cat_features": torch.tensor([0, 1]),
                "num_features": torch.tensor([0.1, 0.2]),
            },
        },
        {
            "id": 2,
            "target": 1,
            "cat_features": torch.tensor([2, 3]),
            "num_features": torch.tensor([0.3, 0.4]),
            "tab_features": {
                "cat_features": torch.tensor([2, 3]),
                "num_features": torch.tensor([0.3, 0.4]),
            },
        },
    ]

    # Test collate_tabular
    collated = collate_fn.collate_tabular([sample for sample in batch])

    assert collated.cat_features.shape == (2, 2)  # 2 samples, 2 cat features
    assert collated.num_features.shape == (2, 2)  # 2 samples, 2 num features


def test_tabular_collate_fn_classification(create_parquet_file):
    """
    Test the __call__ method for classification (target_column="target").
    """
    # dataset = TabularDataset(path=os.path.dirname(create_parquet_file))
    collate_fn = TabularCollateFn(target_column="target", is_regression=False)

    # Simulate a batch of tabular data
    batch = [
        {
            "id": 1,
            "target": 0,
            "cat_features": torch.tensor([0, 1]),
            "num_features": torch.tensor([0.1, 0.2]),
            "tab_features": {
                "cat_features": torch.tensor([0, 1]),
                "num_features": torch.tensor([0.1, 0.2]),
            },
        },
        {
            "id": 2,
            "target": 1,
            "cat_features": torch.tensor([2, 3]),
            "num_features": torch.tensor([0.3, 0.4]),
            "tab_features": {
                "cat_features": torch.tensor([2, 3]),
                "num_features": torch.tensor([0.3, 0.4]),
            },
        },
    ]

    # Process the batch
    processed_batch = collate_fn(batch)

    # Check if targets are LongTensor for classification
    assert processed_batch["targets"].dtype == torch.long
    assert processed_batch["targets"].shape == torch.Size([2])

    # Check if tabular features are properly collated
    assert "tab_features" in processed_batch
    assert processed_batch["tab_features"].cat_features.shape == (2, 2)
    assert processed_batch["tab_features"].num_features.shape == (2, 2)


def test_tabular_collate_fn_regression(create_parquet_file):
    """
    Test the __call__ method for regression (target_column="target").
    """
    # dataset = TabularDataset(path=os.path.dirname(create_parquet_file))
    collate_fn = TabularCollateFn(target_column="target", is_regression=True)

    # Simulate a batch of tabular data
    batch = [
        {
            "id": 1,
            "target": 0,
            "cat_features": torch.tensor([0, 1]),
            "num_features": torch.tensor([0.1, 0.2]),
            "tab_features": {
                "cat_features": torch.tensor([0, 1]),
                "num_features": torch.tensor([0.1, 0.2]),
            },
        },
        {
            "id": 2,
            "target": 1,
            "cat_features": torch.tensor([2, 3]),
            "num_features": torch.tensor([0.3, 0.4]),
            "tab_features": {
                "cat_features": torch.tensor([2, 3]),
                "num_features": torch.tensor([0.3, 0.4]),
            },
        },
    ]

    # Process the batch
    processed_batch = collate_fn(batch)

    # Check if targets are FloatTensor for regression
    assert processed_batch["targets"].dtype == torch.float
    assert processed_batch["targets"].shape == torch.Size([2])

    # Check if tabular features are properly collated
    assert "tab_features" in processed_batch
    assert processed_batch["tab_features"].cat_features.shape == (2, 2)
    assert processed_batch["tab_features"].num_features.shape == (2, 2)


def test_tabular_collate_fn_hidden_state(create_parquet_file):
    """
    Test the __call__ method for classification (target_column="target").
    """
    # dataset = TabularDataset(path=os.path.dirname(create_parquet_file))
    collate_fn = TabularCollateFn(target_column="target", is_regression=False)

    # Simulate a batch of tabular data
    batch = [
        {
            "id": 1,
            "target": 0,
            "cat_features": torch.tensor([0, 1]),
            "num_features": torch.tensor([0.1, 0.2]),
            "tab_features": {
                "cat_features": torch.tensor([0, 1]),
                "num_features": torch.tensor([0.1, 0.2]),
                "_hidden_states": {"ext_emb": torch.FloatTensor([0.0, 0.0, 0.0])},
            },
        },
        {
            "id": 2,
            "target": 1,
            "cat_features": torch.tensor([2, 3]),
            "num_features": torch.tensor([0.3, 0.4]),
            "tab_features": {
                "cat_features": torch.tensor([2, 3]),
                "num_features": torch.tensor([0.3, 0.4]),
                "_hidden_states": {"ext_emb": torch.FloatTensor([1.0, 1.0, 1.0])},
            },
        },
    ]

    # Process the batch
    processed_batch = collate_fn(batch)

    # Check if targets are LongTensor for classification
    assert processed_batch["targets"].dtype == torch.long
    assert processed_batch["targets"].shape == torch.Size([2])

    hidden_states = processed_batch["tab_features"].hidden_states
    assert set(hidden_states) == {"ext_emb"}
    assert hidden_states["ext_emb"].dtype == torch.float
    assert hidden_states["ext_emb"].shape == torch.Size([2, 3])
    assert torch.all(
        hidden_states["ext_emb"]
        == torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )

    # Check if tabular features are properly collated
    assert "tab_features" in processed_batch
    assert processed_batch["tab_features"].cat_features.shape == (2, 2)
    assert processed_batch["tab_features"].num_features.shape == (2, 2)


# -- a column that already has the name the batch wants ----------------------
#
# The metrics declare the keys they read, and ``group`` is one of them — so
# naming the source column ``group`` is the obvious thing to do. Both collate
# functions used to write the tensor under that name and then delete the source
# column by the same name, removing what they had just built.


def records_with_a_group_column():
    return [
        {
            "epk_id": index,
            "target": index % 2,
            "treatment": index % 2,
            "group": index % 3,
            "split_type": "calib",
            "tab_features": {
                "cat_features": torch.tensor([0, 1]),
                "num_features": torch.tensor([0.1, 0.2]),
            },
        }
        for index in range(4)
    ]


def test_uplift_collate_keeps_a_group_column_called_group():
    collate_fn = UpliftCollateFn(
        target_column="target", treatment_column="treatment", group_column="group"
    )

    batch = collate_fn(records_with_a_group_column())

    assert torch.equal(batch["group"], torch.LongTensor([0, 1, 2, 0]))
    assert torch.equal(batch["is_treat"], torch.LongTensor([0, 1, 0, 1]))


def test_uplift_collate_still_inverts_the_treatment_flag():
    collate_fn = UpliftCollateFn(
        target_column="target",
        treatment_column="treatment",
        group_column="group",
        inverse_treatment=True,
    )

    batch = collate_fn(records_with_a_group_column())

    assert torch.equal(batch["is_treat"], torch.LongTensor([1, 0, 1, 0]))


def test_supervised_collate_keeps_a_column_renamed_to_itself():
    collate_fn = SupervisedCollateFn(
        target_column="target", add_extra_columns={"group": "group"}
    )

    batch = collate_fn(records_with_a_group_column())

    assert torch.equal(batch["group"], torch.tensor([0, 1, 2, 0]))
    # A string column is left alone — the metrics read split_type as it comes.
    assert batch["split_type"] == ["calib"] * 4


def test_supervised_collate_still_renames_a_differently_named_column():
    collate_fn = SupervisedCollateFn(
        target_column="target", add_extra_columns={"group": "treatment"}
    )

    batch = collate_fn(records_with_a_group_column())

    assert torch.equal(batch["group"], torch.tensor([0, 1, 0, 1]))
    assert "treatment" not in batch


if __name__ == "__main__":
    pytest.main()
