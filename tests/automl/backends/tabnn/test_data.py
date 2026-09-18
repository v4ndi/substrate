"""The processed-data directory: named by content, proven by its manifest."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from fmlib.automl import BinaryTaskConfig
from fmlib.automl.backends.tabnn.data import (
    CONTRACT_VERSION,
    MANIFEST_NAME,
    build_schema,
    clear_processed_data,
    content_key,
    prepare_processed_data,
)
from fmlib.automl.data import CanonicalColumnMapper, ParquetSource
from fmlib.automl.exceptions import ArtifactIntegrityError, SchemaError
from fmlib.automl.execution import ExecutionContext
from fmlib.automl.tasks.preparation import DataPreparation


def _frame(rows: int, seed: int, hidden_dtype=None) -> pl.DataFrame:
    hidden_dtype = hidden_dtype or pl.List(pl.Float32)
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(rows, 4)).astype(np.float32)
    return pl.DataFrame({
        "epk_id": np.arange(rows) + seed * 1000,
        "report_month": pl.Series(["2025-01-01"] * rows).str.to_date(),
        "segment": rng.choice(["a", "b", "c"], rows),
        "balance": rng.normal(size=rows),
        "seq_hidden_state": [row.tolist() for row in hidden],
        "target": rng.integers(0, 2, rows),
    }).with_columns(pl.col("seq_hidden_state").cast(hidden_dtype))


@pytest.fixture
def sources(tmp_path):
    paths = {}
    for name, rows, seed in (("train", 400, 1), ("valid", 200, 2)):
        directory = tmp_path / name
        directory.mkdir()
        frame = _frame(rows, seed)
        half = rows // 2
        frame.head(half).write_parquet(directory / "part-0.parquet")
        frame.tail(rows - half).write_parquet(directory / "part-1.parquet")
        paths[name] = ParquetSource.resolve(directory)
    return paths


@pytest.fixture
def config(tmp_path):
    return BinaryTaskConfig(
        env_type="local",
        backend="tabnn",
        engine="tabular_transformer",
        device="cpu",
        target_column="target",
        client_id_column="epk_id",
        group_column=None,
        date_column="report_month",
        categorical_columns=["segment"],
        numerical_columns=["balance"],
        hidden_state_columns=["seq_hidden_state"],
        hyperopt=False,
        output_dir=tmp_path / "outputs",
        environment={},
    )


def _prepare(config, sources, **overrides):
    context = ExecutionContext.from_config(config)
    internal = context.internal_config
    mapper = CanonicalColumnMapper.from_config(config)
    schema = build_schema(sources["train"], internal, mapper.to_external)
    kwargs = dict(
        config=internal,
        schema=schema,
        train_source=sources["train"],
        valid_source=sources["valid"],
        train_manifest=DataPreparation.source_manifest(sources["train"]),
        valid_manifest=DataPreparation.source_manifest(sources["valid"]),
        to_external=mapper.to_external,
    )
    kwargs.update(overrides)
    return prepare_processed_data(**kwargs), schema


# --------------------------------------------------------------------------- #
# Schema from footers                                                          #
# --------------------------------------------------------------------------- #
def test_schema_records_embedding_widths_and_keeps_them_out_of_the_features(
    config, sources
):
    context = ExecutionContext.from_config(config)
    mapper = CanonicalColumnMapper.from_config(config)
    schema = build_schema(sources["train"], context.internal_config, mapper.to_external)
    assert schema.hidden_states == {"seq_hidden_state": 4}
    assert schema.feature_order == ("segment", "balance")
    assert "seq_hidden_state" not in schema.feature_order


def test_a_fixed_size_array_width_comes_from_the_dtype_alone(tmp_path, config):
    directory = tmp_path / "arrayed"
    directory.mkdir()
    _frame(50, 3, hidden_dtype=pl.Array(pl.Float32, 4)).write_parquet(
        directory / "part-0.parquet"
    )
    context = ExecutionContext.from_config(config)
    mapper = CanonicalColumnMapper.from_config(config)
    schema = build_schema(
        ParquetSource.resolve(directory), context.internal_config, mapper.to_external
    )
    assert schema.hidden_states == {"seq_hidden_state": 4}


def test_a_missing_column_is_named(tmp_path, config):
    directory = tmp_path / "short"
    directory.mkdir()
    _frame(20, 4).drop("balance").write_parquet(directory / "part-0.parquet")
    context = ExecutionContext.from_config(config)
    mapper = CanonicalColumnMapper.from_config(config)
    with pytest.raises(SchemaError, match="balance"):
        build_schema(
            ParquetSource.resolve(directory),
            context.internal_config,
            mapper.to_external,
        )


# --------------------------------------------------------------------------- #
# Encoding and publication                                                     #
# --------------------------------------------------------------------------- #
def test_encoding_publishes_a_complete_directory(config, sources):
    processed, _ = _prepare(config, sources)

    assert processed.root.is_dir()
    assert processed.root.name == processed.key
    assert processed.reused is False
    assert processed.train.rows == 400
    assert processed.valid.rows == 200
    assert processed.vocab_size > 0
    assert processed.num_numerical == 1
    assert processed.hidden_states == {"seq_hidden_state": 4}

    manifest = json.loads((processed.root / MANIFEST_NAME).read_text())
    assert manifest["complete"] is True
    assert manifest["contract_version"] == CONTRACT_VERSION
    for split in ("train", "valid"):
        parts = manifest[split]["parts"]
        assert parts
        assert all((processed.root / split / part).is_file() for part in parts)


def test_the_encoded_records_carry_what_the_dataset_reads(config, sources):
    processed, _ = _prepare(config, sources)
    frame = pl.read_parquet(processed.train.path)
    assert {"cat_features", "num_features"} <= set(frame.columns)
    assert set(processed.identity_columns) <= set(frame.columns)
    assert frame["cat_features"].list.len().unique().to_list() == [1]
    assert frame["num_features"].list.len().unique().to_list() == [1]
    assert frame["seq_hidden_state"].list.len().unique().to_list() == [4]


def test_nothing_is_left_behind_when_the_directory_is_published(config, sources):
    processed, _ = _prepare(config, sources)
    siblings = [path.name for path in processed.root.parent.iterdir()]
    assert not [name for name in siblings if ".tmp-" in name]


def test_the_preprocessor_is_its_own_file_and_is_not_a_pickle(config, sources):
    processed, _ = _prepare(config, sources)
    text = processed.preprocessor_path.read_text()
    assert processed.preprocessor_path.suffix == ".yaml"
    assert "!!python" not in text
    assert processed.preprocessor().vocab_size == processed.vocab_size


# --------------------------------------------------------------------------- #
# Reuse                                                                        #
# --------------------------------------------------------------------------- #
def test_the_same_inputs_reuse_the_same_directory(config, sources):
    first, _ = _prepare(config, sources)
    second, _ = _prepare(config, sources)
    assert second.key == first.key
    assert second.reused is True
    assert second.train.rows == first.train.rows
    assert second.identity_columns == first.identity_columns


def test_a_different_source_is_simply_a_different_directory(config, sources, tmp_path):
    first, _ = _prepare(config, sources)
    other = tmp_path / "train2"
    other.mkdir()
    _frame(400, 9).write_parquet(other / "part-0.parquet")
    sources["train"] = ParquetSource.resolve(other)
    second, _ = _prepare(config, sources)
    assert second.key != first.key
    assert first.root.is_dir() and second.root.is_dir()


def test_a_different_preprocessing_configuration_is_a_different_directory(
    config, sources
):
    first, _ = _prepare(config, sources)
    second, _ = _prepare(config, sources, preprocessor_kwargs={"batch_rows": 1024})
    assert second.key != first.key


def test_an_interrupted_encoding_is_never_reused(config, sources):
    processed, _ = _prepare(config, sources)
    (processed.root / MANIFEST_NAME).unlink()
    again, _ = _prepare(config, sources)
    assert again.reused is False
    assert (again.root / MANIFEST_NAME).is_file()


def test_a_marker_that_disagrees_with_the_content_is_an_error_not_an_overwrite(
    config, sources
):
    processed, _ = _prepare(config, sources)
    missing = processed.root / "train" / processed.train.parts[0]
    missing.unlink()
    with pytest.raises(ArtifactIntegrityError) as error:
        _prepare(config, sources)
    message = str(error.value)
    assert "listed but missing" in message
    assert "Nothing was overwritten" in message
    assert (processed.root / MANIFEST_NAME).is_file()


def test_an_older_contract_version_is_refused_with_its_reason(config, sources):
    processed, _ = _prepare(config, sources)
    manifest_path = processed.root / MANIFEST_NAME
    payload = json.loads(manifest_path.read_text())
    payload["contract_version"] = "0"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactIntegrityError, match="contract version"):
        _prepare(config, sources)


# --------------------------------------------------------------------------- #
# Keys and deletion                                                            #
# --------------------------------------------------------------------------- #
def test_the_key_is_stable_across_calls_with_the_same_inputs(config, sources):
    context = ExecutionContext.from_config(config)
    mapper = CanonicalColumnMapper.from_config(config)
    schema = build_schema(sources["train"], context.internal_config, mapper.to_external)
    arguments = dict(
        train_manifest=DataPreparation.source_manifest(sources["train"]),
        valid_manifest=DataPreparation.source_manifest(sources["valid"]),
        preprocessing={"categorical_columns": ["segment"]},
        schema=schema,
    )
    assert content_key(**arguments) == content_key(**arguments)


def test_deleting_is_a_call_and_reports_what_it_removed(config, sources):
    processed, _ = _prepare(config, sources)
    root = processed.root.parent
    assert (root / "index.json").is_file()
    assert clear_processed_data(root) == 1
    assert not processed.root.exists()
    assert not (root / "index.json").exists()
    assert clear_processed_data(root) == 0


def test_deleting_one_key_leaves_the_others(config, sources, tmp_path):
    first, _ = _prepare(config, sources)
    other = tmp_path / "train2"
    other.mkdir()
    _frame(300, 11).write_parquet(other / "part-0.parquet")
    sources["train"] = ParquetSource.resolve(other)
    second, _ = _prepare(config, sources)

    assert clear_processed_data(first.root.parent, key=first.key) == 1
    assert not first.root.exists()
    assert second.root.is_dir()
    index = json.loads((first.root.parent / "index.json").read_text())
    assert set(index) == {second.key}
