import pytest
from pyspark.sql import types as T

from avatar.preprocessing.spark.pipeline import TabularPreprocessor


@pytest.fixture
def sample_data(spark):
    data = [
        ("cat1", 10.5, "A"),
        ("cat2", 20.3, "B"),
        ("cat1", 15.2, "A"),
        ("cat3", 12.1, "C"),
    ]
    schema = T.StructType([
        T.StructField("category", T.StringType(), True),
        T.StructField("numeric", T.DoubleType(), True),
        T.StructField("identity", T.StringType(), True),
    ])
    return spark.createDataFrame(data, schema)


def test_initialization():
    # Test basic initialization
    preprocessor = TabularPreprocessor(
        categorical_columns=["cat_col"],
        numeric_columns=["num_col"],
        spec_tokens={"pad": 0},
    )

    assert preprocessor.cat_cols == ["cat_col"]
    assert preprocessor.num_cols == ["num_col"]
    assert preprocessor.spec_tokens == {"pad": 0}
    assert preprocessor.vocab_size == 1  # for pad token

    # Test initialization without required columns
    with pytest.raises(AssertionError):
        TabularPreprocessor(categorical_columns=None, numeric_columns=None)


def test_fit(sample_data):
    preprocessor = TabularPreprocessor(
        categorical_columns=["category", "identity"],
        numeric_columns=["numeric"],
        spec_tokens={"pad": 0},
    )

    preprocessor.fit(sample_data)

    # Check that offset_map was populated
    assert len(preprocessor.offset_map) == 2
    assert "category" in preprocessor.offset_map
    assert "identity" in preprocessor.offset_map

    # Check vocab_size was updated correctly
    # pad + 3 categories in "category" + 3 categories in "identity"
    assert preprocessor.vocab_size == 1 + 4 + 4

    # Check that label encoder was fitted
    assert "category" in preprocessor.label_encoder.values_to_id
    assert len(preprocessor.label_encoder.values_to_id["category"]) == 4


def test_transform(sample_data):
    preprocessor = TabularPreprocessor(
        categorical_columns=["category", "identity"],
        numeric_columns=["numeric"],
        spec_tokens={"pad": 0},
    )
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    # Check output columns
    assert "cat_features" in transformed.columns
    assert "num_features" in transformed.columns

    # Check types
    assert isinstance(transformed.schema["cat_features"].dataType, T.ArrayType)
    assert isinstance(transformed.schema["num_features"].dataType, T.ArrayType)
    assert transformed.schema["cat_features"].dataType.elementType == T.LongType()
    assert transformed.schema["num_features"].dataType.elementType == T.FloatType()

    # Check identity columns functionality
    transformed_with_identity = preprocessor.transform(
        sample_data, identity_cols=["identity"]
    )
    assert "identity" in transformed_with_identity.columns
    assert "source_identity" not in transformed_with_identity.columns


def test_only_cat_features(sample_data):
    preprocessor = TabularPreprocessor(
        categorical_columns=["category", "identity"],
        numeric_columns=None,
        spec_tokens={},
    )
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    # Check output columns
    assert preprocessor.vocab_size == 8
    assert "cat_features" in transformed.columns
    assert "num_features" not in transformed.columns

    # Check types
    assert isinstance(transformed.schema["cat_features"].dataType, T.ArrayType)
    assert transformed.schema["cat_features"].dataType.elementType == T.LongType()


def test_only_num_features(sample_data):
    preprocessor = TabularPreprocessor(
        categorical_columns=None, numeric_columns=["numeric"], spec_tokens={}
    )
    preprocessor.fit(sample_data)
    transformed = preprocessor.transform(sample_data)

    # Check output columns
    assert preprocessor.vocab_size == 0
    assert "cat_features" not in transformed.columns
    assert "num_features" in transformed.columns

    # Check types
    assert isinstance(transformed.schema["num_features"].dataType, T.ArrayType)
    assert transformed.schema["num_features"].dataType.elementType == T.FloatType()


def test_fit_transform(sample_data):
    preprocessor = TabularPreprocessor(
        categorical_columns=["category"],
        numeric_columns=["numeric"],
        spec_tokens={"pad": 0},
    )
    transformed = preprocessor.fit_transform(sample_data)

    assert "cat_features" in transformed.columns
    assert "num_features" in transformed.columns
    assert len(preprocessor.offset_map) == 1
    assert preprocessor.vocab_size == 1 + 3 + 1  # pad + 3 categories + 'unk'


def test_dump_and_load(sample_data):
    preprocessor = TabularPreprocessor(
        categorical_columns=["category", "identity"],
        numeric_columns=["numeric"],
        spec_tokens={"pad": 0, "unk": 1},
    )
    preprocessor.fit(sample_data)

    # Dump the state
    state = preprocessor.dump()

    # Check dumped state
    assert "spec_tokens" in state
    assert "offset_map" in state
    assert "vocab_size" in state
    assert state["vocab_size"] == 10

    # Load into new instance
    new_preprocessor = TabularPreprocessor.load(state)

    # Check loaded state
    assert new_preprocessor.cat_cols == ["category", "identity"]
    assert new_preprocessor.num_cols == ["numeric"]
    assert new_preprocessor.spec_tokens == {"pad": 0, "unk": 1}
    assert new_preprocessor.offset_map == preprocessor.offset_map
    assert new_preprocessor.vocab_size == preprocessor.vocab_size

    # Verify transform works the same
    original_transformed = preprocessor.transform(sample_data)
    loaded_transformed = new_preprocessor.transform(sample_data)

    original_data = original_transformed.collect()
    loaded_data = loaded_transformed.collect()

    for orig, loaded in zip(original_data, loaded_data, strict=False):
        assert orig["cat_features"] == loaded["cat_features"]
        assert orig["num_features"] == loaded["num_features"]


def test_transform_with_identity_columns(sample_data):
    preprocessor = TabularPreprocessor(
        categorical_columns=["category"],
        numeric_columns=["numeric"],
        spec_tokens={"pad": 0},
    )
    preprocessor.fit(sample_data)

    # Transform with identity columns
    transformed = preprocessor.transform(sample_data, identity_cols=["identity"])

    # Check that identity column is preserved
    assert "identity" in transformed.columns
    assert "source_identity" not in transformed.columns

    # Check that the identity column values are unchanged
    original_identities = [row["identity"] for row in sample_data.collect()]
    transformed_identities = [row["identity"] for row in transformed.collect()]
    assert original_identities == transformed_identities

    assert "cat_features" in transformed.columns
