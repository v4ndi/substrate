import pytest

from avatar.preprocessing.spark import LabelEncoder


@pytest.fixture
def sample_data(spark_session):
    def _sample_data(data, columns):
        return spark_session.createDataFrame(data, columns)

    return _sample_data


def test_initialization():
    # Test initialization with valid columns
    columns = ["col1", "col2"]
    spec_tokens = {"pad": 3, "bos": 1, "eos": 2, "unk": 0}
    encoder = LabelEncoder(columns=columns, spec_tokens=spec_tokens)
    assert encoder.columns == columns
    assert encoder.spec_tokens == spec_tokens
    assert encoder.values_to_id == {col: spec_tokens.copy() for col in columns}


def test_fit(sample_data):
    data = [("a", "x"), ("b", "y")]
    df = sample_data(data, ["col1", "col2"])
    spec_tokens = {"pad": 3, "bos": 1, "eos": 2, "unk": 0}
    encoder = LabelEncoder(
        columns=["col1", "col2"], spec_tokens=spec_tokens, frequency_encoder=False
    )

    encoder.fit(df)

    # Check col1 mappings
    expected_col1_v1 = {"pad": 3, "bos": 1, "eos": 2, "unk": 0, "a": 5, "b": 4}
    expected_col1_v2 = {"pad": 3, "bos": 1, "eos": 2, "unk": 0, "x": 5, "y": 4}
    assert (encoder.values_to_id["col1"] == expected_col1_v1) ^ (
        encoder.values_to_id["col1"] == expected_col1_v2
    )

    # Check col2 mappings
    # expected_col2_v1 = {"pad": 0, "bos": 1, "eos": 2, "unk": 3, "x": 4, "y": 5}
    expected_col2_v2 = {"pad": 0, "bos": 1, "eos": 2, "unk": 3, "x": 5, "y": 4}
    assert (encoder.values_to_id["col2"] == expected_col1_v2) ^ (
        encoder.values_to_id["col2"] == expected_col2_v2
    )


def test_fit_frequency_encoder(sample_data):
    data = [
        (4, 115, "B", 4),
        (0, 123, "B", 4),
        (4, 332, "B", 4),
        (None, 322, "A", 3),
    ]
    df = sample_data(data, ["f1", "nf", "f2", "f3"])
    cat_features = ["f1", "f2", "f3"]
    encoder = LabelEncoder(
        columns=cat_features, spec_tokens={"unk": 0}, frequency_encoder=True
    )
    encoder.fit(df)

    expected_values_to_id = {
        "f1": {"unk": 0, 4: 1, 0: 2},
        "f2": {"unk": 0, "B": 1, "A": 2},
        "f3": {"unk": 0, 4: 1, 3: 2},
    }
    assert encoder.values_to_id == expected_values_to_id


def test_fit_transform_frequency_encoder(sample_data):
    data = [
        (4, 115, "B", 4),
        (0, 123, "B", 4),
        (4, 332, "B", 4),
        (None, 322, "A", 3),
    ]
    df = sample_data(data, ["f1", "nf", "f2", "f3"])
    cat_features = ["f1", "f2", "f3"]
    encoder = LabelEncoder(
        columns=cat_features, spec_tokens={"unk": 0}, frequency_encoder=True
    )
    results = encoder.fit_transform(df).collect()

    expected = {
        "f1": [1, 2, 1, 0],
        "nf": [115, 123, 332, 322],
        "f2": [1, 1, 1, 2],
        "f3": [1, 1, 1, 2],
    }
    for idx, row in enumerate(results):
        print(row, idx)
        for col in cat_features + ["nf"]:
            assert row.asDict()[col] == expected[col][idx]


def test_fit_with_spec_token_collision(sample_data):
    data = [("pad", "unk")]
    df = sample_data(data, ["col1", "col2"])
    spec_tokens = {"pad": 3, "bos": 1, "eos": 2, "unk": 0}
    encoder = LabelEncoder(columns=["col1", "col2"], spec_tokens=spec_tokens)

    with pytest.raises(AssertionError):
        encoder.fit(df)


def test_transform(sample_data):
    data = [("a", "x"), ("b", "z"), (None, None)]
    df = sample_data(data, ["col1", "col2"])
    spec_tokens = {"pad": 3, "bos": 1, "eos": 2, "unk": 0}
    encoder = LabelEncoder(columns=["col1", "col2"], spec_tokens=spec_tokens)

    # Fit on data with known values
    fit_data = [("a", "x"), ("b", "y")]
    fit_df = sample_data(fit_data, ["col1", "col2"])
    encoder.fit(fit_df)

    transformed_df = encoder.transform(df)
    results = transformed_df.collect()

    # Check transformed values
    expected = [
        {"col1": [4, 5], "col2": [4, 5]},  # a, x
        {"col1": [4, 5], "col2": [0]},  # b ->5, z->unk
        {"col1": [0], "col2": [0]},  # nulls->unk
    ]
    for idx, row in enumerate(results):
        for col in ["col1", "col2"]:
            if len(expected[idx][col]) > 1:
                assert (row.asDict()[col] == expected[idx][col][0]) ^ (
                    row.asDict()[col] == expected[idx][col][1]
                )
            else:
                assert row.asDict()[col] == expected[idx][col][0]


def test_fit_transform(sample_data):
    data = [("c", "y"), ("d", "z")]
    df = sample_data(data, ["col1", "col2"])
    spec_tokens = {"pad": 3, "bos": 1, "eos": 2, "unk": 0}
    encoder = LabelEncoder(columns=["col1", "col2"], spec_tokens=spec_tokens)

    transformed_df = encoder.fit_transform(df)
    # Verify transformed values
    results = transformed_df.collect()
    expected = [
        {"col1": [4, 5], "col2": [4, 5]},  # c, y
        {"col1": [4, 5], "col2": [4, 5]},  # d, z (unk in fit data)
        # {"col1": [3], "col2": [3]}   # nulls
    ]
    for idx, row in enumerate(results):
        for col in ["col1", "col2"]:
            if len(expected[idx][col]) > 1:
                assert (row.asDict()[col] == expected[idx][col][0]) ^ (
                    row.asDict()[col] == expected[idx][col][1]
                )
            else:
                assert row.asDict()[col] == expected[idx][col][0]


def test_dump_and_load(sample_data):
    # Create sample data
    data = [("a", "x"), ("b", "y"), ("c", "x")]
    df = sample_data(data, ["col1", "col2"])
    spec_tokens = {"pad": 3, "bos": 1, "eos": 2, "unk": 0}

    original_encoder = LabelEncoder(columns=["col1", "col2"], spec_tokens=spec_tokens)
    df1 = original_encoder.fit_transform(df)

    state = original_encoder.dump()

    loaded_encoder = LabelEncoder.load(state)
    df2 = loaded_encoder.transform(df)

    assert df1.collect() == df2.collect(), (
        "Transformed data should be identical after dump and load"
    )

    # Step 6: Verify that the internal state is preserved
    assert loaded_encoder.__dict__ == original_encoder.__dict__, "Columns should match"


def test_unseen_values(sample_data):
    data = [("new_value",)]
    df = sample_data(data, ["col1"])
    spec_tokens = {"pad": 3, "bos": 1, "eos": 2, "unk": 0}
    encoder = LabelEncoder(columns=["col1"], spec_tokens=spec_tokens)

    # Fit with different data
    fit_data = [("known_value",)]
    fit_df = sample_data(fit_data, ["col1"])
    encoder.fit(fit_df)

    transformed_df = encoder.transform(df)
    results = transformed_df.collect()

    # Unseen value should map to unk (3)
    assert all(row["col1"] == 0 for row in results)


def test_update(sample_data):
    data = [
        (4, 115, "B", 4),
        (0, 123, "B", 4),
        (4, 332, "B", 4),
        (None, 322, "A", 3),
    ]

    df = sample_data(data, ["f1", "nf", "f2", "f3"])

    cat_features = ["f1", "f2", "f3"]
    encoder = LabelEncoder(
        columns=[cat_features[0]], spec_tokens={"unk": 0}, frequency_encoder=True
    )

    encoder.fit(df)
    encoder.update(df, [cat_features[1]])
    encoder.update(df, [cat_features[2]])
    results = encoder.transform(df).collect()

    expected = {
        "f1": [1, 2, 1, 0],
        "nf": [115, 123, 332, 322],
        "f2": [1, 1, 1, 2],
        "f3": [1, 1, 1, 2],
    }
    for idx, row in enumerate(results):
        print(row, idx)
        for col in cat_features + ["nf"]:
            assert row.asDict()[col] == expected[col][idx]
