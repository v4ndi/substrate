"""Local-only tests for ``fmlib.preprocessing.local.TabularPreprocessor``."""

import pyarrow as pa
import pytest
import yaml

from fmlib.preprocessing.local import TabularPreprocessor

CAT = ["cat_a", "cat_b"]
NUM = ["num_1", "num_2", "num_3"]
KW = dict(
    categorical_columns=CAT,
    numeric_columns=NUM,
    spec_tokens={"pad": 0},
    label_encoder_kwargs={"frequency_encoder": True},
    standard_scaler_kwargs={"to_log_columns": ["num_2"]},
)


def test_fit_shapes(write_parquet, tabular_table):
    d = write_parquet(tabular_table)
    pp = TabularPreprocessor(**KW).fit(d)
    # pad + (unk + 4 cats - 1 null bucket already in unk) ...
    assert pp.vocab_size == len(pp.spec_tokens) + sum(
        len(pp.label_encoder.values_to_id[c]) for c in CAT
    )
    assert pp.offset_map["cat_a"] == 1
    out = pp.transform(d, identity_cols=["cat_b"])
    assert set(out.column_names) == {
        "epk_id",
        "target",
        "cat_features",
        "num_features",
        "cat_b",
    }
    assert out.num_rows == tabular_table.num_rows
    assert len(out["cat_features"][0]) == 2
    assert len(out["num_features"][0]) == 3


@pytest.mark.parametrize("batch_rows", [128, 3000, 10**9])
def test_streaming_batch_size_invariance(write_parquet, tabular_table, batch_rows):
    d = write_parquet(tabular_table)
    ref = TabularPreprocessor(**KW, batch_rows=4321)
    ref.fit(d)
    ref_out = ref.transform(d, identity_cols=["cat_b"])

    pp = TabularPreprocessor(**KW, batch_rows=batch_rows)
    pp.fit(d)
    assert pp.label_encoder.values_to_id == ref.label_encoder.values_to_id
    # streaming mean/std can wobble by ~1 ULP with batch size (non-associative
    # float summation); the float32-packed transform output is exact.
    for c in NUM:
        assert pp.standard_scaler.mean_std[c]["mean"] == pytest.approx(
            ref.standard_scaler.mean_std[c]["mean"], rel=1e-12, nan_ok=True
        )
        assert pp.standard_scaler.mean_std[c]["std"] == pytest.approx(
            ref.standard_scaler.mean_std[c]["std"], rel=1e-12, nan_ok=True
        )
    assert pp.transform(d, identity_cols=["cat_b"]).equals(ref_out)


def test_dump_load_yaml_roundtrip(write_parquet, tabular_table):
    d = write_parquet(tabular_table)
    pp = TabularPreprocessor(**KW).fit(d)
    cfg = yaml.safe_load(yaml.safe_dump(pp.dump()))
    assert cfg["_backend"] == "local"
    pp2 = TabularPreprocessor.load(cfg)
    assert pp.transform(d).equals(pp2.transform(d))


def test_wide_output(write_parquet, tabular_table):
    d = write_parquet(tabular_table)
    pp = TabularPreprocessor(**KW).fit(d)
    wide = pp.transform(d, output="wide")
    assert set(wide.column_names) == {"epk_id", "target", *CAT, *NUM}
    packed = pp.transform(d)
    # wide categorical column == first element of the packed array, row 0
    assert wide["cat_a"][0].as_py() == packed["cat_features"][0][0].as_py()


def test_unknown_category_and_null_map_to_unk(write_parquet):
    train = pa.table({"c": ["a", "b", "a", "b"], "n": [1.0, 2.0, 3.0, 4.0]})
    test = pa.table({"c": ["a", "z", None], "n": [1.0, 2.0, 3.0]})
    dtr, dte = write_parquet(train, name="tr"), write_parquet(test, name="te")
    pp = TabularPreprocessor(
        categorical_columns=["c"], numeric_columns=["n"], spec_tokens={}
    ).fit(dtr)
    out = pp.transform(dte)
    unk_global = pp.offset_map["c"] + pp.label_encoder.values_to_id["c"]["unk"]
    assert [row[0].as_py() for row in out["cat_features"]] == [
        pp.offset_map["c"] + pp.label_encoder.values_to_id["c"]["a"],
        unk_global,
        unk_global,
    ]


def test_only_numeric_or_only_categorical(write_parquet, tabular_table):
    d = write_parquet(tabular_table)
    only_num = TabularPreprocessor(
        categorical_columns=None, numeric_columns=NUM, spec_tokens={}
    ).fit(d)
    assert only_num.vocab_size == 0
    out = only_num.transform(d)
    assert "num_features" in out.column_names and "cat_features" not in out.column_names

    only_cat = TabularPreprocessor(
        categorical_columns=CAT, numeric_columns=None, spec_tokens={"pad": 0}
    ).fit(d)
    out2 = only_cat.transform(d)
    assert (
        "cat_features" in out2.column_names and "num_features" not in out2.column_names
    )


def test_transform_writes_parquet(write_parquet, tabular_table, tmp_path):
    d = write_parquet(tabular_table)
    pp = TabularPreprocessor(**KW).fit(d)
    dst = str(tmp_path / "out.parquet")
    assert pp.transform(d, output_path=dst) == dst
    import pyarrow.parquet as pq

    assert pq.read_table(dst).num_rows == tabular_table.num_rows
