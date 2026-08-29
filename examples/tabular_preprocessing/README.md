# Tabular preprocessing: Spark and local backends

`avatar.preprocessing` has two interchangeable implementations of the same
tabular preprocessor:

| | `avatar.preprocessing.spark.pipeline.TabularPreprocessor` | `avatar.preprocessing.local.TabularPreprocessor` |
|---|---|---|
| engine | PySpark on a cluster | pyarrow + numpy, one machine |
| needs | Spark / YARN / JVM | nothing extra (deps already in `requirements.txt`) |
| memory | distributed | streaming, bounded — works on datasets larger than RAM |
| target | any size | tuned for ~8 CPU / 128 GB, up to ~100M rows × ~500 features |
| categorical id order | `collect_set` (non-deterministic) | deterministic (`sorted` / `count_desc` / `first_seen`) |

Both produce the **same artifact** (`dump()` / `load()`) and the **same output**:
`cat_features` (`list<int64>`, cumulative-offset ids for a single shared
embedding) and `num_features` (`list<float32>`, standardized).

## What the preprocessor does

* **categorical** — `LabelEncoder`: `value -> id` per column, unseen value / null
  → `unk` (0); then a cumulative `offset_map` so all columns share one
  `nn.Embedding` of size `vocab_size`.
* **numeric** — `StandardScaler`: optional `signed_log1p` on `to_log_columns`,
  then `(x - mean) / (std + 1e-8)`, then null → `0.0`.
* `identity_cols` are kept in raw form alongside the packed arrays.
* `output="wide"` skips packing and returns one transformed column per feature
  (for dataset-level / on-GPU preprocessing — see [`../../TODO.md`](../../TODO.md)).

## Run it

```bash
# 1. synthetic data (200k rows, 8 categorical + 24 numeric, some nulls)
python examples/tabular_preprocessing/generate_data.py

# 2. fit both backends, cross-load the artifact each way, assert identical output
python examples/tabular_preprocessing/run_both_backends.py
```

`run_both_backends.py` runs the local backend unconditionally and the Spark
backend when a compatible JDK (8/11/17) is found — otherwise it prints the
local-only result and exits.

### Expected output

```
categorical=8  numeric=24  to_log=['num_0', 'num_4', ...]  identity=['cat_0']
[local] vocab_size=74  offset_map={'cat_0': 1, 'cat_1': 12, ...}
[spark] vocab_size=74  offset_map={'cat_0': 1, 'cat_1': 12, ...}
[fit] statistics identical across backends
[cross-load] dump() from one backend -> load() into the other:
  OK  spark.fit          vs local.fit  (200000 rows)
  OK  spark.fit          vs local.load(spark.dump())  (200000 rows)
  OK  spark.load(local.dump()) vs local.fit  (200000 rows)
All backends and cross-loaded artifacts produce identical output.
```

## Typical usage

```python
from avatar.preprocessing.local import TabularPreprocessor
import yaml

pp = TabularPreprocessor(
    categorical_columns=cat_cols,
    numeric_columns=num_cols,
    spec_tokens={"pad": 0},
    standard_scaler_kwargs={"to_log_columns": log_cols},
    label_encoder_kwargs={"frequency_encoder": True},
)
pp.fit("/data/train")                                   # one streaming pass
pp.transform("/data/train", "/data/train_processed", identity_cols=[...])
pp.transform("/data/valid", "/data/valid_processed")
pp.transform("/data/test",  "/data/test_processed")

yaml.safe_dump(pp.dump(), open("artifacts/tabular_preprocessor.yaml", "w"))

# at inference / in the other backend:
pp = TabularPreprocessor.load(yaml.safe_load(open("artifacts/tabular_preprocessor.yaml")))
```

`fit` accepts a path, a glob, a directory, a list of those, or a
`pyarrow.dataset.Dataset`. `transform` writes parquet when given `output_path`,
otherwise returns a `pyarrow.Table` (fine for small data / `output="wide"`).

## Tuning for large data

* `batch_rows` (default 250k) — rows per streaming chunk; peak memory ≈
  `batch_rows × n_columns × 8 B`.
* `max_cardinality` / `on_overflow="topk"` on the label encoder — guard against
  id-like columns that should use a hash embedding instead of label encoding.
