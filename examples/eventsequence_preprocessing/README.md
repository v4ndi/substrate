# Event-sequence preprocessing: Spark and local backends

`fmlib.preprocessing` has two interchangeable implementations of the
event-sequence preprocessor:

| | `fmlib.preprocessing.spark.pipeline.EventSequencePreprocessor` | `fmlib.preprocessing.local.EventSequencePreprocessor` |
|---|---|---|
| engine | PySpark on a cluster | pyarrow + numpy + pandas, one machine |
| needs | Spark / YARN / JVM | nothing extra |
| memory | distributed | streaming fit; `transform` shards by `id` hash into bounded buckets |
| event order in output | best-effort (see below) | **deterministically time-sorted** |

Both produce the **same artifact** (`dump()` / `load()`) and the same per-user
lists.

## What the preprocessor does

Input is one row per event. `fit` learns the `LabelEncoder` / `StandardScaler`
statistics (streaming pass, shared with the tabular pipeline). `transform`:

1. label-encode categorical columns, standardize numeric columns
   (no cumulative `offset_map` here — the sequence model uses per-column
   embeddings, so ids stay column-local);
2. sort events by `(id_column, event_time_column)`;
3. scale `event_time_column` to `unix_seconds / {days|weeks|months}` (float32);
4. `groupby(id_column, *groupby_columns)` and collect every attribute into a
   time-ordered list.

Output: one row per `id` (× `groupby_columns`), each feature a `list<…>` column,
plus `event_time_column` and the event-type id column as lists.

### Ordering note

Spark's `sort().groupBy().collect_list()` does **not** guarantee event order —
the group-by shuffle discards the sort. In practice it is usually sorted on small
data but not always. The local backend sorts on the full-precision raw timestamp
and is always monotonic. `run_both_backends.py` therefore compares the per-user
event *multiset* (co-sorting each parallel list by timestamp) and separately
asserts the local output is time-ordered.

## Run it

```bash
python examples/eventsequence_preprocessing/generate_data.py          # 6k users
python examples/eventsequence_preprocessing/run_both_backends.py
```

Local runs unconditionally; Spark runs when a compatible JDK (8/11/17) is found.

### Expected output

```
events=…  [local] users=6000  columns_meta={'mcc': {'type': 'categorical', ...}, ...}
[fit] statistics + columns_meta identical across backends
[cross-load] dump() from one backend -> load() into the other:
  OK  spark.fit          vs local.fit  (6000 users, local time-sorted 6000/6000)
  OK  spark.fit          vs local.load(spark.dump())  (6000 users, ...)
  OK  spark.load(local.dump()) vs local.fit  (6000 users, ...)
  OK  spark.fit          vs local (8 hash buckets)  (6000 users, ...)
All backends and cross-loaded artifacts produce matching sequences.
```

## Typical usage

```python
from fmlib.preprocessing.local import EventSequencePreprocessor
import yaml

pp = EventSequencePreprocessor(
    categorical_columns=["mcc", "direction"],
    numeric_columns=["amount"],
    event_time_column="timestamps",
    id_column="epk_id",
    event_type_ids_column="event_ids",
    time_unit="days",
    label_encoder_kwargs={"frequency_encoder": True},
)
pp.fit("/data/events")
pp.transform("/data/events", "/data/sequences")          # n_buckets auto from row count
yaml.safe_dump(pp.dump(), open("artifacts/sequence_preprocessor.yaml", "w"))

pp = EventSequencePreprocessor.load(
    yaml.safe_load(open("artifacts/sequence_preprocessor.yaml"))
)
```

`pp.columns_meta` gives the `{column: {type, n_classes}}` map the sequence
embedding layer needs.

## Tuning for large data

* `n_buckets` (auto: `ceil(rows / 5_000_000)`) — `transform` hashes `id` into
  this many temp shards, sorts + groups each independently, so peak memory is one
  shard, not the whole dataset. Pass it explicitly to override.
* Pre-partitioning the input parquet by `id` (bucketed upstream) makes the shard
  step a no-op and is the recommended layout for 100M-row event logs.
* `batch_rows` — streaming chunk size for `fit` and the pass-1 shard write.
