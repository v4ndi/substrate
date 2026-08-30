# avatar

Foundation models for tabular and event-sequence customer data: a tabular
transformer (`STEv2`) with external-embedding support, event-sequence models,
uplift/multi-task pipelines, and a preprocessing layer with two interchangeable
backends.

## Install

```bash
python -m pip install -e ".[spark,dev]"
```

Extras:

| extra      | pulls in            | needed for                                              |
|------------|---------------------|--------------------------------------------------------|
| `spark`    | `pyspark==3.5.0`    | `avatar.preprocessing.spark` (also needs a JDK 8/11/17) |
| `catboost` | `catboost`          | `avatar.metrics.campaign` CatBoost benchmark            |
| `dev`      | `pytest`, `ruff`, … | running the test suite / pre-commit                     |

`requirements.txt` is a convenience wrapper that installs `-e .[spark,catboost,dev]`;
the authoritative dependency versions are in `pyproject.toml`.

> Spark 3.5 does not run on JDK 21+. Point `JAVA_HOME` at a JDK 8, 11 or 17.

## Layout

| path                     | what                                                        |
|--------------------------|------------------------------------------------------------|
| `avatar/nn/`             | model building blocks (tabular / sequence / embeddings)     |
| `avatar/pipeline/`       | task pipelines (classification, uplift, multi-task, next-k) |
| `avatar/preprocessing/`  | `spark` and `local` (pyarrow+numpy) backends, shared `base` |
| `avatar/data/`           | datasets, collate fns, parquet IO                           |
| `avatar/metrics/`        | training / campaign / uplift metrics                        |
| `avatar/train/`          | training loop, callbacks, checkpoints (Hydra + torch.distributed) |
| `examples/`              | runnable end-to-end examples (see `examples/README.md`)     |
| `docs/`                  | topic notes (parquet, event-sequence batch, checkpointing)  |

## Tests

```bash
python -m pytest                 # default: skips @slow; Spark tests auto-skip without a JDK
python -m pytest -m slow         # the long multi-worker sharding sweeps
JAVA_HOME=/path/to/jdk17 python -m pytest tests/spark tests/local
```

`tests/conftest.py` auto-discovers a Spark-compatible JDK (checks
`$SPARK_JDK`, common sdkman / system locations, then `$JAVA_HOME`); when none is
found the Spark and Spark-parity tests are skipped rather than failing.

## Preprocessing backends

`avatar.preprocessing` ships the same API on two engines:

- `avatar.preprocessing.spark` — PySpark, for cluster-scale fit/transform.
- `avatar.preprocessing.local` — pyarrow + numpy, single machine, streaming
  (handles datasets larger than RAM), no Spark/JVM.

Artifacts (`dump()` / `load()`) are interchangeable between the two. See
`examples/tabular_preprocessing/` and `examples/eventsequence_preprocessing/`.
