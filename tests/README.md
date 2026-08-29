# Tests

```bash
python -m pytest                    # everything runnable here without extra setup
python -m pytest -m slow            # long multi-worker sharding sweeps (deselected by default)
```

## What needs what

| area                     | requires                              | without it                    |
|--------------------------|---------------------------------------|-------------------------------|
| `tests/data`, `tests/nn`, `tests/pipeline`, `tests/metrics` | core deps only | run always |
| `tests/local` (pure)     | pyarrow + numpy                       | run always                    |
| `tests/local` (parity)   | `[spark]` + JDK 8/11/17               | `skip`                        |
| `tests/spark`            | `[spark]` + JDK 8/11/17               | `skip`                        |

## Spark / JDK

Spark 3.5 does not start on JDK 21+. `tests/conftest.py::_ensure_java` probes,
in order: `$SPARK_JDK`, a pinned sdkman Temurin 17 path, common system JVM
locations, then `$JAVA_HOME` — and points `JAVA_HOME` at the first JDK whose
major version is 8, 11 or 17. If none qualifies, the `spark_session` / `spark`
fixtures call `pytest.skip`, so Spark tests never hard-fail the run.

Force a specific JDK:

```bash
SPARK_JDK=/path/to/jdk17 python -m pytest tests/spark
```

## Synthetic data

`tests/conftest.py::generate_sequence_dataset` writes partitioned parquet event
logs with pandas/pyarrow (no Spark), exposed through the `synth_sequence_dataset`
fixture. `tests/local/conftest.py` has the tabular / sequence table fixtures for
the local preprocessing backend.
