# Tests

```bash
python -m pytest                    # everything runnable here without extra setup
python -m pytest -m ""              # the lot: clears the default deselection
python -m pytest -m slow            # long multi-worker sharding sweeps only
```

## Run modes

Three markers are deselected by default, so the bare `pytest` stays runnable on
a laptop with no hardware and no cluster access. `-m ""` clears the expression
and selects everything; whatever the machine still cannot do is **skipped with
a reason**, never failed.

| marker    | what it needs                | selected by            | without it |
|-----------|------------------------------|------------------------|------------|
| _(none)_  | core deps                    | always                 | —          |
| `slow`    | minutes, several processes   | `-m slow`, `-m ""`     | deselected |
| `gpu`     | a CUDA device                | `-m gpu`, `-m ""`      | skipped    |
| `cluster` | a real Osiris scheduler      | `FMLIB_OSIRIS_TESTS=1` | skipped    |

`--strict-markers` is on: a marker that is not declared in `pyproject.toml` is
an error, not a marker that silently does nothing.

## When to run what

| moment          | command                                        | time    |
|-----------------|------------------------------------------------|---------|
| while editing   | `pytest tests/<area>`                          | seconds |
| before a commit | `pytest`                                       | ~2.5 min |
| before a push   | `pytest -m ""`                                 | ~9.5 min |
| before a release| `FMLIB_OSIRIS_TESTS=1 pytest -m ""` + wheel smoke | ~15 min |

The boosting parity gate is part of `-m slow` and can be run alone:

```bash
python -m pytest -m slow tests/automl/test_boosting_parity.py
```

## Order, seeds and flakes

Test order is randomised on every run. A dependency between two tests — one
leaving state the next relies on — cannot settle in unnoticed, which matters
here because until this was switched on the suite had only ever executed in one
order.

Every run prints its seed. Reproduce a failure exactly:

```bash
python -m pytest --randomly-seed=12345    # the seed the failing run printed
python -m pytest -p no:randomly           # fixed order, for one run
```

Switched on only after it was shown to hold: five randomised runs of the
default selection and three of everything, all green, plus three ordinary full
runs the same day. Eleven clean runs, no flake.

## Running it in parallel

`pytest-xdist` is installed but off by default, because a sequential run is
easier to read when something fails. Measured on this machine:

| selection | sequential | `-n 4` |
|-----------|-----------|--------|
| default   | ~2:30     | 1:45   |
| `-m ""`   | ~12:10    | 5:36   |

```bash
python -m pytest -m "" -n 4
```

Both were green, including the torchrun subprocess tests and the GPU tests
sharing one card. On a smaller card four workers may not fit; drop to `-n 2`.

## Timeouts

Every test has a 300-second ceiling (`pytest-timeout`). It is a deadlock
catcher, not a budget: the slowest test in the repo runs for 26 seconds, so
anything past five minutes is hung. Tests that launch subprocesses set their
own, shorter, timeout — so the failure says what hung rather than only that
something did.

## The Osiris contract

There is no cluster here, so the remote path is tested against a stand-in. A
stand-in written alongside the producer shares its assumptions, so the contract
lives on its own in `tests/automl/osiris_contract.py` and both fakes call it
before recording anything. Every test that submits a job checks the request
shape as a side effect.

The contract is read off the submit path, not off a live Osiris. It freezes what
we believe and catches drift; confirming the belief needs cluster access, and
the checklist for that is in `docs/decisions/testing_plan.md`.

## The built package

`tests/test_packaging.py` builds a wheel and checks it carries every data file
the tree has, carries nothing outside the package, and trains and scores in a
process where the working copy is not importable. The build starts from a clean
`build/` **and** a clean `*.egg-info` — a stale source listing inside the egg-info will
otherwise ship data whose declaration has been removed, and the test passes while the
package is broken.

## Reading in a second process

`tests/automl/test_fresh_interpreter.py` writes in the test process and reads in
a subprocess started from scratch, with `cwd=/`. It covers scoring from a saved
artifact (both backend families), reattaching to an entity the way the recovery
docs say to, calibration, and the refusal to load an artifact with a file
missing. Nothing here passes objects between the two sides — only a path.

## Frozen documents

`tests/automl/test_contracts.py` snapshots every document that crosses a
boundary — the Osiris run spec and create request, the submit log, the job
handle, the operation record, a trial's spec and result, the processed-data
completion marker, and both artifact layouts. Each snapshot is produced by the
real producer, never assembled by the test.

```bash
AUTOML_CONTRACT_RECORD=1 pytest -m "" tests/automl/test_contracts.py
```

Re-recording is a reviewable diff. Floats are normalised to `<float>` on
purpose: what a model scored is the parity gate's question, not this one's.

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
