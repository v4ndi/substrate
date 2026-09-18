"""Feature preprocessing.

Two interchangeable backends, imported lazily so that neither ``pyspark`` nor the
local stack is a hard import cost:

* :mod:`fmlib.preprocessing.spark` -- distributed, requires a Spark environment.
* :mod:`fmlib.preprocessing.local` -- single machine (pyarrow + numpy), streaming.

Artifacts produced by ``dump()`` are compatible across backends.
"""

import importlib

__all__ = ["local", "spark"]  # noqa: F822 -- provided lazily by __getattr__


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
