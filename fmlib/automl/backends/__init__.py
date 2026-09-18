"""Model backend implementations for AutoML tasks."""

#: Backend families whose fit path needs the whole split in memory, so the
#: training coordinator materializes it for them. Anything else receives a
#: source descriptor and streams. Named here rather than branched on inline so
#: there is one place that says which families read their own data.
MATERIALIZED_BACKENDS = frozenset({"boosting"})

__all__ = ["MATERIALIZED_BACKENDS"]
