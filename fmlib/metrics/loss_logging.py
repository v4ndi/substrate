"""Log every loss field an output dataclass happens to carry."""

from collections import defaultdict
from dataclasses import fields

from fmlib.metrics.base import ScalarMetric


class UniversalLossesMetric(ScalarMetric):
    """Running mean of every ``*loss`` field on the model's output dataclass.

    Discovers the fields by reflection rather than by name, so a pipeline that
    grows an auxiliary loss starts reporting it without touching the config.
    The field named exactly ``loss`` is reported as ``basic_loss`` to keep it
    apart from the trainer's own ``loss`` entry.

    Fields that are ``None`` on a given batch are skipped, and each field is
    averaged over **the batches where it actually appeared** — it keeps its own
    counter. Dividing by a counter shared across fields understated an
    intermittent loss by exactly the fraction of batches it was missing from.

    Note:
        Because the reported fields are whatever the output carries,
        :attr:`required_outputs` stays ``None``: this metric cannot tell the
        evaluation loop which fields to gather, so a run that logs losses
        gathers the whole output.
    """

    required_inputs = ()
    required_outputs = None

    def __init__(self):
        self.reset()

    def update(self, inputs, outputs) -> None:
        """Fold one batch's losses into the per-field running means."""
        for field in fields(outputs):
            if not field.name.endswith("loss"):
                continue
            loss = getattr(outputs, field.name)
            if loss is None:
                continue
            name = field.name if field.name != "loss" else "basic_loss"
            seen = self.counters[name]
            self.totals[name] = (
                self.totals[name] * seen + loss.cpu().detach().item()
            ) / (seen + 1)
            self.counters[name] = seen + 1

    def compute(self) -> dict[str, float]:
        """Return the per-field means as a plain dict."""
        return dict(self.totals)

    def reset(self) -> None:
        """Clear every field's mean and counter."""
        self.totals: defaultdict[str, float] = defaultdict(float)
        self.counters: defaultdict[str, int] = defaultdict(int)
