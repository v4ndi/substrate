"""Log every loss field an output dataclass happens to carry."""

from collections import defaultdict
from dataclasses import fields

from avatar.metrics.base import BaseMetric


class UniversalLossesMetric(BaseMetric):
    """Running mean of every ``*loss`` field on the model's output dataclass.

    Discovers the fields by reflection rather than by name, so a pipeline that
    grows an auxiliary loss starts reporting it without touching the config.
    The field named exactly ``loss`` is reported as ``basic_loss`` to keep it
    apart from the trainer's own ``loss`` entry.

    Fields that are ``None`` on a given batch are skipped, so a loss that only
    applies to some batches is averaged over the batches where it exists.
    """

    def __init__(self):
        self.reset()

    def update(self, inputs, outputs) -> None:
        for field in fields(outputs):
            if not field.name.endswith("loss"):
                continue
            loss = getattr(outputs, field.name)
            if loss is None:
                continue
            loss = loss.cpu().detach().item()
            name = field.name if field.name != "loss" else "basic_loss"
            self.field_dict[name] = (
                self.field_dict[name] * self.update_counter + loss
            ) / (self.update_counter + 1)
        self.update_counter += 1

    def compute(self) -> dict[str, float]:
        return self.field_dict

    def reset(self) -> None:
        self.update_counter = 0
        self.field_dict = defaultdict(int)
