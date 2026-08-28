from collections import defaultdict
from dataclasses import fields

from avatar.metrics.base import BaseMetric


class UniversalLossesMetric(BaseMetric):
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
