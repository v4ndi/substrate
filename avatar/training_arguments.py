from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


@dataclass
class TrainingArguments:
    """
    A class to store and manage arguments for training a model.

    Attributes:
        num_epochs (int): The total number of epochs to train the model. Defaults to 1000.
        seed (int): Random seed for reproducibility. Defaults to 42.
        device_specific (bool): DEPRECATED ALWAYS FALSE
            Whether to differ the seed on each device slightly with
            `self.process_index`. Defaults to True.
        max_saved_checkpoints (Optional[int]): Maximum number of checkpoints to save
            during training. Defaults to None.
        start_epoch (Optional[int]): The epoch to start training from. Useful for
            resuming training. Defaults to 0.
        clip_grad_norm (Optional[float]): Maximum norm for gradient clipping.
        checkpoint_state (Optional[str]): Path to a checkpoint state to
            resume training from. Defaults to None.
        model_state (Optional[str]): Path to a model state dict to resume training from.
            Defaults to None.

    Example:
        >>> args = TrainingArguments(num_epochs=500, seed=123, clip_grad_norm=True)
        >>> print(args.num_epochs)
        500
    """

    num_epochs: int = 1000
    seed: int = 42
    device_specific: bool = False  # deprecated, always False
    start_epoch: int | None = 0
    clip_grad_norm: float | None = None
    max_saved_checkpoints: int | None = None
    checkpoint_state: str | None = None
    model_state: str | None = None
    steps_before_evaluation: int | None = None

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.__dict__)
