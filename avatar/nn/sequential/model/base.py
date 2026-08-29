import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.sequential.backbone.base import SequenceBackbone
from avatar.nn.sequential.event_encoder.base import BaseEventEncoder


class BaseSequenceModel(nn.Module):
    """Base class for sequence models combining event encoding and sequence processing.

    This class provides a framework for models that:
    1. Encode raw features into embeddings (via event_encoder)
    2. Process the sequence (via backbone)

    Args:
        event_encoder (BaseEventEncoder): Module for converting raw features to embeddings
        backbone (SequenceBackbone): Module for sequence processing

    Note:
        This is an abstract class - child classes must implement the forward method.

    Example:
        >>> event_encoder = MyEventEncoder()
        >>> backbone = MyBackbone()
        >>> model = BaseSequenceModel(event_encoder, backbone)
    """

    def __init__(
        self,
        event_encoder: BaseEventEncoder,
        backbone: SequenceBackbone,
    ) -> None:
        super().__init__()
        self.event_encoder = event_encoder
        self.backbone = backbone

    @property
    def feature_encoder(self) -> BaseEventEncoder:
        """Deprecated alias for :attr:`event_encoder` (dropped with the shims)."""
        return self.event_encoder

    def forward(self, seq_features: EventSequenceBatch):
        raise NotImplementedError("Forward method must be implemented by child class")
