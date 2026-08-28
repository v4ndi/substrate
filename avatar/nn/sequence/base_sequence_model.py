import torch
import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.feature_encoder.base_feature_encoder import BaseSequenceFeatureEncoder


class BaseSequenceBackbone(nn.Module):
    """Abstract base class for sequence processing backbones.

    This class defines the interface for all sequence processing backbones that can be used
    with the BaseSequenceModel framework. Child classes must implement the forward pass.

    Note:
        This is an abstract class - child classes must implement the forward method.

    Example:
        >>> class MyBackbone(BaseSequenceBackbone):
        ...     def forward(self, inputs_embeds, attention_mask=None):
        ...         # Implementation here
        ...         return processed_sequence
    """

    def __init__(self):
        """Initialize the base sequence backbone."""
        super().__init__()

    def forward(
        self, inputs_embeds: torch.FloatTensor, attention_mask: torch.LongTensor = None
    ):
        """Process input sequence embeddings through the backbone.

        Args:
            inputs_embeds: Embedded input sequence
                shape: (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask
                shape: (batch_size, sequence_length)

        Returns:
            Processed sequence output (implementation specific)

        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Forward method must be implemented by child classes")


class BaseSequenceModel(nn.Module):
    """Base class for sequence models combining feature encoding and sequence processing.

    This class provides a framework for models that:
    1. Encode raw features into embeddings (via feature_encoder)
    2. Process the sequence (via backbone)

    Args:
        feature_encoder (BaseSequenceFeatureEncoder): Module for converting raw features to embeddings
        backbone (BaseSequenceBackbone): Module for sequence processing

    Note:
        This is an abstract class - child classes must implement the forward method.

    Example:
        >>> feature_encoder = MyFeatureEncoder()
        >>> backbone = MyBackbone()
        >>> model = BaseSequenceModel(feature_encoder, backbone)
    """

    def __init__(
        self,
        feature_encoder: BaseSequenceFeatureEncoder,
        backbone: BaseSequenceBackbone,
    ) -> None:
        super().__init__()
        self.feature_encoder = feature_encoder
        self.backbone = backbone

    def forward(self, seq_features: EventSequenceBatch):
        raise NotImplementedError("Forward method must be implemented by child class")
