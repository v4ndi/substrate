"""Event-sequence model stack, split into three sub-packages:

- ``avatar.nn.sequential.event_encoder`` -- raw event features -> ``(B, S, H)``
  (``EventEncoder``, feature-attention aggregation, temporal encoding).
- ``avatar.nn.sequential.backbone`` -- ``(B, S, H)`` -> contextualised
  ``(B, S, H)`` (``SequenceBackbone`` protocol, ``BaseBackbone``).
- ``avatar.nn.sequential.model`` -- assembles encoder + backbone into a model
  (``BaseSequenceModel``, ``TransformersWrapper``).

The full public API is re-exported here so ``avatar.nn.sequential.<Name>`` keeps
working (Hydra ``_target_`` configs, external imports).
"""

from avatar.nn.sequential.backbone import BaseBackbone, SequenceBackbone
from avatar.nn.sequential.event_encoder import (
    BaseEventEncoder,
    EventAggregator,
    EventEncoder,
    IntraFeatureAttention,
    build_event_attention_mask,
)
from avatar.nn.sequential.model import BaseSequenceModel, TransformersWrapper

__all__ = [
    "BaseBackbone",
    "BaseEventEncoder",
    "BaseSequenceModel",
    "EventAggregator",
    "EventEncoder",
    "IntraFeatureAttention",
    "SequenceBackbone",
    "TransformersWrapper",
    "build_event_attention_mask",
]
