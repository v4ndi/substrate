"""Event-sequence model stack, split into three sub-packages.

- ``fmlib.nn.sequential.event_encoder`` -- raw event features -> ``(B, S, H)``
  (``EventEncoder``, feature-attention aggregation, temporal encoding).
- ``fmlib.nn.sequential.backbone`` -- ``(B, S, H)`` -> contextualised
  ``(B, S, H)`` (``SequenceBackbone`` protocol, ``BaseBackbone``).
- ``fmlib.nn.sequential.model`` -- assembles encoder + backbone into a model
  (``BaseSequenceModel``, ``TransformersWrapper``).

The full public API is re-exported here so ``fmlib.nn.sequential.<Name>`` keeps
working (Hydra ``_target_`` configs, external imports).
"""

from fmlib.nn.sequential.backbone import BaseBackbone, SequenceBackbone
from fmlib.nn.sequential.event_encoder import (
    BaseEventEncoder,
    EventAggregator,
    EventEncoder,
    IntraFeatureAttention,
    build_event_attention_mask,
)
from fmlib.nn.sequential.model import BaseSequenceModel, TransformersWrapper

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
