"""The backbone contract: hidden states in, hidden states out.

A ``Protocol`` rather than a base class, so a HuggingFace model satisfies it
without inheriting from anything of ours.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class SequenceBackbone(Protocol):
    """Structural type for a sequence-processing backbone.

    ``BaseSequenceModel`` only relies on this call signature; any module that
    matches it works as a backbone, including bare Hugging Face models such as
    ``transformers.GPT2Model``.

    A call must accept ``inputs_embeds`` of shape
    ``(batch_size, sequence_length, hidden_size)`` and an ``attention_mask`` of
    shape ``(batch_size, sequence_length)``, and return an object exposing
    ``last_hidden_state`` (optionally ``hidden_states`` / ``router_logits``).
    """

    def __call__(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ): ...


class BaseBackbone(nn.Module):
    """Optional ``nn.Module`` base for backbones defined in this codebase.

    Subclassing is not required -- see :class:`SequenceBackbone` for the actual
    contract ``BaseSequenceModel`` depends on. This class only pins the forward
    signature for our own implementations.

    Example:
        >>> class MyBackbone(BaseBackbone):
        ...     def forward(self, inputs_embeds, attention_mask=None):
        ...         ...
        ...         return processed_sequence
    """

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
