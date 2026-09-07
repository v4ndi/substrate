"""The tabular-encoder contract: feature tokens in, feature tokens out.

Encoders take embeddings, never a batch, and never compute a loss — that is
what lets the same encoder serve classification, uplift and multi-task.
"""

import torch
import torch.nn as nn

from avatar.outputs import BaseTabularOutput


class BaseTabularEncoder(nn.Module):
    """Base class for tabular feature-token encoders.

    Contextualises a set of feature-token embeddings: ``(B, F, D) -> (B, F, D)``.
    The embedding layer lives outside the encoder
    (``avatar.nn.embedding.tabular``); the pipeline composes
    ``embedding + encoder + aggregation``.

    Contract:
        Input
            inputs_embeds  (B, F, D) - one embedding vector per feature token
            attention_mask (B, F)    - optional, 1 = keep, 0 = pad
        Output
            BaseTabularOutput with ``last_hidden_state`` (B, F, D);
            ``hidden_states`` is populated iff ``output_hidden_states=True``.

    Args:
        hidden_size: embedding dim ``D`` (must match the embedding layer).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> BaseTabularOutput:
        raise NotImplementedError(
            "forward must be implemented by BaseTabularEncoder subclasses"
        )
