"""How an external client embedding joins the tabular feature tokens.

This is the *early fusion* seam: ``LayerNormConcatenate`` adds the external
vector as one more token (so the aggregation's ``num_features`` grows by one),
``LayerNormSum`` adds it into every token instead.
"""

import torch
import torch.nn as nn


class BaseHiddenStateAggregator(nn.Module):
    """Base Hidden State Aggregator.

    Args:
        hidden_state_dim: int - dim of the external hidden state
        embedding_dim: int - dim of the input embeddings
    """

    def __init__(
        self,
        hidden_state_dim: int,
        embeddings_dim: int,
    ):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.embedding_dim = embeddings_dim

    def forward(self, hidden_states, embeddings):
        """Fuse the external embedding into the feature tokens.

        Args:
            hidden_states: External embedding, ``(batch_size, hidden_dim)``.
            embeddings: Feature tokens, ``(batch_size, n_features, embedding_dim)``.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError("This must be implemented in the subclass.")


class LayerNormConcatenate(BaseHiddenStateAggregator):
    """Normalise the external embedding and append it as one more token.

    Adds a token, so the downstream ``aggregation_config.num_features`` must be
    one larger than the number of real features. A linear projection is added
    automatically when the two widths differ.

    Args:
        hidden_state_dim: Width of the external embedding.
        embedding_dim: Width of the feature tokens.
    """

    def __init__(
        self,
        hidden_state_dim: int,
        embedding_dim: int,
    ):
        super().__init__(
            hidden_state_dim=hidden_state_dim, embeddings_dim=embedding_dim
        )
        self.layer_norm = nn.LayerNorm(hidden_state_dim)
        if hidden_state_dim != embedding_dim:
            self.proj = nn.Linear(hidden_state_dim, embedding_dim)
        else:
            self.proj = None

    def forward(
        self, hidden_states: torch.FloatTensor, embeddings: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Prepend the normalised external embedding to the feature tokens.

        Args:
            hidden_states: External embedding, ``(batch_size, hidden_dim)``.
            embeddings: Feature tokens, ``(batch_size, n_features, embedding_dim)``.

        Returns:
            ``(batch_size, n_features + 1, embedding_dim)``.
        """
        hidden_states = self.layer_norm(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(hidden_states)

        return torch.cat((hidden_states.unsqueeze(1), embeddings), dim=1)


class LayerNormSum(BaseHiddenStateAggregator):
    """Normalise the external embedding and add it into every feature token.

    Unlike :class:`LayerNormConcatenate`, the token count is unchanged, so the
    downstream ``num_features`` stays as it was.

    Args:
        hidden_state_dim: Width of the external embedding.
        embedding_dim: Width of the feature tokens.
    """

    def __init__(
        self,
        hidden_state_dim: int,
        embedding_dim: int,
    ):
        super().__init__(
            hidden_state_dim=hidden_state_dim, embeddings_dim=embedding_dim
        )
        self.layer_norm = nn.LayerNorm(hidden_state_dim)
        if hidden_state_dim != embedding_dim:
            self.proj = nn.Linear(hidden_state_dim, embedding_dim)
        else:
            self.proj = None

    def forward(
        self, hidden_states: torch.FloatTensor, embeddings: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Broadcast the normalised external embedding over the feature tokens.

        Args:
            hidden_states: External embedding, ``(batch_size, hidden_dim)``.
            embeddings: Feature tokens, ``(batch_size, n_features, embedding_dim)``.

        Returns:
            ``(batch_size, n_features, embedding_dim)``.
        """
        hidden_states = self.layer_norm(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(hidden_states)

        return hidden_states.unsqueeze(1) + embeddings
