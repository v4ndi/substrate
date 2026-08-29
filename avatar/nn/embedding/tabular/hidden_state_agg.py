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
        """
        Args:
            hidden_states torch.FloatTensor: batch_size x hidden_dim
            embeddings torch.FloatTensor: batch_size x n_features x embedding_dim
        """
        raise NotImplementedError("This must be implemented in the subclass.")


class LayerNormConcatenate(BaseHiddenStateAggregator):
    """Apply layer normalization and concatenate with embeddings
    Args:
        hidden_state_dim: int - dim of the hidden state
        embedding_dim: int - dim of the embeddings
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
        """
        Args:
            hidden_states (torch.FloatTensor): batch_size x hidden_dim
            embeddings (torch.FloatTensor): batch_size x n_features x embedding_dim

        Returns:
            batch_size x ( n_features + 1) x embedding_dim
        """
        hidden_states = self.layer_norm(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(hidden_states)

        return torch.cat((hidden_states.unsqueeze(1), embeddings), dim=1)


class LayerNormSum(BaseHiddenStateAggregator):
    """Apply layer normalization, broadcast and sum with embeddings
    Args:
        hidden_state_dim: int - dim of the hidden state
        embedding_dim: int - dim of the embeddings
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
        """
        Args:
            hidden_states (torch.FloatTensor): batch_size x hidden_dim
            embeddings (torch.FloatTensor): batch_size x n_features x embedding_dim

        Returns:
            batch_size x ( n_features) x embedding_dim
        """
        hidden_states = self.layer_norm(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(hidden_states)

        return hidden_states.unsqueeze(1) + embeddings
