import torch
import torch.nn as nn


def get_aggregation_layer(name: str, **kwargs):
    """
    Returns an aggregation layer based on the provided name.

    Args:
        name (str): The name of the aggregation layer to be returned.
        **kwargs: Additional keyword arguments to be passed to the aggregation layer.

    Returns:
        The aggregation layer corresponding to the provided name.

    Raises:
        ValueError: If the provided name is not a recognized aggregation layer.
    """
    if name == "sum":
        return Sum(**kwargs)
    elif name == "sum_layernorm":
        return SumLayerNorm(**kwargs)
    elif name == "mean":
        return MeanHiddenState(**kwargs)
    elif name == "last":
        return LastHiddenState()
    elif name == "linear":
        return LinearAggregation(**kwargs)
    elif name == "conv":
        return ConvAggregation(**kwargs)
    else:
        raise ValueError(f"Unkown aggregation name: {name}")


class BaseAggregation(nn.Module):
    """
    Base class for aggregation layers.

    Args:
        layer_idx (int): The index of the layer to be used for aggregation. Defaults to -1.
        output_dim (int): The output dimension of the aggregation layer. Defaults to None.
            None means that input_dim equal to output_dim
    """

    def __init__(self, layer_idx=-1):
        assert layer_idx <= -1, "Positive index to last layer is not supported"
        super().__init__()
        self.layer_idx = layer_idx
        self.output_dim = None

    def expand_attn_mask(self, states, attn_msk=None):
        """
        Expand the attention mask to match the shape of the states.

        Args:
            states (torch.Tensor): The input states.
            attn_msk (torch.Tensor): The attention mask.

        Returns:
            seq_len (torch.Tensor): The sequence length.
            expand_attn (torch.Tensor): The expanded attention mask.
        """
        if attn_msk is None:
            attn_msk = torch.ones(states.shape[:-1], dtype=torch.long).to(states.device)
        b, max_l = attn_msk.shape
        d = states.shape[-1]
        seq_len = attn_msk.sum(dim=1).long()
        expand_attn = attn_msk.unsqueeze(2).expand(b, max_l, d)
        return seq_len, expand_attn

    def apply_expanded_mask(self, states, attn_msk=None):
        """
        Apply the expanded attention mask to the states.

        Args:
            states Union[torch.Tensor, Tuple(Torch.Tensor)]: The input states.
            attn_msk (torch.Tensor): The attention mask.

        Returns:
            output (torch.Tensor): The output states after applying the expanded attention mask.
            seq_len (torch.Tensor): The sequence length.
        """
        if self.layer_idx < -1:
            states = states.hidden_states[self.layer_idx]
        elif hasattr(states, "last_hidden_state"):
            states = states.last_hidden_state
        seq_len, expand_attn = self.expand_attn_mask(states=states, attn_msk=attn_msk)
        output = states * expand_attn
        return output, seq_len


class SumLayerNorm(BaseAggregation):
    """
    Aggregation layer that sums the states and applies layer normalization.

    Args:
        emb_dim (int): The dimension of the embedding.
        **kwargs: Additional keyword arguments to be passed to the base class.
    """

    def __init__(self, emb_dim, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, states, attn_msk=None):
        """
        Forward pass of the aggregation layer.

        Args:
            states Union[torch.Tensor, Tuple(Torch.Tensor)]: The input states.
            attn_msk (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The aggregated and normalized states.
        """
        masked_states, _ = self.apply_expanded_mask(states=states, attn_msk=attn_msk)
        aggregate_hidden_state = masked_states.sum(dim=1)
        return self.layer_norm(aggregate_hidden_state)


class ConvAggregation(BaseAggregation):
    """
    Aggregation layer that sums the states and applies layer normalization.

    Args:
        emb_dim (int): The dimension of the embedding.
        kernel_size (int): The kernel size of the convolutional layer. Defaults to 1.
        **kwargs: Additional keyword arguments to be passed to the base class.
    """

    def __init__(self, emb_dim, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        self.conv_layer = nn.Conv1d(emb_dim, emb_dim, kernel_size)
        self.pool_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, states, attn_msk=None):
        """
        Forward pass of the aggregation layer.

        Args:
            states Union[torch.Tensor, Tuple(Torch.Tensor)]: The input states.
            attn_msk (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The aggregated and normalized states.
        """
        masked_states, _ = self.apply_expanded_mask(states=states, attn_msk=attn_msk)
        aggregate_hidden_state = self.conv_layer(masked_states.permute(0, 2, 1))
        aggregate_hidden_state = self.pool_layer(aggregate_hidden_state)
        return aggregate_hidden_state.squeeze(-1)


class Sum(BaseAggregation):
    """
    Aggregation layer that sums the states.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, states, attn_msk=None):
        """
        Forward pass of the aggregation layer.

        Args:
            states Union[torch.Tensor, Tuple(Torch.Tensor)]: The input states.
            attn_msk (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The aggregated states.
        """
        masked_states, _ = self.apply_expanded_mask(states=states, attn_msk=attn_msk)
        aggregate_hidden_state = masked_states.sum(dim=1)
        return aggregate_hidden_state


class LastHiddenState(BaseAggregation):
    """
    Aggregation layer that returns the last hidden state.
    """

    def __init__(self):
        super().__init__()

    def forward(self, states, attn_msk=None):
        """
        Forward pass of the aggregation layer.

        Args:
            states Union[torch.Tensor, Tuple(Torch.Tensor)]: The input states.
            attn_msk (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The last hidden state.
        """
        masked_states, seq_len = self.apply_expanded_mask(
            states=states, attn_msk=attn_msk
        )
        aggregate_hidden_state = masked_states[
            torch.arange(masked_states.shape[0]), seq_len - 1
        ]
        return aggregate_hidden_state


class MeanHiddenState(BaseAggregation):
    """
    Aggregation layer that returns the mean of the hidden states.
    """

    def __init__(self):
        super().__init__()

    def forward(self, states, attn_msk=None):
        """
        Forward pass of the aggregation layer.

        Args:
            states Union[torch.Tensor, Tuple(Torch.Tensor)]: The input states.
            attn_msk (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            torch.Tensor: The mean of the hidden states.
        """
        masked_states, seq_len = self.apply_expanded_mask(
            states=states, attn_msk=attn_msk
        )
        aggregate_hidden_state = masked_states.sum(dim=1) / seq_len.unsqueeze(1).expand(
            masked_states.shape[0], masked_states.shape[-1]
        )

        return aggregate_hidden_state


class LinearAggregation(BaseAggregation):
    """
    Aggregation layer using linear transformations.
    Could be useful for tabular encoders.
    This aggregation could be used for states with fixed size by second dim.
    Args:
        num_features (int): Number of features by second dim (batch_size, n_features, emb_dim).
        emb_dim (int): Size of embeddings.
    """

    def __init__(self, num_features: int, emb_dim: int):
        super().__init__()
        self.agg_features = nn.Linear(num_features, 1)
        self.act = nn.SELU()
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, states, attn_msk=None):
        """
        Forward pass of the aggregation layer.

        Args:
            states (torch.FloatTensor): The input tensor.
            attn_mask (torch.LongTensor): The attention mask.
        Returns:
            torch.Tensor: The aggregated and transformed tensor.
        """
        masked_states, _ = self.apply_expanded_mask(states=states, attn_msk=attn_msk)
        out = masked_states.permute(0, 2, 1)  # (batch_size, emb_dim, n_features)
        out = self.agg_features(out)  # (batch_size, emb_dim, 1)
        out = out.squeeze(-1)  # (batch_size, emb_dim)
        out = self.act(out)
        out = self.proj(out)
        out = self.layer_norm(out)

        return out
