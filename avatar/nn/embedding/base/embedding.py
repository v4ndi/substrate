import torch.nn as nn


class BaseEmbedding(nn.Module):
    """Base class for all embedding layers.

    Attributes:
        hidden_size (int): Dimensionality of the embeddings
    """

    def __init__(self, hidden_size: int):
        """Initialize the base embedding.

        Args:
            hidden_size: Dimensionality of the embeddings
        """
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        self.hidden_size = hidden_size
