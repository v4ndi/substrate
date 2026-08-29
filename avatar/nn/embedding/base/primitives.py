"""Low-level embedding primitives shared by the sequential and tabular stacks."""

import torch
import torch.nn as nn
import xxhash
from torch import Tensor
from torch.nn.parameter import Parameter


def _check_input_shape(x: Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValueError(
            f"The input must have at least one dimension, however: {x.ndim=}"
        )
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            "The last dimension of the input was expected to be"
            f" {expected_n_features}, however, {x.shape[-1]=}"
        )


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    Shape

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    Examples
    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, hidden_size: int) -> None:
        """
        Args:
            n_features: the number of continuous features.
            hidden_size: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features=}")
        if hidden_size <= 0:
            raise ValueError(f"d_embedding must be positive, however: {hidden_size=}")

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, hidden_size))
        self.bias = Parameter(torch.empty(n_features, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_shape(x, self.weight.shape[0])
        return torch.addcmul(self.bias, self.weight, x[..., None])


class HashEmbedding(nn.Module):
    """Embedding layer using multiple hash functions for each input ID.

    Args:
        vocab_size (int): Number of unique tokens.
        embedding_dim (int): Size of each embedding vector.
        num_hashes (int): Number of hash functions to use.
        seed (int, optional): Random seed for hashing. Defaults to 42.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, num_hashes: int, seed=42):
        super().__init__()
        self.num_hashes = num_hashes
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def _get_hashes(self, num, device):
        """Compute multiple hashes for a given input number.

        Args:
            num (int): Input number to hash.
            device (torch.device): Device to place the hash tensors.

        Returns:
            list[torch.Tensor]: List of hash indices.
        """
        hashes = []
        encoded_num = str(num).encode("utf-8")
        for i in range(self.num_hashes):
            current_hash = (
                xxhash.xxh64(encoded_num, seed=i).intdigest() % self.vocab_size
            )
            current_hash = torch.tensor(current_hash, dtype=torch.long).to(device)
            hashes.append(current_hash)
        return hashes

    def _get_embed(self, hashes: list[torch.Tensor]):
        """Averages the embeddings for the given hashes.
        Args:
            hashes (list[torch.Tensor]): List of hash indices.

        Returns:
            torch.Tensor: Averaged embedding vector.
        """
        return sum([self.embedding(hash) for hash in hashes]) / len(hashes)

    def forward(self, ids: torch.LongTensor):
        """Computes embeddings for a batch of token IDs.

        Args:
            ids (torch.LongTensor): Batch of input IDs.

        Returns:
            torch.Tensor: Batch of embedding vectors.
        """
        hashes = [
            self._get_hashes(id, ids.device) for id in ids
        ]  # batch of tensors of size num_hashes
        embeds = torch.stack([
            self._get_embed(hashes) for hashes in hashes
        ])  # batch of embeddings
        return embeds
