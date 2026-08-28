import torch
import torch.nn as nn
import xxhash


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
