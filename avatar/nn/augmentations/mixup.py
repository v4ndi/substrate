import random

import torch
import torch.distributions as dist
import torch.nn.functional as F


class MixupEmbeddingProcessor:
    """Performs mixup augmentation on embeddings and their corresponding targets.

    Mixup is a data augmentation technique that creates convex combinations of
    pairs of samples and their targets. This implementation specifically:
    - Applies mixup to a subset of the batch (controlled by augmentation_ratio)
    - Can be configured to prefer mixing with positive class samples
    - Handles both embeddings and target labels properly

    Args:
        alpha: Parameter for Beta distribution (Beta(α, α)).
               Higher values make the mixup coefficients more concentrated around 0.5.
               Default: 1.0
        augmentation_ratio: Fraction of batch samples to augment.
                          Must be in [0, 1]. Default: 0.1
        augm_with_pos_targets: Whether to prioritize mixing with positive class samples.
                             If True, will try to mix augmented samples with positive
                             targets when available. Default: True

    Example:
        >>> processor = MixupEmbeddingProcessor(alpha=0.4, augmentation_ratio=0.5)
        >>> embeddings = torch.randn(32, 128)  # batch_size=32, embedding_dim=128
        >>> targets = torch.randint(0, 2, (32, 1))  # binary classification
        >>> aug_embeddings, aug_targets = processor(embeddings, targets)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        augmentation_ratio: float = 0.1,
        augm_with_pos_targets=True,
    ):
        self.distribution = dist.Beta(alpha, alpha)
        self.augmentation_ratio = augmentation_ratio
        self.augm_with_pos_targets = augm_with_pos_targets

    def forward(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies mixup augmentation to embeddings and targets.

        Args:
            embeddings: Input embeddings tensor of shape [batch_size, ...]
            targets: Target labels tensor of shape [batch_size, 1] (values 0 or 1)
        """
        assert embeddings is not None and targets is not None
        batch_size = embeddings.shape[0]
        num_augmentation_samples = int(batch_size * self.augmentation_ratio)

        augm_idxs = random.sample(range(batch_size), num_augmentation_samples)
        other_augm_idxs = random.sample(range(batch_size), num_augmentation_samples)
        if self.augm_with_pos_targets:
            positive_targets_idxs = torch.where(targets == 1)[0]
            if len(positive_targets_idxs) > num_augmentation_samples:
                other_augm_idxs = positive_targets_idxs[
                    :num_augmentation_samples
                ]  # temporary solution for simplicity
            elif len(positive_targets_idxs) > 0:
                other_augm_idxs[: len(positive_targets_idxs)] = (
                    positive_targets_idxs.tolist()
                )

        coefs = self.distribution.sample((num_augmentation_samples,)).to(
            embeddings.device
        )
        embeddings[augm_idxs] = embeddings[augm_idxs] * coefs.unsqueeze(
            -1
        ) + embeddings[other_augm_idxs] * (1 - coefs.unsqueeze(-1))
        targets = F.one_hot(targets, num_classes=2).to(targets.device).to(torch.float)
        targets[augm_idxs] = targets[augm_idxs] * coefs.unsqueeze(-1) + targets[
            other_augm_idxs
        ] * (1 - coefs.unsqueeze(-1))

        return embeddings, targets
