"""The GoldFish objective."""

import torch
import torch.nn as nn


class GoldFishLoss(nn.Module):
    """Drop every k-th token from the loss.

    A memorisation counter-measure: the model never receives a training signal
    for the masked positions, so verbatim sequences are harder to reproduce.
    """

    def __init__(self, strategy: str, k: int, goldfish_start_position: int = 0):
        """Mask every k-th token out of the loss.

        ``targets`` is not updated in place, so ``apply_goldfish`` can be
        called independently for analysis, debugging or logging.

        Args:
            strategy: The strategy to use for goldfish.
                options implemented:
                    - "static": Ignore every k-th token starting from `goldfish_start_position`.
                    - "seeded_random": Randomly drop tokens with probability 1/k.
            k: The frequency with which tokens are ignored?
            goldfish_start_position: The position to start ignoring tokens from.

        Returns:
            The target with the mask applied and the indices of the dropped tokens.
        """
        super().__init__()
        assert strategy in ["static", "seeded_random"], (
            f"{strategy} goldfish strategy is not implemented. Try 'static' instead."
        )
        self.strategy = strategy
        self.k = k
        self.goldfish_start_position = goldfish_start_position
        self.ignore_index = -100
        self.loss = nn.CrossEntropyLoss(reduction="sum")

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return self.loss(input_tensor, targets)

        device = input_tensor.device
        _, block_size = input_tensor.shape
        masked_input = input_tensor.clone()

        if self.strategy == "static":
            dropped_token_indices = torch.arange(block_size, device=device)[
                self.goldfish_start_position :: self.k
            ].long()
            masked_input[:, dropped_token_indices] = self.ignore_index
        elif self.strategy == "seeded_random":
            random_tensor = torch.randint(
                1, self.k + 1, size=input_tensor.size(), device=device
            )
            dropped_token_indices = random_tensor == self.k
            masked_input[dropped_token_indices] = self.ignore_index

        return self.loss(masked_input, targets)
