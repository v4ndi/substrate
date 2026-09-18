"""How the treatment flag reaches the representation.

Every variant takes the feature tokens and the treatment embedding and returns
tokens for the encoder, so swapping one for another is a config change. The
choice matters: concatenation lets attention decide how much treatment
influences each feature, while sum and elementwise force the effect onto every
token.
"""

import torch
import torch.nn as nn


class BaseTreatmentInteraction(nn.Module):
    """Combine feature tokens with the treatment (or group) embedding.

    Subclasses implement :meth:`forward` only; there is no state to set up
    beyond what a particular variant needs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, states, treatment_embeddings):
        """Combine the two into the tokens the encoder will see.

        Args:
            states: Feature tokens, ``(batch, num_features, hidden_dim)``.
            treatment_embeddings: Treatment embedding, ``(batch, 1, hidden_dim)``.

        Returns:
            Tokens for the encoder. The sequence length may change — see
            :class:`ConcatTreatmentInteraction`.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError()


class ConcatTreatmentInteraction(BaseTreatmentInteraction):
    """Append the treatment embedding as one more token.

    The default. Adds 1 to the token count, which is why ``num_features`` in
    ``aggregation_config`` must be one larger than the number of real features
    — the single most common sizing mistake in uplift configs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, states, treatment_embeddings):
        return torch.cat([states, treatment_embeddings], dim=1)


class ElementwiseTreatmentInteraction(BaseTreatmentInteraction):
    """Multiply every feature token by the treatment embedding, then normalise.

    Treatment modulates each feature rather than sitting beside them. Keeps the
    token count unchanged.

    Args:
        hidden_size: Width of the tokens, for the layer norm.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, states, treatment_embeddings):
        return self.layer_norm(states * treatment_embeddings)


class SumTreatmentInteraction(BaseTreatmentInteraction):
    """Add the treatment embedding to every feature token, then normalise.

    The additive counterpart of :class:`ElementwiseTreatmentInteraction`. Keeps
    the token count unchanged.

    Args:
        hidden_size: Width of the tokens, for the layer norm.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, states, treatment_embeddings):
        return self.layer_norm(states + treatment_embeddings)


class IgnoreTreatmentInteraction(BaseTreatmentInteraction):
    """Drop the treatment embedding entirely.

    Used to disable one of the two interaction slots — typically to feed the
    group embedding while ignoring treatment, or as an ablation baseline that
    keeps the rest of the architecture identical.
    """

    def __init__(self):
        super().__init__()

    def forward(self, states, treatment_embeddings):
        return states
