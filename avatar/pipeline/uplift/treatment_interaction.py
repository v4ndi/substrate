import torch
import torch.nn as nn


class BaseTreatmentInteraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, states, treatment_embeddings):
        """
        Args:
            states torch.FloatTensor: hidden states of size (batch_size, seq_len, hidden_dim)
            treatment_embeddings torch.FloatTensor: embeddings of (batch_size, 1, hidden_dim)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()


class ConcatTreatmentInteraction(BaseTreatmentInteraction):
    def __init__(self):
        super().__init__()

    def forward(self, states, treatment_embeddings):
        return torch.cat([states, treatment_embeddings], dim=1)


class ElementwiseTreatmentInteraction(BaseTreatmentInteraction):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, states, treatment_embeddings):
        return self.layer_norm(states * treatment_embeddings)


class SumTreatmentInteraction(BaseTreatmentInteraction):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, states, treatment_embeddings):
        return self.layer_norm(states + treatment_embeddings)


class IgnoreTreatmentInteraction(BaseTreatmentInteraction):
    def __init__(self):
        super().__init__()

    def forward(self, states, treatment_embeddings):
        return states
