"""``TabularBatch``: categorical ids, numeric values, optional hidden states."""

import torch

from fmlib.data.base.batch import move_to_device

__all__ = ["TabularBatch", "UpliftTabularBatch", "move_to_device"]


class TabularBatch:
    """A batch of tabular data containing categorical and numerical features.

    This class encapsulates a batch of tabular data for machine learning tasks,
    separating categorical features (as LongTensor) and numerical features
    (as FloatTensor).
    Optionally includes a hidden states or external embeddings.

    Args:
        cat_features (torch.LongTensor): Batch of categorical features
            with shape [batch_size, n_cat_features]. Must be integer/long type.
        num_features (torch.FloatTensor): Batch of numerical features
            with shape[batch_size, n_num_features]. Must be float type.
        hidden_state (dict[str, torch.FloatTensor], optional):
            dict of hidden_states. Shape of a hidden_s [batch_size, hidden_dim]. Defaults to None.

    Attributes:
        cat_features (torch.LongTensor): Categorical features tensor.
        num_features (torch.FloatTensor): Numerical features tensor.
        hidden_states (torch.FloatTensor): Optional hidden state tensor.
        n_features (int): Total number of features (categorical + numerical).
    """

    def __init__(
        self,
        cat_features: torch.LongTensor = None,
        num_features: torch.FloatTensor = None,
        hidden_states: dict[str, torch.Tensor] | None = None,
    ):
        assert (
            cat_features is not None
            or num_features is not None
            or hidden_states is not None
        ), "Must provide either categorical, numerical or hidden_state features."
        self._cat_features = cat_features
        self._num_features = num_features
        self._hidden_states = hidden_states

    def to(self, device: torch.device) -> None:
        return TabularBatch(
            cat_features=move_to_device(self.cat_features, device),
            num_features=move_to_device(self.num_features, device),
            hidden_states=move_to_device(self.hidden_states, device),
        )

    @property
    def cat_features(self) -> torch.LongTensor:
        """Categorical feature ids, ``(batch_size, n_cat_features)``."""
        return self._cat_features

    @property
    def num_features(self) -> torch.FloatTensor:
        """Standardised numeric features, ``(batch_size, n_num_features)``."""
        return self._num_features

    @property
    def hidden_states(self) -> dict[str, torch.Tensor]:
        """External embeddings by column name, each ``(batch_size, hidden_dim)``."""
        return self._hidden_states

    @property
    def n_features(self) -> int:
        """int: Total number of features (sum of categorical and numerical features)."""
        return self._num_features.shape[1] + self._cat_features.shape[1]

    @property
    def payload(self) -> dict:
        """The batch's feature tensors, as a plain dict."""
        return {"cat_features": self.cat_features, "num_features": self.num_features}

    def __getitem__(self, mask: torch.Tensor) -> "TabularBatch":
        """Slice the batch using a boolean mask or index tensor.

        Args:
            mask (torch.Tensor): Boolean mask or index tensor for slicing.
                Should be compatible with the batch size dimension.

        Returns:
            TabularBatch: A new TabularBatch instance containing only the
                selected elements.
        """
        new_cat_features = (
            self.cat_features[mask] if self.cat_features is not None else None
        )
        new_num_features = (
            self.num_features[mask] if self.num_features is not None else None
        )
        new_hidden_states = (
            {
                hidden_name: hidden_state[mask]
                for hidden_name, hidden_state in self.hidden_states.items()
            }
            if self.hidden_states is not None
            else None
        )

        return TabularBatch(
            cat_features=new_cat_features,
            num_features=new_num_features,
            hidden_states=new_hidden_states,
        )


class UpliftTabularBatch(TabularBatch):
    """A batch of tabular data for uplift modeling tasks.

    Extends TabularBatch with treatment indicators and optional channel types
    for causal inference and uplift modeling scenarios.

    Args:
        cat_features (torch.LongTensor): Batch of categorical features with
            shape [batch_size, n_cat_features].
        num_features (torch.FloatTensor): Batch of numerical features with
            shape [batch_size, n_num_features].
        is_treatment (torch.LongTensor): Binary treatment indicators with
            shape [batch_size]. 1 = treatment group, 0 = control group.
        channel_type (torch.LongTensor, optional): Channel/campaign type
        indicators with shape [batch_size]. Defaults to None.
        hidden_states (torch.FloatTensor, optional): Hidden state tensor for
            models that require it. Defaults to None.

    Attributes:
        is_treatment (torch.LongTensor): Treatment group indicators.
        channel_type (torch.LongTensor): Optional channel type indicators.
        Inherits all attributes from TabularBatch.
    """

    def __init__(
        self,
        cat_features: torch.LongTensor,
        num_features: torch.FloatTensor,
        is_treatment: torch.LongTensor,
        channel_type: torch.LongTensor = None,
        hidden_states: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__(
            cat_features=cat_features,
            num_features=num_features,
            hidden_states=hidden_states,
        )
        self._is_treatment = is_treatment
        self._channel_type = channel_type

    @property
    def is_treatment(self) -> torch.LongTensor:
        """Binary treatment indicator: 1 is treated, 0 is control."""
        return self._is_treatment

    @property
    def channel_type(self) -> torch.LongTensor:
        """Optional channel id, for multi-campaign runs."""
        return self._channel_type

    def to(self, device: torch.device) -> None:
        device_tabular_batch = super().to(device)

        return UpliftTabularBatch(
            cat_features=device_tabular_batch.cat_features,
            num_features=device_tabular_batch.num_features,
            hidden_states=device_tabular_batch.hidden_states,
            is_treatment=move_to_device(self.is_treatment, device),
            channel_type=move_to_device(self.channel_type, device),
        )
