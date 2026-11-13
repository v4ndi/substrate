"""``EventSequenceBatch``: padded event sequences and their masks.

Field-by-field walkthrough in ``docs/data/event_sequence_batch.md``.
"""

import torch
import torch.nn.functional as F

from avatar.data.base.batch import move_to_device


class EventSequenceBatch:
    """A batch of event sequences with associated metadata and utility methods.

    This class represents a batch of temporal event sequences, where each sequence
    consists of multiple events with timestamps. Provides methods for manipulating
    and analyzing event sequences, including time delta calculations, event filtering,
    and device management.

    Args:
        events: Dictionary mapping event names to their tensor representations.
            Each tensor should have shape (batch_size, seq_len, ...).
        timestamps: Optional tensor of event timestamps with shape
            (batch_size, seq_len). If None, assumes events are not timestamped.
        attention_mask: Optional mask tensor indicating valid events (1) vs padding (0)
            with shape (batch_size, seq_len). If None, assumes all events are valid.
        event_ids: Optional tensor of event type identifiers with shape
            (batch_size, seq_len). If None, assumes single event type.

    Attributes:
        _events (dict[str, torch.Tensor]): Dictionary mapping event types to their
            tensor representations. Each tensor has shape (batch_size, seq_len, ...).
        _timestamps (torch.FloatTensor): Tensor of event timestamps with shape
            (batch_size, seq_len).
        _attention_mask (torch.LongTensor): Attention mask indicating valid events
            with shape (batch_size, seq_len).
        _event_ids (torch.LongTensor): Tensor of event type identifiers with shape
            (batch_size, seq_len).

    Example:
        >>> events = {"click": torch.rand(2, 10, 5), "purchase": torch.rand(2, 10, 3)}
        >>> timestamps = torch.arange(20).float().view(2, 10)
        >>> mask = torch.ones(2, 10).long()
        >>> batch = EventSequenceBatch(events, timestamps, mask)
        >>> batch.seq_len  # Returns lengths of each sequence
        tensor([10, 10])
        >>> batch.get_timedeltas()  # Computes inter-event durations
    """

    def __init__(
        self,
        events: dict[str, torch.Tensor],
        timestamps: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        event_ids: torch.LongTensor | None = None,
        targets: torch.LongTensor | None = None,
    ):
        """Initializes the EventSequenceBatch.

        Raises:
            ValueError: If input tensors have inconsistent batch or sequence dimensions.
        """
        self._events = events
        self._timestamps = timestamps
        self._attention_mask = attention_mask
        self._event_ids = event_ids
        self._targets = targets

    def __getitem__(self, key: str) -> torch.Tensor:
        """Accesses event tensors by name.

        Args:
            key: Name of the event type to retrieve.

        Returns:
            The tensor corresponding to the requested event type.

        Raises:
            KeyError: If the requested event type is not found.
        """
        return self._events[key]

    def __setitem__(self, key: str, value: torch.Tensor):
        """Set a tensor attribute in the batch by name.

        Args:
            key: Name of the event type to set.s
            value: Tensor to set for the specified event type.

        Raises:
            KeyError: If the event type is not found in the batch.
        """
        if key in self._events:
            self._events[key] = value
        else:
            raise KeyError(f"Event type '{key}' not found in the batch.")

    @property
    def timestamps(self) -> torch.FloatTensor | None:
        """Returns the timestamp tensor if available."""
        return self._timestamps

    @property
    def attention_mask(self) -> torch.LongTensor | None:
        """Returns the attention mask tensor if available."""
        return self._attention_mask

    @property
    def seq_len(self) -> torch.LongTensor:
        """Computes the length of each sequence in the batch.

        Returns:
            Tensor of sequence lengths with shape (batch_size,).
        """
        return self.attention_mask.sum(dim=1)

    @property
    def event_ids(self) -> torch.LongTensor | None:
        """Returns the event type identifiers if available."""
        return self._event_ids

    @property
    def targets(self) -> torch.LongTensor | None:
        """Returns targets for each sequence in the batch."""
        return self._targets

    @property
    def sequence_columns(self) -> list[str]:
        """Lists all available event types in the batch.

        Returns:
            List of event type names.
        """
        return list(self._events.keys())

    def get_timedeltas(self, step: int = 1) -> torch.FloatTensor:
        """Computes time differences between events.

        Args:
            step: Number of steps to compute deltas over (default=1).

        Returns:
            Tensor of time deltas with same shape as timestamps, padded with zeros.
            Invalid positions (mask=0) are zeroed out.

        Raises:
            ValueError: If timestamps are not available.
        """
        time_deltas = self._timestamps[:, step:] - self._timestamps[:, :-step]
        time_deltas = F.pad(time_deltas, (step, 0), value=0.0)
        time_deltas[~self.attention_mask.bool()] = 0.0
        return time_deltas

    def event_attn_mask(
        self, event_id: int | list[int] | None = None
    ) -> torch.LongTensor:
        """Creates attention mask filtered by event type(s).

        Args:
            event_id: Single event type or list of types to include in mask.

        Returns:
            Attention mask tensor where only specified event types are marked valid.
            If event_id is None, returns the original attention mask.

        Raises:
            ValueError: If event_ids are not available when filtering is requested.
        """
        if event_id is not None:
            if isinstance(event_id, int):
                event_id = [event_id]

            event_id = torch.LongTensor(event_id).to(self.device)
            mask = (self.event_ids.unsqueeze(-1) == event_id).any(-1)

            return mask.long()
        else:
            return self._attention_mask

    def num_items(self, event_id: int | None = None) -> int:
        """Counts occurrences of specific event type(s).

        Args:
            event_id: Event type to count. If None, counts all valid events.

        Returns:
            Total count of specified events in the batch.
        """
        return self.event_attn_mask(event_id).sum()

    def to(self, device: torch.device):
        """Moves all tensors to the specified device.

        Args:
            device (torch.device): Target device (e.g., 'cpu' or 'cuda:0').

        Returns:
            New EventSequenceBatch instance with all tensors on the target device.
        """
        return EventSequenceBatch(
            events={key: val.to(device) for key, val in self._events.items()},
            timestamps=move_to_device(self._timestamps, device),
            attention_mask=move_to_device(self._attention_mask, device),
            event_ids=move_to_device(self._event_ids, device),
            targets=move_to_device(self._targets, device),
        )

    @property
    def device(self) -> torch.device:
        """Returns the device where the event tensors are stored."""
        return self._events[self.sequence_columns[0]].device
