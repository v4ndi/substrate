from copy import deepcopy

import torch
import torch.nn as nn

from avatar.data.event_seq_batch import EventSequenceBatch
from avatar.nn.sequence import BaseSequenceModel
from avatar.nn.utils.agg import get_aggregation_layer
from avatar.outputs import SequenceOutput


class NextKTokensPrediction(nn.Module):
    def __init__(
        self,
        model: BaseSequenceModel,
        horizon: int = 1,
        horizion_loss_weight: float = 1,
        feature_loss_weights: dict[str, float] | None = None,
        aggregation_config: dict[str, any] | None = None,
        numeric_loss=None,
        categorical_loss=None,
        enable_event_id_prediction: bool = False,
    ):
        """
        Args:
            model: BaseSequenceModel - sequence model
            feature_loss_weights - dictionary of weights for each feature.
            If "timedelta" is not in the dictionary, this,
            it will have a carefully selected coefficient
            For other parameters, the default weight is 1.0.
            aggregation_config: kwargs for avatar.nn.utils.agg.get_aggregation_layer
            horizon: int - number of tokens to predict.
            horizon_loss_coef: float - coef for horizon loss. Negative values not recommended.
            If horizon_loss_coef is zero, then all losses are equally weighted.
            numeric_loss: torch.nn.Module.loss - the loss function that is calculated for numeric features.
            categorical_loss: torch.nn.Module.loss - the loss function that is calculated for categorical features.

            loss(horizon_i, horizon_loss_coef) = loss_i / horizon_i ** horizon_loss_coef
        """
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        assert horizion_loss_weight >= 0.0, "horizon_loss_coef should be >= 0.0"
        super().__init__()
        self.model = model
        self.model_columns_meta = deepcopy(
            self.model.feature_encoder.embedding.columns_meta
        )

        self.event_id_embedding = None
        if enable_event_id_prediction:

            def handle_meta(meta):
                event_id = meta["event_id"]
                if event_id is None:
                    return 0
                if isinstance(event_id, int):
                    return event_id
                return max(event_id)

            num_event_ids = max(map(handle_meta, self.model_columns_meta.values())) + 1
            self.num_event_ids = num_event_ids
            self.event_id_embedding = nn.Embedding(
                num_embeddings=num_event_ids + 1,  # +1 for padding fake class
                embedding_dim=model.feature_encoder.embedding.hidden_size,
            )

        self.lm_heads = nn.ModuleList([
            nn.ModuleDict({
                key: nn.Linear(
                    model.feature_encoder.embedding.hidden_size,
                    meta["n_classes"],
                )
                for key, meta in self.model_columns_meta.items()
            })
            for _ in range(horizon)
        ])
        if enable_event_id_prediction:
            for module_dict in self.lm_heads:
                module_dict["event_id"] = nn.Linear(
                    model.feature_encoder.embedding.hidden_size, num_event_ids
                )

        self.feature_loss_weights = {key: 1.0 for key in self.model_columns_meta.keys()}
        if feature_loss_weights is not None:
            assert all(
                key in self.model_columns_meta.keys()
                for key in feature_loss_weights.keys()
            )
            self.feature_loss_weights.update(feature_loss_weights)
        if "timedelta" not in self.feature_loss_weights:
            self.feature_loss_weights["timedelta"] = 0.6

        if self.feature_loss_weights["timedelta"] != 0.0:
            for horizon_module in self.lm_heads:
                horizon_module["timedelta"] = nn.Linear(
                    model.feature_encoder.embedding.hidden_size, 1
                )
        self.horizon = horizon
        self.horizion_loss_weight = horizion_loss_weight
        self.coefs = [(coef + 1) ** horizion_loss_weight for coef in range(horizon)]
        self.numeric_loss = (
            numeric_loss if numeric_loss is not None else nn.L1Loss(reduction="none")
        )
        self.categorical_loss = (
            categorical_loss
            if categorical_loss is not None
            else nn.CrossEntropyLoss(reduction="sum")
        )

        self.aggregation_layer = get_aggregation_layer(**aggregation_config)

    @staticmethod
    def create_labels(
        input_ids, logits, horizon_offset, n_classes: int, pad_token: int = 0
    ):
        """
        n_classes == 1 -> numeric, else categorical
        """
        with torch.no_grad():
            labels = input_ids.detach().clone()
            if n_classes > 1:
                pad_tokens_mask = labels == pad_token
                labels[pad_tokens_mask] = -100
        shifted_labels = labels[:, horizon_offset:].contiguous()  # shift labels
        shifted_logits = logits[:, :-horizon_offset, :].contiguous()  # shift logits

        return shifted_logits, shifted_labels

    def calculate_loss(
        self,
        shifted_logits,
        shifted_labels,
        n_classes,
        attention_mask=None,
        horizon_offset=None,
    ):
        """
        n_classes == 1 -> numeric, else categorical
        """
        assert shifted_logits.shape[1] == shifted_labels.shape[1]
        if n_classes == 1:
            shifted_logits = shifted_logits.squeeze(dim=-1)
            loss = self.numeric_loss(shifted_logits, shifted_labels)

            num_items = attention_mask[:, horizon_offset:]
            loss *= attention_mask[:, horizon_offset:]

            return loss.sum(), num_items.sum()
        elif n_classes > 1:
            loss = self.categorical_loss(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
            )
            num_items = (shifted_labels != -100).sum()

            return loss, num_items
        else:
            raise ValueError()

    def calculate_delta_loss(self, logits, labels, horizon_offset, attention_mask):
        shifted_logits, shifted_labels = self.create_labels(
            input_ids=labels.get_timedeltas(horizon_offset),
            logits=logits,
            horizon_offset=horizon_offset,
            n_classes=1,
        )
        loss, head_num_items = self.calculate_loss(
            shifted_logits=shifted_logits,
            shifted_labels=shifted_labels,
            n_classes=1,
            attention_mask=attention_mask,
            horizon_offset=horizon_offset,
        )
        return loss, head_num_items

    def _calc_event_id_loss(
        self,
        seq_features: EventSequenceBatch,
        last_hidden_state: torch.Tensor,
        horizon_offset: int,
        horizon_idx: int,
    ):
        event_id_logits = self.lm_heads[horizon_idx]["event_id"](last_hidden_state)
        event_ids = seq_features.event_ids
        shifted_event_logits, shifted_event_labels = self.create_labels(
            input_ids=event_ids,
            logits=event_id_logits,
            horizon_offset=horizon_offset,
            n_classes=self.num_event_ids,
            pad_token=-100,
        )
        event_id_loss, num_items = self.calculate_loss(
            shifted_logits=shifted_event_logits,
            shifted_labels=shifted_event_labels,
            n_classes=self.num_event_ids,
            attention_mask=None,
            horizon_offset=horizon_offset,
        )

        return event_id_loss, num_items

    def forward(self, seq_features: EventSequenceBatch, **kwargs):
        output = self.model(seq_features=seq_features)
        last_hidden_state = output.last_hidden_state

        losses = {}
        total_num_items = {}

        for horizon_idx in range(self.horizon):
            horizon_offset = horizon_idx + 1
            if horizon_offset >= last_hidden_state.shape[1]:
                break

            if self.event_id_embedding is not None:
                event_id_loss, head_num_items = self._calc_event_id_loss(
                    seq_features, last_hidden_state, horizon_offset, horizon_idx
                )
                loss_key = f"event_id_head_{horizon_idx}"
                total_num_items[loss_key] = head_num_items
                losses[loss_key] = event_id_loss

                event_id_embeds = self.event_id_embedding(
                    torch.where(
                        seq_features.event_ids != -100,
                        seq_features.event_ids,
                        self.num_event_ids,
                    )
                )
                last_hidden_state = last_hidden_state + event_id_embeds

            for key, meta in self.model_columns_meta.items():
                input_ids = seq_features[key]
                logits = self.lm_heads[horizon_idx][key](last_hidden_state)

                shifted_logits, shifted_labels = self.create_labels(
                    input_ids=input_ids,
                    logits=logits,
                    horizon_offset=horizon_offset,
                    n_classes=meta["n_classes"],
                )

                loss, head_num_items = self.calculate_loss(
                    shifted_logits=shifted_logits,
                    shifted_labels=shifted_labels,
                    n_classes=meta["n_classes"],
                    attention_mask=seq_features.event_attn_mask(meta["event_id"]),
                    horizon_offset=horizon_offset,
                )
                if self.training:
                    loss *= self.feature_loss_weights[key]
                    loss /= self.coefs[horizon_idx]

                loss_key = f"{key}_head_{horizon_idx}"
                total_num_items[loss_key] = head_num_items
                losses[loss_key] = loss

            if self.feature_loss_weights["timedelta"] != 0.0:
                if seq_features.timestamps is None:
                    raise ValueError(
                        "Timestamps are not in dataset or there not flag in config (dataset.read_columns)."
                    )
                logits = self.lm_heads[horizon_idx]["timedelta"](last_hidden_state)

                loss, head_num_items = self.calculate_delta_loss(
                    logits=logits,
                    labels=seq_features,
                    horizon_offset=horizon_offset,
                    attention_mask=seq_features.attention_mask,
                )
                loss_key = f"timedelta_head_{horizon_idx}"
                total_num_items[loss_key] = head_num_items
                if self.training:
                    loss *= self.feature_loss_weights["timedelta"]
                losses[loss_key] = loss / self.coefs[horizon_idx]

        aggregated_hidden_state = self.aggregation_layer(
            last_hidden_state, seq_features.attention_mask
        )
        return SequenceOutput(
            loss=None,
            losses=losses,
            num_items=total_num_items,
            aggregated_hidden_state=aggregated_hidden_state,
            router_logits=output.router_logits,
        )
