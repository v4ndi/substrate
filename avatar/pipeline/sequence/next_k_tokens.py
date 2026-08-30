from copy import deepcopy

import torch
import torch.nn as nn

from avatar.data.sequential.batch import EventSequenceBatch
from avatar.losses.next_k_tokens import HeadPrediction, NextKTokensLoss
from avatar.nn.sequential import BaseSequenceModel
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
        loss: NextKTokensLoss | None = None,
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
            loss: NextKTokensLoss - overrides the loss built from the arguments
            above. The prediction heads stay here because they carry
            parameters; only the label shifting, criteria and per-head
            weighting live in the loss module.

            loss(horizon_i, horizon_loss_coef) = loss_i / horizon_i ** horizon_loss_coef
        """
        if aggregation_config is None:
            aggregation_config = {"name": "mean"}
        assert horizion_loss_weight >= 0.0, "horizon_loss_coef should be >= 0.0"
        super().__init__()
        self.model = model
        self.model_columns_meta = deepcopy(
            self.model.event_encoder.embedding.columns_meta
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
                embedding_dim=model.event_encoder.embedding.hidden_size,
            )

        self.lm_heads = nn.ModuleList([
            nn.ModuleDict({
                key: nn.Linear(
                    model.event_encoder.embedding.hidden_size,
                    meta["n_classes"],
                )
                for key, meta in self.model_columns_meta.items()
            })
            for _ in range(horizon)
        ])
        if enable_event_id_prediction:
            for module_dict in self.lm_heads:
                module_dict["event_id"] = nn.Linear(
                    model.event_encoder.embedding.hidden_size, num_event_ids
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
                    model.event_encoder.embedding.hidden_size, 1
                )
        self.horizon = horizon
        self.horizion_loss_weight = horizion_loss_weight
        self.loss = (
            loss
            if loss is not None
            else NextKTokensLoss(
                horizon=horizon,
                feature_loss_weights=self.feature_loss_weights,
                horizion_loss_weight=horizion_loss_weight,
                numeric_loss=numeric_loss,
                categorical_loss=categorical_loss,
            )
        )

        self.aggregation_layer = get_aggregation_layer(**aggregation_config)

    def forward(self, seq_features: EventSequenceBatch, **kwargs):
        output = self.model(seq_features=seq_features)
        last_hidden_state = output.last_hidden_state

        # Collect what each head predicted, per horizon, and score it all at
        # once. The loop cannot be hoisted into the loss module: the event-id
        # embedding feeds back into the hidden state between horizons.
        horizons: list[dict[str, HeadPrediction]] = []

        for horizon_idx in range(self.horizon):
            horizon_offset = horizon_idx + 1
            if horizon_offset >= last_hidden_state.shape[1]:
                break

            heads: dict[str, HeadPrediction] = {}
            if self.event_id_embedding is not None:
                heads["event_id"] = HeadPrediction(
                    logits=self.lm_heads[horizon_idx]["event_id"](last_hidden_state),
                    input_ids=seq_features.event_ids,
                    n_classes=self.num_event_ids,
                    pad_token=-100,
                )
                event_id_embeds = self.event_id_embedding(
                    torch.where(
                        seq_features.event_ids != -100,
                        seq_features.event_ids,
                        self.num_event_ids,
                    )
                )
                last_hidden_state = last_hidden_state + event_id_embeds

            for key, meta in self.model_columns_meta.items():
                heads[key] = HeadPrediction(
                    logits=self.lm_heads[horizon_idx][key](last_hidden_state),
                    input_ids=seq_features[key],
                    n_classes=meta["n_classes"],
                    attention_mask=seq_features.event_attn_mask(meta["event_id"]),
                )

            if self.feature_loss_weights["timedelta"] != 0.0:
                if seq_features.timestamps is None:
                    raise ValueError(
                        "Timestamps are not in dataset or there not flag in config (dataset.read_columns)."
                    )
                heads["timedelta"] = HeadPrediction(
                    logits=self.lm_heads[horizon_idx]["timedelta"](last_hidden_state),
                    input_ids=seq_features.get_timedeltas(horizon_offset),
                    n_classes=1,
                    attention_mask=seq_features.attention_mask,
                    scale_by_coef_in_eval=True,
                )
            horizons.append(heads)

        loss_output = self.loss(horizons)

        aggregated_hidden_state = self.aggregation_layer(
            last_hidden_state, seq_features.attention_mask
        )
        return SequenceOutput(
            loss=loss_output.loss,
            losses=loss_output.components,
            num_items=loss_output.num_items,
            aggregated_hidden_state=aggregated_hidden_state,
            router_logits=output.router_logits,
        )
