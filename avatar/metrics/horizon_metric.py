import os
import pickle
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hotpp.data import PaddedBatch
from hotpp.metrics import TMAPMetric
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .base import BaseMetric


class HorizonMetric(BaseMetric):
    def __init__(
        self,
        compute_thresholds: Optional[bool] = False,
        map_deltas: Optional[List[float]] = None,
        map_target_length: Optional[int] = None,
        path_to_save_thresholds: Optional[str] = None,
        linspace_thresholds=10,
        empty_token=None,
        compute_per_class=False,
        me_models_path=None,
        num_processes_calib=8,
    ):
        if map_deltas is not None:
            if map_target_length is None:
                raise ValueError(
                    "Need the max target sequence length for mAP computation"
                )
            self.map_target_length = map_target_length
            self.tmap = TMAPMetric(time_delta_thresholds=map_deltas)
        else:
            self.map_target_length = None
            self.tmap = None

        self.me_models_path = me_models_path
        self.compute_thresholds = compute_thresholds
        self.optimal_thresholds = None
        self.linspace_thresholds = linspace_thresholds
        self.path_to_save_thresholds = path_to_save_thresholds
        self.empty_token = empty_token
        self.compute_per_class = compute_per_class
        self.num_processes_calib = num_processes_calib
        self.reset()

    def compute_optimal_thresholds(
        self,
        val_logits: List[torch.Tensor],
        val_targets: List[torch.Tensor],
        predicted_labels: List[torch.Tensor],
    ):
        """Compute optimal thresholds for classification."""
        val_logits = torch.cat(val_logits).squeeze(1, 3)
        val_targets = pad_second_dim(val_targets)
        predicted_labels = torch.cat(predicted_labels).squeeze(1)
        val_logits = torch.exp(val_logits)

        k = val_logits.shape[1]
        thresholds = torch.zeros(k, device=val_logits.device)

        for i in tqdm(range(k), desc="Computing optimal thresholds"):
            best_thresh = 0.5
            best_f1 = 0
            f_micro_unordered = 0

            logits_i = val_logits[:, i]
            min_val, max_val = logits_i.min().item(), logits_i.max().item()

            for thresh in np.linspace(min_val, max_val, self.linspace_thresholds):
                thresholds[i] = thresh
                preds = predicted_labels * (val_logits > thresholds)
                dict_for_threshold = compute_f_score(
                    preds, val_targets, num_processes=self.num_processes_calib
                )
                f1 = np.mean(dict_for_threshold["f_score"])
                tps, fps, fns = (
                    dict_for_threshold["tp"],
                    dict_for_threshold["fp"],
                    dict_for_threshold["fn"],
                )

                if f1 > best_f1:
                    best_f1 = f1
                    print(best_f1)
                    recall = tps / (tps + fns + 1e-8)
                    precision = tps / (tps + fps + 1e-8)
                    f_micro_unordered = (
                        2 * (precision * recall) / (precision + recall + 1e-8)
                    )
                    best_thresh = thresh

            thresholds[i] = best_thresh

        self.optimal_thresholds = thresholds
        return best_f1, f_micro_unordered

    def update(self, inputs, outputs) -> None:
        # losses = outputs.losses

        labels = inputs["targets_features"]._events["targets_events"][:, 1:]
        timestamps = inputs["targets_features"]._events["timestamps"][:, 1:]
        indices = outputs.indices
        seq_predicted_timestamps = outputs.sequences.payload["timestamps"]
        seq_predicted_labels = outputs.sequences.payload["targets_events"]
        seq_predicted_labels_logits = outputs.sequences.payload["targets_events_logits"]
        seq_predicted_weights = outputs.sequences.payload.get("_weights", None)
        seq_predicted_presence_logits = outputs.sequences.payload["_presence_logits"]

        self._additional_metrics = outputs.additional_metrics
        self._additional_losses = outputs.losses

        mask = labels != 0
        seq_lens = mask.sum(dim=1)

        features = PaddedBatch({"timestamps": timestamps, "labels": labels}, seq_lens)
        predictions = {
            "timestamps": seq_predicted_timestamps.squeeze(1),
            "labels": seq_predicted_labels.squeeze(1),
            "labels_logits": seq_predicted_labels_logits.squeeze(1),
        }
        if seq_predicted_weights is not None:
            predictions["_weights"] = seq_predicted_weights.squeeze(1)
        predictions = PaddedBatch(predictions, indices.seq_lens)
        targets = features
        targets_mask = features.seq_len_mask
        predictions_mask = predictions.payload["_weights"] > 0

        self._target_lengths.append(targets_mask.sum(1).cpu().flatten())  # (V).
        self._predicted_lengths.append(predictions_mask.sum(1).cpu().flatten())  # (BI).

        # Update deltas stats.
        predicted_timestamps = predictions.payload["timestamps"]  # (V, N).
        predicted_labels = predictions.payload["labels"]  # (V, N).
        if (len(predicted_timestamps) > 0) and (predicted_timestamps.shape[1] >= 2):
            deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]
            self._horizon_predicted_deltas_sums.append(
                deltas.float().mean().cpu() * deltas.numel()
            )
            self._horizon_n_predicted_deltas += deltas.numel()

        # Update entropies.
        not_event = predicted_labels.max().item() + 1
        predicted_labels_masked = predictions.payload["labels"].masked_fill(
            ~predictions_mask, not_event
        )  # .flatten(1, 2)  # (B, IN).
        assert predicted_labels_masked.ndim == 2
        counts = batch_bincount(predicted_labels_masked, not_event + 1)[
            :, :-1
        ]  # (B, C).
        probs = counts / counts.sum(dim=1, keepdim=True).clip(min=1)
        entropies = -(probs * probs.clip(min=1e-6).log()).sum(1)  # (B).
        self._sequence_labels_entropies.append(entropies.cpu())

        horizon_predictions_mask = torch.ones(
            predictions_mask.shape, device=predictions_mask.device
        ).bool()

        # Update F_score

        predictions_f = predictions.payload["labels"] * predictions_mask
        labels_f = targets.payload["labels"]

        if self.compute_thresholds or self.me_models_path:
            self._all_logits.append(seq_predicted_presence_logits)
            self._all_predicted_labels.append(seq_predicted_labels)
            self._all_labels_logits.append(seq_predicted_labels_logits)
            self._all_targets.append(labels_f.clone())

        self.update_f_score(predictions_f, labels_f)
        if self.tmap is not None:
            self.tmap.update(
                target_mask=targets_mask[:, : self.map_target_length],  # (V, K).
                target_times=targets.payload["timestamps"][
                    :, : self.map_target_length
                ],  # (V, K).
                target_labels=targets.payload["labels"][
                    :, : self.map_target_length
                ],  # (V, K).
                predicted_mask=horizon_predictions_mask,  # (V, N).
                predicted_times=predicted_timestamps,  # (V, N).
                predicted_labels_scores=predictions.payload[
                    "labels_logits"
                ],  # (V, N, C).
            )

    def compute(self) -> Dict[str, float]:
        """Compute and return all metrics."""
        metrics = {}

        # Compute optimal thresholds and F-scores
        if self.compute_thresholds:
            f1_macro, f1_micro = self.compute_optimal_thresholds(
                self._all_logits, self._all_targets, self._all_predicted_labels
            )

        if self.me_models_path:
            self.prepare_me_calib()

        # Add basic metrics if available
        if self._target_lengths:
            target_lengths = torch.cat(self._target_lengths)
            predicted_lengths = torch.cat(self._predicted_lengths)
            sequence_labels_entropies = torch.cat(self._sequence_labels_entropies)

            metrics.update({
                "mean-target-length": target_lengths.sum().item()
                / target_lengths.numel(),
                "mean-predicted-length": predicted_lengths.sum().item()
                / predicted_lengths.numel(),
                "horizon-mean-time-step": torch.stack(
                    self._horizon_predicted_deltas_sums
                )
                .sum()
                .item()
                / self._horizon_n_predicted_deltas,
                "sequence-labels-entropy": sequence_labels_entropies.mean().item(),
                **self._additional_losses,
                **self._additional_metrics,
                "empty_sequences": self._number_of_empty_sequences
                / (self._number_of_empty_sequences + len(self._f_scores) + 1e-8),
            })

        # Add F-scores if available

        for key, _ in self._per_class_tp.items():
            per_class_precision = self._per_class_tp[key] / (
                self._per_class_tp[key] + self._per_class_fp[key] + 1e-8
            )
            per_class_recall = self._per_class_tp[key] / (
                self._per_class_tp[key] + self._per_class_fn[key] + 1e-8
            )
            per_class_f_score = (
                2
                * (per_class_precision * per_class_recall)
                / (per_class_precision + per_class_recall + 1e-8)
            )
            metrics[f"recall-{key}"] = per_class_recall
            metrics[f"precision-{key}"] = per_class_precision
            metrics[f"f-score-{key}"] = per_class_f_score

        if self.compute_thresholds:
            if self._f_scores:
                metrics["f_macro_unordered"] = f1_macro
            if self._tps:
                metrics["f_micro_unordered"] = f1_micro
        else:
            metrics.update({"f_macro_unordered": np.mean(self._f_scores)})
            recall = np.sum(self._tps) / (np.sum(self._tps) + np.sum(self._fns) + 1e-8)
            precision = np.sum(self._tps) / (
                np.sum(self._tps) + np.sum(self._fps) + 1e-8
            )
            f_micro_unordered = 2 * (precision * recall) / (precision + recall)
            metrics.update({"f_micro_unordered": f_micro_unordered})

        # Add TMAP metrics if available
        if self.tmap is not None:
            metrics.update(self.tmap.compute())
        if self.compute_thresholds:
            with open(self.path_to_save_thresholds, "wb") as f:
                pickle.dump(self.optimal_thresholds, f)
        return metrics

    def reset(self) -> None:
        """Reset all metric accumulators."""
        self._f_scores = []
        self._f_scores_old = []
        self._tps = []
        self._fps = []
        self._fns = []
        self._target_lengths = []
        self._predicted_lengths = []
        self._horizon_predicted_deltas_sums = []
        self._horizon_n_predicted_deltas = 0
        self._sequence_labels_entropies = []
        self._all_logits = []
        self._all_labels_logits = []
        self._all_targets = []
        self._all_predicted_labels = []
        self._per_class_tp = defaultdict(int)
        self._per_class_fp = defaultdict(int)
        self._per_class_fn = defaultdict(int)
        self._number_of_empty_sequences = 0

    def update_f_score(
        self, predictions_f: torch.Tensor, labels_f: torch.Tensor
    ) -> None:
        """Update F-score metrics for the current batch."""
        max_len = max(predictions_f.size(1), labels_f.size(1))
        predictions_f = F.pad(predictions_f, (0, max_len - predictions_f.size(1)))
        labels_f = F.pad(labels_f, (0, max_len - labels_f.size(1)))

        batch_size, seq_len = predictions_f.shape
        preds_flat = predictions_f.view(-1)
        labels_flat = labels_f.view(-1)

        mask_pred = preds_flat != 0
        mask_label = labels_flat != 0

        tps, fps, fns = [], [], []
        is_empty = True

        for i in range(batch_size):
            pred = preds_flat[i * seq_len : (i + 1) * seq_len][
                mask_pred[i * seq_len : (i + 1) * seq_len]
            ]
            label = labels_flat[i * seq_len : (i + 1) * seq_len][
                mask_label[i * seq_len : (i + 1) * seq_len]
            ]

            unique_pred, counts_pred = torch.unique(pred, return_counts=True)
            unique_label, counts_label = torch.unique(label, return_counts=True)

            pred_dict = dict(zip(unique_pred.tolist(), counts_pred.tolist()))
            label_dict = dict(zip(unique_label.tolist(), counts_label.tolist()))
            classes_in_seq = set(pred_dict.keys()) | set(label_dict.keys())

            if not self.empty_token or unique_label[0].item() != self.empty_token:
                is_empty = False
                tp = sum(
                    min(pred_dict.get(k, 0), label_dict.get(k, 0))
                    for k in set(pred_dict) & set(label_dict)
                )
                fp = sum(pred_dict.values()) - tp
                fn = sum(label_dict.values()) - tp

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f_score = 2 * precision * recall / (precision + recall + 1e-8)

            if not is_empty:
                self._f_scores.append(f_score)
                tps.append(tp)
                fps.append(fp)
                fns.append(fn)
                if self.compute_per_class:
                    for c in classes_in_seq:
                        tp_c = min(pred_dict.get(c, 0), label_dict.get(c, 0))
                        fp_c = pred_dict.get(c, 0) - tp_c
                        fn_c = label_dict.get(c, 0) - tp_c

                        # Accumulate counts in per-class dictionaries
                        self._per_class_tp[c] += tp_c
                        self._per_class_fp[c] += fp_c
                        self._per_class_fn[c] += fn_c
            else:
                self._number_of_empty_sequences += 1
            is_empty = True

        # Convert to tensors and compute batch metrics
        tps_t = torch.tensor(tps, device=predictions_f.device)
        fps_t = torch.tensor(fps, device=predictions_f.device)
        fns_t = torch.tensor(fns, device=predictions_f.device)

        precision = tps_t / (tps_t + fps_t + 1e-8)
        recall = tps_t / (tps_t + fns_t + 1e-8)
        f_scores = 2 * precision * recall / (precision + recall + 1e-8)

        self._tps.extend(tps)
        self._fps.extend(fps)
        self._fns.extend(fns)
        self._f_scores_old.extend(f_scores.tolist())

    def prepare_me_calib(self):
        L = torch.cat(self._all_labels_logits)
        P = torch.cat(self._all_logits)

        P_prob = torch.exp(P)
        R = torch.nn.functional.softmax(L, dim=-1).squeeze(1)
        P_prob = P_prob.squeeze(2)
        R = R.squeeze(2)
        S = (R * P_prob.squeeze(1)).sum(dim=1)
        max_target_length = 0

        for batch in self._all_targets:
            max_target_length = max(max_target_length, batch.shape[1])
        targets = []
        for i in range(len(self._all_targets)):
            targets.append(
                F.pad(
                    self._all_targets[i],
                    (0, max_target_length - self._all_targets[i].shape[1]),
                )
            )
        targets = torch.cat(targets)
        count_matrix = torch.zeros((targets.shape[0], L.shape[-1]))
        for i in range(targets.shape[0]):
            filtered = targets[i][targets[i] != 0]
            counts = torch.bincount(filtered)
            count_matrix[i, : counts.shape[0]] = counts
        models = [
            LinearRegression(fit_intercept=True) for i in range(count_matrix.shape[1])
        ]
        for i in range(count_matrix.shape[1]):
            models[i].fit(
                S[:, i].cpu().numpy().reshape(-1, 1),
                count_matrix[:, i].cpu().numpy().reshape(-1, 1),
            )
        with open(self.me_models_path, "wb") as f:
            pickle.dump(models, f)
        return models


class HorizonInference(BaseMetric):
    def __init__(
        self,
        save_steps: int,
        decoding_dict: str,
        path_to_save: str,
        target_date: str,
        prefix: Optional[str] = None,
        path_to_thresholds: Optional[str] = None,
        path_to_me_models: Optional[str] = None,
    ):
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        self.path_to_me_models = path_to_me_models
        if self.path_to_me_models:
            with open(path_to_me_models, "rb") as f:
                self.me_models = pickle.load(f)
            self._coefs = torch.tensor([
                model.coef_.flatten()[0] for model in self.me_models
            ])
            self._intercepts = torch.tensor([
                model.intercept_[0] for model in self.me_models
            ])
        with open(decoding_dict, "rb") as f:
            self.decoding_dict = pickle.load(f)

        os.makedirs(self.path_to_save, exist_ok=True)
        self.prefix = prefix
        self.path_to_thresholds = path_to_thresholds
        if self.path_to_thresholds:
            with open(self.path_to_thresholds, "rb") as f:
                self.thresholds = pickle.load(f)
        self.target_date = datetime.strptime(target_date, "%d-%m-%Y")
        self.reset()

    def update(self, inputs: Dict, outputs: Dict) -> None:
        """Collect predictions for saving."""
        epk_id = inputs["epk_id"]
        seq_predicted = outputs.sequences.payload

        timestamps = seq_predicted["timestamps"].squeeze(1)
        labels = seq_predicted["targets_events"].squeeze(1)
        presence_logits = seq_predicted["_presence_logits"].squeeze(1)
        weights = seq_predicted.get("_weights", None).squeeze(1)
        presence_probs = torch.exp(presence_logits.squeeze(2))
        labels_logits = seq_predicted["targets_events_logits"]

        if self.path_to_thresholds:
            predictions_mask = presence_probs > self.thresholds
        else:
            predictions_mask = weights > 0
        predictions_f = labels * predictions_mask
        timestamps_f = timestamps * predictions_mask
        presence_f = torch.exp(presence_logits.squeeze(2)) * predictions_mask

        if self.path_to_me_models:
            P_prob = presence_probs
            R = torch.nn.functional.softmax(labels_logits, dim=-1).squeeze(1)
            R = R.squeeze(2)
            S = (R * P_prob.unsqueeze(-1)).sum(dim=1)
            preds_me = S * self._coefs.unsqueeze(0).to(
                S.device
            ) + self._intercepts.unsqueeze(0).to(S.device)

        for i in range(predictions_f.size(0)):
            dict_for_user = {"target_name": "vnv_predict", "epk_id": epk_id[i]}
            value = []

            valid_indices = predictions_f[i] != 0
            valid_labels = predictions_f[i][valid_indices]
            valid_timestamps = timestamps_f[i][valid_indices]
            valid_presences = presence_f[i][valid_indices]

            for label, timestamp, presence in zip(
                valid_labels, valid_timestamps, valid_presences
            ):
                vnv = self.decoding_dict[label.item()].split("_")
                date = datetime.fromtimestamp(int(timestamp * 86400))
                if (
                    date.month != self.target_date.month
                    or date.year != self.target_date.year
                ):
                    date = self.target_date

                value.append({
                    "evt_attr_2": vnv[1],
                    "evt_attr_3": vnv[2],
                    "txn_datetime": [date],
                    "presence": presence.item(),
                })

            dict_for_user["value"] = value
            self._preds.append(dict_for_user)

            if self.path_to_me_models:
                dict_for_user_me = {"target_name": "vnv_predict", "epk_id": epk_id[i]}
                value = []
                for j in range(2, preds_me.shape[1]):
                    vnv = self.decoding_dict[j].split("_")
                    dict_for_one_vnv = {
                        "evt_attr_2": vnv[1],
                        "evt_attr_3": vnv[2],
                        "txn_datetime": [self.target_date],
                        "num": preds_me[
                            i, j
                        ].item(),  # Используем предвычисленные значения
                    }
                    value.append(dict_for_one_vnv)
                dict_for_user_me["value"] = value
                self._preds_me.append(dict_for_user_me)
        if len(self._preds) >= self.save_steps:
            self.compute()

    def compute(self) -> Dict[str, float]:
        """Save predictions to parquet file."""
        submit = pd.DataFrame(self._preds)
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = (
            f"{time_now}_{self.prefix}.parquet"
            if self.prefix
            else f"{time_now}.parquet"
        )

        cur_path = os.path.join(self.path_to_save, filename)
        submit.to_parquet(cur_path, index=False)

        if self.path_to_me_models:
            filename_me = (
                f"{time_now}_{self.prefix}_me.parquet"
                if self.prefix
                else f"{time_now}_me.parquet"
            )
            submit_me = pd.DataFrame(self._preds_me)
            cur_path_me = os.path.join(self.path_to_save, filename_me)
            submit_me.to_parquet(cur_path_me, index=False)

        self.reset()
        print(f"predict_saved: {cur_path}")
        return {}

    def reset(self) -> None:
        """Reset predictions accumulator."""
        self._preds = []
        self._preds_me = []


class KeyEventsInference(BaseMetric):
    def __init__(
        self,
        save_steps: int,
        decoding_dict: str,
        path_to_save: str,
        lower_target_date: str,
        upper_target_date: str,
        prefix: Optional[str] = None,
        path_to_thresholds: Optional[str] = None,
        empty_token=None,
    ):
        self.path_to_save = path_to_save
        self.save_steps = save_steps
        with open(decoding_dict, "rb") as f:
            self.decoding_dict = pickle.load(f)

        os.makedirs(self.path_to_save, exist_ok=True)
        self.prefix = prefix
        self.path_to_thresholds = path_to_thresholds
        if self.path_to_thresholds:
            with open(self.path_to_thresholds, "rb") as f:
                self.thresholds = pickle.load(f)
        self.lower_target_date = datetime.strptime(lower_target_date, "%d-%m-%Y")
        self.upper_target_date = datetime.strptime(upper_target_date, "%d-%m-%Y")
        self.empty_token = empty_token
        self.reset()

    def update(self, inputs: Dict, outputs: Dict) -> None:
        """Collect predictions for saving."""
        epk_id = inputs["epk_id"]
        seq_predicted = outputs.sequences.payload

        timestamps = seq_predicted["timestamps"].squeeze(1)
        labels = seq_predicted["targets_events"].squeeze(1)
        presence_logits = seq_predicted["_presence_logits"].squeeze(1)
        weights = seq_predicted.get("_weights", None).squeeze(1)
        presence_probs = torch.exp(presence_logits.squeeze(2))

        if self.path_to_thresholds:
            predictions_mask = presence_probs > self.thresholds
        else:
            predictions_mask = weights > 0
        predictions_f = labels * predictions_mask
        timestamps_f = timestamps * predictions_mask

        for i in range(predictions_f.size(0)):
            dict_for_user = {"target_name": "key_event", "epk_id": epk_id[i]}
            value = []

            valid_indices = predictions_f[i] != 0
            valid_labels = predictions_f[i][valid_indices]
            valid_timestamps = timestamps_f[i][valid_indices]

            for label, timestamp in zip(valid_labels, valid_timestamps):
                if label.item() != self.empty_token:
                    key_event = self.decoding_dict[label.item()]
                    date = datetime.fromtimestamp(int(timestamp * 2630016))
                    if (
                        self.lower_target_date <= date
                        and date <= self.upper_target_date
                    ):
                        date = self.lower_target_date

                        value.append({
                            "day_part": [date],
                            "target_id": key_event,
                        })
                    else:
                        date = self.lower_target_date

                        value.append({
                            "day_part": [date],
                            "target_id": key_event,
                        })

            dict_for_user["value"] = value
            self._preds.append(dict_for_user)

        if len(self._preds) >= self.save_steps:
            self.compute()

    def compute(self) -> Dict[str, float]:
        """Save predictions to parquet file."""
        submit = pd.DataFrame(self._preds)
        time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = (
            f"{time_now}_{self.prefix}.parquet"
            if self.prefix
            else f"{time_now}.parquet"
        )
        cur_path = os.path.join(self.path_to_save, filename)
        submit.to_parquet(cur_path, index=False)
        self.reset()
        print(f"predict_saved: {cur_path}")
        return {}

    def reset(self) -> None:
        """Reset predictions accumulator."""
        self._preds = []


@staticmethod
def batch_bincount(x: torch.Tensor, minlength: int = 0) -> torch.Tensor:
    """Count occurrences of each value in a batch."""
    # x: (B, N)
    # returns: (B, C)
    b, _ = x.shape
    c = max(x.max().item() + 1, minlength)
    counts = x.new_zeros(b, c)
    counts.scatter_add_(dim=1, index=x.long(), src=torch.ones_like(x))
    return counts


def _process_batch_element(args):
    """Helper function for parallel processing with empty token handling."""
    i, pred, label, seq_len, empty_token = args
    batch_slice = slice(i * seq_len, (i + 1) * seq_len)
    pred_i = pred[batch_slice]
    label_i = label[batch_slice]

    # Get non-zero elements
    pred_mask = pred_i != 0
    label_mask = label_i != 0
    pred_i = pred_i[pred_mask]
    label_i = label_i[label_mask]

    # Case 1: Empty prediction (all 2s) but non-empty target
    if len(pred_i) == 0 or (len(pred_i) == 1 and pred_i[0] == empty_token):
        if len(label_i) > 0 and (len(label_i) > 1 or label_i[0] != empty_token):
            return (0, 0, len(label_i))  # (tp, fp, fn)
        return (0, 0, 0)

    # Case 2: Non-empty prediction but empty target (all 2s)
    if len(label_i) == 0 or (len(label_i) == 1 and label_i[0] == empty_token):
        if len(pred_i) > 0 and (len(pred_i) > 1 or pred_i[0] != empty_token):
            return (0, len(pred_i), 0)  # (tp, fp, fn)
        return (0, 0, 0)

    # Case 3: Both prediction and target are non-empty - normal calculation
    # Find maximum class value to determine array size
    max_class = (
        max(
            np.max(pred_i) if len(pred_i) > 0 else 0,
            np.max(label_i) if len(label_i) > 0 else 0,
            empty_token,
        )
        + 1
    )

    pred_counts = np.bincount(pred_i, minlength=max_class)
    label_counts = np.bincount(label_i, minlength=max_class)

    # Ignore the empty token for normal TP/FP/FN calculation
    pred_counts = np.concatenate([
        pred_counts[:empty_token],
        pred_counts[empty_token + 1 :],
    ])
    label_counts = np.concatenate([
        label_counts[:empty_token],
        label_counts[empty_token + 1 :],
    ])

    # Ensure both arrays have the same length
    max_len = max(len(pred_counts), len(label_counts))
    pred_counts = np.pad(pred_counts, (0, max_len - len(pred_counts)))
    label_counts = np.pad(label_counts, (0, max_len - len(label_counts)))

    tp = np.minimum(pred_counts, label_counts).sum()
    fp = pred_counts.sum() - tp
    fn = label_counts.sum() - tp

    return (tp, fp, fn)


@staticmethod
def compute_f_score(
    predictions_f: torch.Tensor,
    labels_f: torch.Tensor,
    num_processes: int = 8,
    empty_token: int = 10000,
) -> Dict[str, Union[float, List[float]]]:
    """Compute F-score between predictions and labels using multiprocessing."""
    max_len = max(predictions_f.size(1), labels_f.size(1))
    predictions_f = F.pad(predictions_f, (0, max_len - predictions_f.size(1)))
    labels_f = F.pad(labels_f, (0, max_len - labels_f.size(1)))

    batch_size, seq_len = predictions_f.shape
    preds_flat = predictions_f.view(-1).cpu().numpy()
    labels_flat = labels_f.view(-1).cpu().numpy()

    # Prepare arguments for parallel processing
    args = [
        (i, preds_flat, labels_flat, seq_len, empty_token) for i in range(batch_size)
    ]

    # Use multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_batch_element, args)

    # Unpack results
    tps, fps, fns = zip(*results)
    tps = torch.tensor(tps, dtype=torch.float32)
    fps = torch.tensor(fps, dtype=torch.float32)
    fns = torch.tensor(fns, dtype=torch.float32)

    precision = tps / (tps + fps + 1e-8)
    recall = tps / (tps + fns + 1e-8)
    f_scores = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "tp": tps.sum().item(),
        "fp": fps.sum().item(),
        "fn": fns.sum().item(),
        "f_score": f_scores.tolist(),
    }


@staticmethod
def pad_second_dim(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad tensors to have the same second dimension."""
    max_len = max(t.size(1) for t in tensors)
    padded_tensors = []

    for t in tensors:
        pad = [0, max_len - t.size(1)]
        full_pad = [0, 0] * (t.dim() - 2) + pad
        padded = F.pad(t, full_pad, mode="constant", value=0)
        padded_tensors.append(padded)

    return torch.cat(padded_tensors)
