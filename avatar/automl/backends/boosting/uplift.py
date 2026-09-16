"""Composite S-, T- and X-learner boosting backend."""

from __future__ import annotations

import json
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import polars as pl

from avatar.automl.backends.boosting.base import BaseBoostingBackend
from avatar.automl.backends.boosting.binary import BinaryBoostingBackend
from avatar.automl.backends.boosting.hyperopt import resolve_default_search_space, suggest_params
from avatar.automl.backends.boosting.interface import BoostingBackend
from avatar.automl.backends.boosting.regression import RegressionBoostingBackend
from avatar.automl.data import FeatureSchema
from avatar.automl.exceptions import ArtifactIntegrityError, MissingDependencyError, SchemaError
from avatar.automl.metrics.base import Metric, MetricInput
from avatar.automl.progress import log_progress

UPLIFT_SCORE_COLUMNS = (
    "score_s",
    "score_s_control",
    "score_s_treatment",
    "score_t",
    "score_t_control",
    "score_t_treatment",
    "score_x",
    "score_x_control",
    "score_x_treatment",
    "score_x_propensity",
)

_COMPONENT_KIND = {
    "s_outcome": "classifier",
    "t_control_outcome": "classifier",
    "t_treatment_outcome": "classifier",
    "x_control_outcome": "classifier",
    "x_treatment_outcome": "classifier",
    "x_control_effect": "regressor",
    "x_treatment_effect": "regressor",
    "x_propensity": "classifier",
}


@dataclass
class UpliftBoostingBackend(BoostingBackend):
    """Contain independently fitted S-, T-, and X-learner components.

    Args:
        engine: Native engine name, ``catboost`` or ``xgboost``.
        params: Parameters shared by learner components.
        random_state: Reproducibility seed.
        device: Runtime ``cpu`` or ``gpu`` device.
        verbose: Native training logging setting.
        estimate_propensity: Fit row-level treatment propensity when ``True``.
    """

    estimate_propensity: bool = False
    treatment_column: str = "treatment"
    components: dict[str, BaseBoostingBackend] = field(default_factory=dict)
    learner_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    learner_metrics: dict[str, float] = field(default_factory=dict)
    _prepared_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _reuse_prepared: bool = field(default=False, repr=False)

    def _component(self, kind: str, params: Mapping[str, Any]) -> BaseBoostingBackend:
        backend_class = BinaryBoostingBackend if kind == "classifier" else RegressionBoostingBackend
        return backend_class(
            engine=self.engine,
            params=dict(params),
            random_state=self.random_state,
            device=self.device,
            verbose=self.verbose,
        )

    def _s_schema(self, schema: FeatureSchema, frame: pl.DataFrame) -> FeatureSchema:
        if self.treatment_column in schema.feature_order:
            return schema
        return FeatureSchema(
            categorical=(*schema.categorical, self.treatment_column),
            numerical=schema.numerical,
            feature_order=(*schema.feature_order, self.treatment_column),
            dtypes=dict(schema.dtypes) | {self.treatment_column: str(frame.schema[self.treatment_column])},
            target_column=schema.target_column,
            client_id_column=schema.client_id_column,
            treatment_column=schema.treatment_column,
            group_column=schema.group_column,
        )

    @staticmethod
    def _validate_part(target: np.ndarray, treatment: np.ndarray, part_name: str) -> None:
        for arm in (0, 1):
            arm_target = np.unique(target[treatment == arm])
            if not len(arm_target):
                msg = f"{part_name}: training data is missing treatment arm {arm}"
                raise SchemaError(msg)
            missing = sorted({0, 1} - set(arm_target.tolist()))
            if missing:
                msg = (
                    f"{part_name}: treatment arm {arm} is missing target class(es) {missing}; "
                    "S/T/X learners require both outcomes in each arm"
                )
                raise SchemaError(msg)

    def _fit_component(
        self,
        kind: str,
        params: Mapping[str, Any],
        train: pl.DataFrame,
        target: np.ndarray,
        valid: pl.DataFrame,
        valid_target: np.ndarray,
        schema: FeatureSchema,
        *,
        cache_key: str | None = None,
    ) -> BaseBoostingBackend:
        component = self._component(kind, params)
        if self._reuse_prepared and cache_key is not None:
            prepared = self._prepared_cache.get(cache_key)
            if prepared is None:
                prepared = component.prepare_fit_data(
                    train,
                    target,
                    schema,
                    valid_frame=valid,
                    valid_target=valid_target,
                    prequantize=True,
                )
                self._prepared_cache[cache_key] = prepared
            component.fit_prepared(prepared)
        else:
            component.fit(train, target, schema, valid_frame=valid, valid_target=valid_target)
        return component

    def _predict_component(self, component: BaseBoostingBackend, frame: pl.DataFrame, schema: FeatureSchema) -> np.ndarray:
        values = np.asarray(component.predict_score(frame, schema), dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            msg = "A constituent uplift model returned NaN or infinite scores"
            raise SchemaError(msg)
        return values

    def _fit_s(
        self,
        params: Mapping[str, Any],
        train: pl.DataFrame,
        y_train: np.ndarray,
        valid: pl.DataFrame,
        y_valid: np.ndarray,
        schema: FeatureSchema,
    ) -> dict[str, BaseBoostingBackend]:
        s_schema = self._s_schema(schema, train)
        model = self._fit_component("classifier", params, train, y_train, valid, y_valid, s_schema, cache_key="s_outcome")
        return {"s_outcome": model}

    def _fit_t(
        self,
        params: Mapping[str, Any],
        train: pl.DataFrame,
        y_train: np.ndarray,
        treatment_train: np.ndarray,
        valid: pl.DataFrame,
        y_valid: np.ndarray,
        treatment_valid: np.ndarray,
        schema: FeatureSchema,
        *,
        prefix: str,
    ) -> dict[str, BaseBoostingBackend]:
        result: dict[str, BaseBoostingBackend] = {}
        for arm, label in ((0, "control"), (1, "treatment")):
            train_mask = treatment_train == arm
            valid_mask = treatment_valid == arm
            result[f"{prefix}_{label}_outcome"] = self._fit_component(
                "classifier",
                params,
                train.filter(pl.Series(train_mask)),
                y_train[train_mask],
                valid.filter(pl.Series(valid_mask)),
                y_valid[valid_mask],
                schema,
                cache_key=f"outcome:{arm}",
            )
        return result

    def _fit_x(
        self,
        params: Mapping[str, Any],
        train: pl.DataFrame,
        y_train: np.ndarray,
        treatment_train: np.ndarray,
        valid: pl.DataFrame,
        y_valid: np.ndarray,
        treatment_valid: np.ndarray,
        schema: FeatureSchema,
    ) -> dict[str, BaseBoostingBackend]:
        result = self._fit_t(
            params,
            train,
            y_train,
            treatment_train,
            valid,
            y_valid,
            treatment_valid,
            schema,
            prefix="x",
        )
        mu0_train = self._predict_component(result["x_control_outcome"], train, schema)
        mu1_train = self._predict_component(result["x_treatment_outcome"], train, schema)
        mu0_valid = self._predict_component(result["x_control_outcome"], valid, schema)
        mu1_valid = self._predict_component(result["x_treatment_outcome"], valid, schema)
        pseudo_train = {0: mu1_train - y_train, 1: y_train - mu0_train}
        pseudo_valid = {0: mu1_valid - y_valid, 1: y_valid - mu0_valid}
        for arm, label in ((0, "control"), (1, "treatment")):
            train_mask = treatment_train == arm
            valid_mask = treatment_valid == arm
            result[f"x_{label}_effect"] = self._fit_component(
                "regressor",
                params,
                train.filter(pl.Series(train_mask)),
                pseudo_train[arm][train_mask],
                valid.filter(pl.Series(valid_mask)),
                pseudo_valid[arm][valid_mask],
                schema,
            )
        if self.estimate_propensity:
            result["x_propensity"] = self._fit_component(
                "classifier",
                params,
                train,
                treatment_train,
                valid,
                treatment_valid,
                schema,
                cache_key="propensity",
            )
        return result

    def _predict_learner(
        self,
        learner: str,
        components: Mapping[str, BaseBoostingBackend],
        frame: pl.DataFrame,
        schema: FeatureSchema,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if learner == "s":
            s_schema = self._s_schema(schema, frame.with_columns(pl.lit(0, dtype=pl.Int8).alias(self.treatment_column)))
            control_frame = frame.with_columns(pl.lit(0, dtype=pl.Int8).alias(self.treatment_column))
            treatment_frame = frame.with_columns(pl.lit(1, dtype=pl.Int8).alias(self.treatment_column))
            control = self._predict_component(components["s_outcome"], control_frame, s_schema)
            treated = self._predict_component(components["s_outcome"], treatment_frame, s_schema)
            return treated - control, control, treated, None
        control = self._predict_component(components[f"{learner}_control_outcome"], frame, schema)
        treated = self._predict_component(components[f"{learner}_treatment_outcome"], frame, schema)
        if learner == "t":
            return treated - control, control, treated, None
        tau0 = np.clip(self._predict_component(components["x_control_effect"], frame, schema), -1.0, 1.0)
        tau1 = np.clip(self._predict_component(components["x_treatment_effect"], frame, schema), -1.0, 1.0)
        propensity = (
            np.clip(self._predict_component(components["x_propensity"], frame, schema), 0.0, 1.0)
            if "x_propensity" in components
            else np.full(frame.height, 0.5)
        )
        effect = propensity * tau0 + (1.0 - propensity) * tau1
        return effect, control, treated, propensity

    def _objective(
        self,
        learner: str,
        components: Mapping[str, BaseBoostingBackend],
        valid: pl.DataFrame,
        y_valid: np.ndarray,
        treatment_valid: np.ndarray,
        schema: FeatureSchema,
        metric: Metric,
    ) -> float:
        effect = self._predict_learner(learner, components, valid, schema)[0]
        try:
            value = metric.compute(MetricInput(y_valid, effect, treatment=treatment_valid))
            if value is None:
                msg = f"Validation {metric.name!r} returned no value for learner {learner!r}"
                raise SchemaError(msg)
            return float(value)
        except ValueError as exc:
            msg = f"Validation {metric.name!r} is undefined for learner {learner!r}: {exc}"
            raise SchemaError(msg) from exc

    def fit_composite(
        self,
        train: pl.DataFrame,
        y_train: np.ndarray,
        treatment_train: np.ndarray,
        valid: pl.DataFrame,
        y_valid: np.ndarray,
        treatment_valid: np.ndarray,
        schema: FeatureSchema,
        *,
        hyperopt: bool,
        n_trials: int | None,
        model_params: Mapping[str, Any],
        search_space: Mapping[str, Any] | None,
        metric: Metric,
        part_name: str,
    ) -> None:
        """Fit all learners and retain each already-fitted best composite trial."""
        self._validate_part(y_train, treatment_train, part_name)
        self._validate_part(y_valid, treatment_valid, f"{part_name} validation")
        fitters = {
            "s": lambda params: self._fit_s(params, train, y_train, valid, y_valid, schema),
            "t": lambda params: self._fit_t(
                params,
                train,
                y_train,
                treatment_train,
                valid,
                y_valid,
                treatment_valid,
                schema,
                prefix="t",
            ),
            "x": lambda params: self._fit_x(params, train, y_train, treatment_train, valid, y_valid, treatment_valid, schema),
        }
        effective_space = (
            resolve_default_search_space(
                self.engine,
                n_trials=n_trials or 0,
                train_frame=train,
                schema=schema,
            )
            if hyperopt and search_space is None
            else search_space
        )
        if hyperopt and n_trials is None:
            msg = "hyperopt=True requires a resolved n_trials value"
            raise RuntimeError(msg)
        preparation_backend = self._component("classifier", model_params)
        self._reuse_prepared = preparation_backend.can_prequantize(effective_space)
        self._prepared_cache = {}
        for learner_index, (learner, fitter) in enumerate(fitters.items(), start=1):
            learner_started = perf_counter()
            log_progress("[uplift learner %d/%d] learner=%s started", learner_index, len(fitters), learner)
            if not hyperopt:
                components = fitter(dict(model_params))
                value = self._objective(learner, components, valid, y_valid, treatment_valid, schema, metric)
                self.components.update(components)
                self.learner_params[learner] = dict(model_params)
                self.learner_metrics[learner] = value
                log_progress(
                    "[uplift learner %d/%d] learner=%s completed duration_seconds=%.3f objective_value=%.12g",
                    learner_index,
                    len(fitters),
                    learner,
                    perf_counter() - learner_started,
                    value,
                )
                continue
            try:
                import optuna
            except ImportError as exc:
                msg = "hyperopt=True requires the 'optuna' package"
                raise MissingDependencyError(msg) from exc
            best_components: dict[str, BaseBoostingBackend] | None = None
            best_params: dict[str, Any] = {}
            best_value = float("-inf")

            def objective(
                trial: Any,
                *,
                current_fitter=fitter,
                current_learner=learner,
                current_learner_index=learner_index,
            ) -> float:
                nonlocal best_components, best_params, best_value
                trial_started = perf_counter()
                params = suggest_params(
                    trial,
                    engine=self.engine,
                    model_params=model_params,
                    search_space=effective_space,
                )
                trial_components = current_fitter(params)
                value = self._objective(current_learner, trial_components, valid, y_valid, treatment_valid, schema, metric)
                if best_components is None or value > best_value:
                    best_components, best_params, best_value = trial_components, params, value
                log_progress(
                    "[uplift learner %d/%d][optuna %d/%d] learner=%s completed duration_seconds=%.3f objective_value=%.12g",
                    current_learner_index,
                    len(fitters),
                    trial.number + 1,
                    n_trials,
                    current_learner,
                    perf_counter() - trial_started,
                    value,
                )
                return value

            log_progress(
                "[uplift learner %d/%d][optuna 0/%d] learner=%s creating study random_state=%d",
                learner_index,
                len(fitters),
                n_trials,
                learner,
                self.random_state,
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(self.verbose))
            if best_components is None:
                msg = f"Hyperparameter search produced no fitted {learner!r} learner"
                raise RuntimeError(msg)
            self.components.update(best_components)
            self.learner_params[learner] = best_params
            self.learner_metrics[learner] = best_value
            log_progress(
                "[uplift learner %d/%d] learner=%s completed duration_seconds=%.3f best_value=%.12g",
                learner_index,
                len(fitters),
                learner,
                perf_counter() - learner_started,
                best_value,
            )

    def predict_score(self, frame: pl.DataFrame, schema: FeatureSchema) -> np.ndarray:
        """Return S/T/X effects, potential outcomes and X propensity."""
        missing = sorted(set(_COMPONENT_KIND) - {"x_propensity"} - set(self.components))
        if missing:
            msg = f"Uplift composite is missing constituent models: {missing}"
            raise ArtifactIntegrityError(msg)
        s = self._predict_learner("s", self.components, frame, schema)
        t = self._predict_learner("t", self.components, frame, schema)
        x = self._predict_learner("x", self.components, frame, schema)
        values = np.column_stack((s[0], s[1], s[2], t[0], t[1], t[2], x[0], x[1], x[2], x[3]))
        if not np.isfinite(values).all():
            msg = "Uplift prediction contains NaN or infinite values"
            raise SchemaError(msg)
        return values

    def feature_importance(self, schema: FeatureSchema) -> pl.DataFrame | None:
        """Return long-format importance for every fitted constituent model."""
        rows: list[dict[str, Any]] = []
        for component_name, component in self.components.items():
            if component_name == "s_outcome":
                feature_names = (*schema.feature_order, self.treatment_column)
            else:
                feature_names = schema.feature_order
            values = component._native_feature_importances(len(feature_names))
            if values is None:
                continue
            for feature, importance in zip(feature_names, values, strict=True):
                rows.append({"component": component_name, "feature": feature, "importance": float(importance)})
        return pl.DataFrame(rows).sort(["component", "importance"], descending=[False, True]) if rows else None

    def save(self, path: Path) -> None:
        """Persist every constituent estimator through its native format."""
        if not self.components:
            msg = "Cannot save an unfitted uplift composite"
            raise ArtifactIntegrityError(msg)
        path.mkdir(parents=True, exist_ok=True)
        for name, component in self.components.items():
            component.save(path / "components" / name)
        metadata = {
            "engine": self.engine,
            "params": dict(self.params),
            "random_state": self.random_state,
            "verbose": self.verbose,
            "estimate_propensity": self.estimate_propensity,
            "treatment_column": self.treatment_column,
            "components": {name: _COMPONENT_KIND[name] for name in self.components},
            "learner_params": self.learner_params,
            "learner_metrics": self.learner_metrics,
        }
        (path / "backend.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "UpliftBoostingBackend":
        """Restore the composite and every native constituent model."""
        metadata_path = path / "backend.json"
        if not metadata_path.exists():
            msg = f"Missing uplift backend metadata: {metadata_path}"
            raise ArtifactIntegrityError(msg)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        backend = cls(
            engine=metadata["engine"],
            params=metadata.get("params", {}),
            random_state=metadata["random_state"],
            device=device,
            verbose=metadata.get("verbose", False),
            estimate_propensity=metadata.get("estimate_propensity", False),
            treatment_column=metadata.get("treatment_column", "treatment"),
            learner_params={name: dict(value) for name, value in metadata.get("learner_params", {}).items()},
            learner_metrics={name: float(value) for name, value in metadata.get("learner_metrics", {}).items()},
        )
        for name, kind in metadata.get("components", {}).items():
            component_class = BinaryBoostingBackend if kind == "classifier" else RegressionBoostingBackend
            backend.components[name] = component_class.load(path / "components" / name, device=device)
        return backend

    def set_runtime_device(self, device: str) -> None:
        """Propagate an inference device override to every learner component."""
        super().set_runtime_device(device)
        for component in self.components.values():
            component.set_runtime_device(device)

    def for_execution(self, device: str) -> "UpliftBoostingBackend":
        """Isolate device overrides in the constituent models that need them."""
        if device == self.device:
            return self
        backend = copy(self)
        BoostingBackend.set_runtime_device(backend, device)
        backend.components = {name: component.for_execution(device) for name, component in self.components.items()}
        return backend
