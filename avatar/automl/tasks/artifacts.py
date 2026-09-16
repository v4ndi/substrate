"""Native artifact storage independent of AutoML entity lifecycle."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from avatar.automl.backends.boosting.interface import BoostingBackend
from avatar.automl.config.base import BaseTaskConfig
from avatar.automl.data import FeatureSchema
from avatar.automl.exceptions import ArtifactError, ArtifactIntegrityError
from avatar.automl.execution import _freeze
from avatar.automl.lifecycle import read_json

from .state import ModelEntry


def jsonable(value: Any) -> Any:
    """Encode config/manifest values without serializing executable state."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ArtifactState:
    config: BaseTaskConfig
    models: tuple[ModelEntry, ...]
    source_manifests: Mapping[str, tuple[dict[str, Any], ...]]
    task_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze(self.config))
        object.__setattr__(self, "source_manifests", _freeze(self.source_manifests))
        object.__setattr__(self, "task_manifest", _freeze(self.task_manifest))


@dataclass(frozen=True)
class ArtifactRepository:
    task_name: str
    artifact_directory: str
    backend_class: type[BoostingBackend]

    def save(
        self,
        state: ArtifactState,
        path: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Atomically save config, schema, metadata and native model files."""
        artifact_config = state.config
        artifact_path = (
            Path(path)
            if path is not None
            else Path(artifact_config.output_dir)
            / "artifacts"
            / self.artifact_directory
        )
        target = artifact_path.expanduser().resolve()
        if target.exists() and not overwrite:
            msg = f"Artifact path already exists: {target}"
            raise ArtifactError(msg)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir(parents=True)
        backup: Path | None = None
        try:
            config_payload = jsonable(asdict(artifact_config))
            (temporary / "config.json").write_text(
                json.dumps(config_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            model_metadata: list[dict[str, Any]] = []
            for index, item in enumerate(state.models):
                relative = Path("model") / str(index)
                item.backend.save(temporary / relative)
                model_metadata.append({
                    "path": relative.as_posix(),
                    "schema": item.schema.to_dict(),
                    "hidden_dimensions": item.hidden_dimensions,
                    "best_params": item.best_params,
                    "validation_metric": item.validation_metric,
                    "model_layout": item.layout,
                    "group_value": item.group_value,
                    "single_group_global": item.single_group_global,
                    "single_group_value": item.single_group_value,
                })
            manifest = {
                "task": self.task_name,
                "backend_family": artifact_config.backend,
                "engine": artifact_config.engine,
                "optimization_metric": artifact_config.optimization_metric,
                "models": model_metadata,
                "source_manifests": state.source_manifests,
            } | dict(state.task_manifest)
            (temporary / "manifest.json").write_text(
                json.dumps(jsonable(manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if target.exists():
                backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.bak")
                os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except Exception:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                raise
            if backup is not None:
                import shutil

                shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            if temporary.exists():
                import shutil

                shutil.rmtree(temporary)
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        return target

    @staticmethod
    def read_config(root: Path, config_class: type[BaseTaskConfig]) -> BaseTaskConfig:
        path = root / "config.json"
        try:
            payload = read_json(path)
            return config_class.from_mapping(payload)
        except ArtifactIntegrityError:
            raise
        except Exception as exc:
            msg = f"Artifact config is missing or invalid: {path}: {exc}"
            raise ArtifactIntegrityError(msg) from exc

    @staticmethod
    def _read_manifest(root: Path) -> Mapping[str, Any]:
        path = root / "manifest.json"
        try:
            return read_json(path)
        except Exception as exc:
            msg = f"Artifact manifest is missing or invalid: {path}: {exc}"
            raise ArtifactIntegrityError(msg) from exc

    def _validate_manifest(
        self,
        root: Path,
        config: BaseTaskConfig,
        manifest: Mapping[str, Any],
        *,
        require_complete: bool,
    ) -> tuple[tuple[Mapping[str, Any], FeatureSchema, Path], ...]:
        expected_header = {
            "task": self.task_name,
            "backend_family": config.backend,
            "engine": config.engine,
        }
        for field, expected in expected_header.items():
            actual = manifest.get(field)
            if actual != expected:
                msg = (
                    f"Artifact {field} mismatch: expected {expected!r}, got {actual!r}"
                )
                raise ArtifactIntegrityError(msg)

        raw_models = manifest.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            msg = "Artifact manifest models must be a non-empty list"
            raise ArtifactIntegrityError(msg)

        validated = []
        identities: set[tuple[str, Any]] = set()
        for index, item in enumerate(raw_models):
            if not isinstance(item, Mapping):
                msg = f"Artifact model #{index} metadata must be a mapping"
                raise ArtifactIntegrityError(msg)
            if "scope" in item:
                msg = "Artifact uses incompatible legacy model scope metadata"
                raise ArtifactIntegrityError(msg)
            layout = item.get("model_layout")
            if layout not in {"global", "per_group"}:
                msg = f"Artifact model_layout is unsupported: {layout!r}; expected 'global' or 'per_group'"
                raise ArtifactIntegrityError(msg)
            group_value = item.get("group_value")
            single_group_global = item.get("single_group_global", False)
            if not isinstance(single_group_global, bool) or (
                single_group_global and layout != "global"
            ):
                msg = (
                    f"Artifact model #{index} has invalid single_group_global metadata"
                )
                raise ArtifactIntegrityError(msg)
            single_group_value = item.get("single_group_value")
            if single_group_global != (single_group_value is not None):
                msg = f"Artifact model #{index} has inconsistent single-group metadata"
                raise ArtifactIntegrityError(msg)
            if (layout == "global" and group_value is not None) or (
                layout == "per_group" and group_value is None
            ):
                msg = f"Artifact model #{index} has inconsistent model_layout={layout!r} and group_value={group_value!r}"
                raise ArtifactIntegrityError(msg)
            try:
                identity = (layout, group_value)
                duplicate = identity in identities
                identities.add(identity)
            except TypeError as exc:
                msg = f"Artifact model #{index} group_value must be a scalar JSON value"
                raise ArtifactIntegrityError(msg) from exc
            if duplicate:
                msg = f"Artifact contains duplicate model identity: {identity!r}"
                raise ArtifactIntegrityError(msg)

            relative_value = item.get("path")
            if not isinstance(relative_value, str) or not relative_value:
                msg = f"Artifact model #{index} path must be a non-empty relative path"
                raise ArtifactIntegrityError(msg)
            relative = Path(relative_value)
            model_root = (root / relative).resolve()
            try:
                model_root.relative_to(root)
            except ValueError as exc:
                msg = f"Artifact model #{index} path escapes the artifact root: {relative_value!r}"
                raise ArtifactIntegrityError(msg) from exc
            if relative.is_absolute() or not model_root.is_dir():
                msg = f"Artifact model #{index} directory is missing or invalid: {model_root}"
                raise ArtifactIntegrityError(msg)

            try:
                schema = FeatureSchema.from_dict(item["schema"])
                hidden_dimensions = item["hidden_dimensions"]
                best_params = item["best_params"]
                float(item["validation_metric"])
            except Exception as exc:
                msg = f"Artifact model #{index} metadata or schema is invalid: {exc}"
                raise ArtifactIntegrityError(msg) from exc
            if not isinstance(hidden_dimensions, Mapping) or not isinstance(
                best_params, Mapping
            ):
                msg = f"Artifact model #{index} hidden_dimensions and best_params must be mappings"
                raise ArtifactIntegrityError(msg)
            features = tuple(schema.feature_order)
            if (
                len(features) != len(set(features))
                or set(features) != set(schema.categorical) | set(schema.numerical)
                or set(schema.categorical) & set(schema.numerical)
                or set(schema.dtypes) != set(features)
            ):
                msg = f"Artifact model #{index} feature schema is inconsistent"
                raise ArtifactIntegrityError(msg)

            backend_metadata_path = model_root / "backend.json"
            try:
                backend_metadata = read_json(backend_metadata_path)
            except Exception as exc:
                msg = f"Artifact backend metadata is missing or invalid: {backend_metadata_path}: {exc}"
                raise ArtifactIntegrityError(msg) from exc
            backend_engine = backend_metadata.get("engine")
            if backend_engine != config.engine:
                msg = f"Artifact backend engine mismatch: expected {config.engine!r}, got {backend_engine!r}"
                raise ArtifactIntegrityError(msg)
            validated.append((item, schema, model_root))

        layouts = [str(item[0].get("model_layout")) for item in validated]
        expected_layout = config.resolved_model_layout
        allowed_layouts = (
            {"global"}
            if expected_layout == "global"
            else {"per_group"}
            if expected_layout == "per_group"
            else {"global", "per_group"}
        )
        valid_layout_set = set(layouts) <= allowed_layouts
        if require_complete:
            valid_layout_set = valid_layout_set and (
                layouts == ["global"]
                if expected_layout == "global"
                else all(layout == "per_group" for layout in layouts)
                if expected_layout == "per_group"
                else layouts.count("global") == 1
                and (
                    "per_group" in layouts
                    or validated[0][0].get("single_group_global", False)
                )
            )
        if not valid_layout_set:
            msg = f"Artifact model layouts mismatch: config expects {expected_layout!r}, got {layouts!r}"
            raise ArtifactIntegrityError(msg)

        source_manifests = manifest.get("source_manifests", {})
        if not isinstance(source_manifests, Mapping) or any(
            not isinstance(paths, list | tuple)
            or any(not isinstance(item, Mapping) for item in paths)
            for paths in source_manifests.values()
        ):
            msg = "Artifact source_manifests must map split names to lists of file metadata"
            raise ArtifactIntegrityError(msg)
        return tuple(validated)

    def restore(
        self,
        root: Path,
        config: BaseTaskConfig,
        *,
        manifest: Mapping[str, Any] | None = None,
        require_complete: bool = False,
        runtime_device: str | None = None,
    ) -> ArtifactState:
        """Read fitted values; the caller explicitly adopts the returned state."""
        root = root.expanduser().resolve()
        device = runtime_device or config.resolved_device
        if manifest is None:
            manifest = self._read_manifest(root)
        validated_models = self._validate_manifest(
            root, config, manifest, require_complete=require_complete
        )
        runtime_device = device
        models = []
        source_manifests = {
            split: tuple(dict(item) for item in paths)
            for split, paths in manifest.get("source_manifests", {}).items()
        }
        for index, (item, schema, model_root) in enumerate(validated_models):
            layout = item["model_layout"]
            try:
                models.append(
                    ModelEntry(
                        backend=self.backend_class.load(
                            model_root, device=runtime_device
                        ),
                        schema=schema,
                        hidden_dimensions=dict(item["hidden_dimensions"]),
                        best_params=dict(item["best_params"]),
                        validation_metric=float(item["validation_metric"]),
                        layout=layout,
                        group_value=item.get("group_value"),
                        single_group_global=item.get("single_group_global", False),
                        single_group_value=item.get("single_group_value"),
                    )
                )
            except ArtifactIntegrityError:
                raise
            except Exception as exc:
                msg = f"Artifact native model #{index} cannot be restored from {model_root}: {exc}"
                raise ArtifactIntegrityError(msg) from exc

        return ArtifactState(config, tuple(models), source_manifests, manifest)
