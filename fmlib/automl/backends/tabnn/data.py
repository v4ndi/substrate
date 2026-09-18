"""One cache contract, in both directions: encode once, then stream.

Preprocessing is not a public action. ``train()`` is handed raw parquet, and
before the first trial can start the data has to be fitted, encoded and written
somewhere every trial and every job can stream from. This module owns that
directory: how it is named, how it is published, when it may be reused and how
it is read back.

**Named by content, checked by manifest, and both are needed.** The directory
name is a hash over the source identity, the preprocessing configuration, the
schema and a contract version, so "the data changed" is simply a different
directory rather than a mistake waiting to happen. The manifest inside it is
what proves the directory is whole.

**The manifest is a completion marker, written last.** A transform writes a
batch of ``part-*.parquet``. After a crash the directory holds, say, 17 files
of 40, and nothing distinguishes that from a dataset of 17 files: a truncated
file would at least fail to open, because parquet writes its footer on close,
but a *missing* file raises nothing at all and training would quietly run on
half the data. So the manifest lists the files that are supposed to be there.

**Publication is atomic.** Encoding writes into ``<key>.tmp-<pid>``, and the
directory is renamed into place only once every file is closed. An unfinished
directory is never visible under the real name. The side benefit is that
concurrent writers are safe: two processes encoding the same key each write
their own temp, one rename wins, the loser finds the target complete, drops its
temp and reads what is there.

**Deletion is an operation, not a setting.** The processed data outlives
``train()`` -- ``calibrate()`` is a separate call, so are the report, a repeated
``evaluate`` and a resumed search -- so nothing here removes anything on its
own. ``clear_processed_data`` is what a user calls.

There is no resume for a half-encoded directory: a crash at 80% re-encodes from
scratch. Resuming would mean persisting accumulator state and proving the
vocabulary saw every shard, which is not worth it for a rare case against a
single streaming pass that :func:`fit` already parallelises.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from fmlib.automl.data import FeatureSchema, ParquetSource
from fmlib.automl.exceptions import ArtifactIntegrityError, SchemaError
from fmlib.automl.progress import log_progress
from fmlib.preprocessing.local import TabularPreprocessor

#: Bumped when the layout or the meaning of what is written changes. It is part
#: of the key, so old directories are simply never matched again.
CONTRACT_VERSION = "1"

MANIFEST_NAME = "_manifest.json"
PREPROCESSOR_NAME = "preprocessor.yaml"
INDEX_NAME = "index.json"

__all__ = [
    "CONTRACT_VERSION",
    "ProcessedData",
    "ProcessedSplit",
    "build_schema",
    "clear_processed_data",
    "content_key",
    "prepare_processed_data",
]


@dataclass(frozen=True)
class ProcessedSplit:
    """One encoded split: where it lives and what it is supposed to contain."""

    path: Path
    rows: int
    parts: tuple[str, ...]


@dataclass(frozen=True)
class ProcessedData:
    """An encoded, published pair of splits and everything needed to read it.

    Attributes:
        root: The content-addressed directory.
        key: Its name, the content key.
        train: The encoded training split.
        valid: The encoded validation split.
        preprocessor_path: The fitted preprocessor, as its own file.
        vocab_size: Size of the shared categorical embedding table.
        num_numerical: How many numeric features a record carries.
        hidden_states: ``{column: width}`` of the pass-through embeddings.
        identity_columns: Columns written through untouched, physical names.
        target_column: Physical name of the target inside the encoded files.
        reused: Whether this run found the directory already complete.
    """

    root: Path
    key: str
    train: ProcessedSplit
    valid: ProcessedSplit
    preprocessor_path: Path
    vocab_size: int
    num_numerical: int
    hidden_states: Mapping[str, int]
    identity_columns: tuple[str, ...]
    target_column: str
    reused: bool = False

    def preprocessor(self) -> TabularPreprocessor:
        """Load the fitted preprocessor that produced these files."""
        return TabularPreprocessor.load(
            yaml.safe_load(self.preprocessor_path.read_text(encoding="utf-8"))
        )


# --------------------------------------------------------------------------- #
# Schema, from parquet footers                                                 #
# --------------------------------------------------------------------------- #
def _hidden_state_width(source: ParquetSource, column: str) -> int:
    """Resolve the width of one embedding column, reading as little as possible.

    A fixed-size ``Array`` declares its width in the dtype, so the footers are
    enough. A ``List`` does not, and one row has to be read to find out.
    """
    dtype = source.scan().collect_schema().get(column)
    if isinstance(dtype, pl.Array):
        return int(dtype.size)
    if not isinstance(dtype, pl.List):
        msg = f"Hidden-state column {column!r} must be List or Array; got {dtype}"
        raise SchemaError(msg)
    head = source.scan().select(pl.col(column).list.len()).head(1).collect()
    if head.height == 0 or head.item() in (None, 0):
        msg = f"Hidden-state column {column!r} must have one non-zero fixed dimension"
        raise SchemaError(msg)
    return int(head.item())


def build_schema(
    source: ParquetSource,
    config: Any,
    to_external: Mapping[str, str] | None = None,
    categorical_role_columns: Sequence[str | None] = (),
) -> FeatureSchema:
    """Derive the feature schema from parquet footers, without reading rows.

    Names in the result are canonical, as everywhere else in the task layer;
    ``to_external`` says what those columns are actually called in the files.

    Args:
        source: The training split.
        config: Canonical (internal) task configuration.
        to_external: Canonical-to-physical role names.
        categorical_role_columns: Role columns that count as categorical
            features in this layout -- the group column for a global model, the
            treatment column for a response task. The same convention the
            boosting path applies, so the two families see the same features.

    Returns:
        A schema whose ``hidden_states`` carries widths and whose
        ``feature_order`` does not mention the embeddings at all -- they reach
        the model as vectors, not as one scalar feature per coordinate.

    Raises:
        SchemaError: If a configured column is missing from the source, or an
            embedding column has no usable width.
    """
    to_external = dict(to_external or {})
    physical = source.scan().collect_schema()
    present = set(physical.names())

    def external(name: str) -> str:
        return to_external.get(name, name)

    required = [
        config.target_column,
        config.client_id_column,
        *(column for column in categorical_role_columns if column is not None),
        *config.categorical_columns,
        *config.numerical_columns,
        *config.hidden_state_columns,
        *(
            column
            for column in (
                config.date_column,
                config.group_column,
                getattr(config, "treatment_column", None),
            )
            if column is not None
        ),
    ]
    missing = sorted(
        {name for name in required if external(name) not in present},
    )
    if missing:
        msg = f"Missing required columns in the training source: {sorted(external(name) for name in missing)}"
        raise SchemaError(msg)

    hidden_states = {
        column: _hidden_state_width(source, external(column))
        for column in config.hidden_state_columns
    }
    categorical = list(config.categorical_columns)
    for column in categorical_role_columns:
        if column and column not in categorical:
            categorical.append(column)
    categorical = tuple(categorical)
    numerical = tuple(config.numerical_columns)
    if not categorical and not numerical and not hidden_states:
        msg = "No features configured; set categorical_columns, numerical_columns or hidden_state_columns"
        raise SchemaError(msg)
    feature_order = (*categorical, *numerical)
    return FeatureSchema(
        categorical=categorical,
        numerical=numerical,
        feature_order=feature_order,
        dtypes={name: str(physical.get(external(name))) for name in feature_order},
        target_column=config.target_column,
        client_id_column=config.client_id_column,
        treatment_column=getattr(config, "treatment_column", None),
        group_column=config.group_column,
        hidden_states=hidden_states,
    )


# --------------------------------------------------------------------------- #
# Content key                                                                  #
# --------------------------------------------------------------------------- #
def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def content_key(
    *,
    train_manifest: Sequence[Mapping[str, Any]],
    valid_manifest: Sequence[Mapping[str, Any]],
    preprocessing: Mapping[str, Any],
    schema: FeatureSchema,
) -> str:
    """Name the directory after everything that determines its content.

    The identity of a source is ``path + size + modified_ns``. A file
    overwritten with the same size and the same mtime therefore does not change
    the key -- a known weak spot, documented rather than hidden. Where a source
    offers an etag or a version, that belongs here instead.

    Args:
        train_manifest: File identity of the training source.
        valid_manifest: File identity of the validation source.
        preprocessing: Everything that configures the preprocessor.
        schema: The feature schema the encoding follows.

    Returns:
        A hex digest, short enough to read and long enough not to collide.
    """
    payload = _stable({
        "contract": CONTRACT_VERSION,
        "train": [dict(item) for item in train_manifest],
        "valid": [dict(item) for item in valid_manifest],
        "preprocessing": dict(preprocessing),
        "schema": schema.to_dict(),
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Publish and reuse                                                            #
# --------------------------------------------------------------------------- #
def _read_manifest(root: Path) -> dict[str, Any] | None:
    """Return the completion marker, or ``None`` when the directory has none."""
    path = root / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = (
            f"Processed data at {root} has an unreadable completion marker: {error}. "
            "Delete the directory or point processed_data_path somewhere else."
        )
        raise ArtifactIntegrityError(msg) from error


def _validate(root: Path, manifest: Mapping[str, Any], key: str) -> None:
    """Refuse to use a directory whose marker and content disagree."""
    problems = []
    if manifest.get("complete") is not True:
        problems.append("the marker does not say the encoding finished")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        problems.append(
            f"contract version {manifest.get('contract_version')!r} != {CONTRACT_VERSION!r}"
        )
    if manifest.get("key") != key:
        problems.append(f"marker names key {manifest.get('key')!r}, not {key!r}")
    for split in ("train", "valid"):
        for part in manifest.get(split, {}).get("parts", ()):
            if not (root / split / part).is_file():
                problems.append(f"{split}/{part} is listed but missing")
    if problems:
        msg = (
            f"Processed data at {root} is not usable: {'; '.join(problems)}. "
            "Nothing was overwritten. Delete the directory, or point "
            "processed_data_path somewhere else."
        )
        raise ArtifactIntegrityError(msg)


def _split_from_manifest(root: Path, manifest: Mapping[str, Any], split: str):
    payload = manifest[split]
    return ProcessedSplit(
        path=root / split,
        rows=int(payload["rows"]),
        parts=tuple(payload["parts"]),
    )


def _written_parts(directory: Path) -> tuple[tuple[str, ...], int]:
    parts = sorted(path.name for path in directory.glob("*.parquet"))
    rows = sum(
        pl.scan_parquet(directory / name).select(pl.len()).collect().item()
        for name in parts
    )
    return tuple(parts), int(rows)


def _update_index(processed_root: Path, key: str, entry: Mapping[str, Any]) -> None:
    """Keep a readable list of what lives under ``processed_data_path``.

    Without it the directory is a wall of hex and nobody can tell what is safe
    to delete. A failure to write it is not a failure of the encoding.
    """
    path = processed_root / INDEX_NAME
    try:
        index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        index = {}
    index[key] = dict(entry)
    temporary = path.with_name(f"{INDEX_NAME}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)


def prepare_processed_data(
    *,
    config: Any,
    schema: FeatureSchema,
    train_source: ParquetSource,
    valid_source: ParquetSource,
    train_manifest: Sequence[Mapping[str, Any]],
    valid_manifest: Sequence[Mapping[str, Any]],
    to_external: Mapping[str, str] | None = None,
    num_workers: int = 1,
    preprocessor_kwargs: Mapping[str, Any] | None = None,
) -> ProcessedData:
    """Encode both splits once, publish atomically, and reuse when possible.

    The preprocessor is fitted on the **full** training split and applied to
    validation without refitting; ``model_layout`` later divides data that is
    already encoded, so every part shares one vocabulary and one scaling.

    Args:
        config: Canonical task configuration.
        schema: Schema from :func:`build_schema`.
        train_source: Raw training parquet.
        valid_source: Raw validation parquet.
        train_manifest: File identity of ``train_source``.
        valid_manifest: File identity of ``valid_source``.
        to_external: Canonical-to-physical role names.
        num_workers: Processes for the fit and the transform.
        preprocessor_kwargs: Extra arguments for :class:`TabularPreprocessor`.

    Returns:
        The published directory, with ``reused`` saying whether this call
        encoded anything.

    Raises:
        ArtifactIntegrityError: If a directory with the right name exists but
            its marker and content disagree. Nothing is overwritten.
    """
    names = dict(to_external or {})

    def external(name: str) -> str:
        return names.get(name, name)

    identity = tuple(
        dict.fromkeys(
            external(name)
            for name in (
                config.target_column,
                config.client_id_column,
                config.date_column,
                config.group_column,
                getattr(config, "treatment_column", None),
                *config.hidden_state_columns,
            )
            if name is not None
        )
    )
    preprocessing = {
        "categorical_columns": [external(name) for name in schema.categorical],
        "numeric_columns": [external(name) for name in schema.numerical],
        "identity_cols": list(identity),
        **dict(preprocessor_kwargs or {}),
    }

    key = content_key(
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        preprocessing=preprocessing,
        schema=schema,
    )
    processed_root = Path(config.resolved_processed_data_path)
    root = processed_root / key

    manifest = _read_manifest(root)
    if manifest is not None:
        _validate(root, manifest, key)
        log_progress("[tabnn data] reusing processed data key=%s", key)
        return ProcessedData(
            root=root,
            key=key,
            train=_split_from_manifest(root, manifest, "train"),
            valid=_split_from_manifest(root, manifest, "valid"),
            preprocessor_path=root / PREPROCESSOR_NAME,
            vocab_size=int(manifest["vocab_size"]),
            num_numerical=int(manifest["num_numerical"]),
            hidden_states=dict(manifest["hidden_states"]),
            identity_columns=tuple(manifest["identity_columns"]),
            target_column=manifest["target_column"],
            reused=True,
        )

    processed_root.mkdir(parents=True, exist_ok=True)
    staging = processed_root / f"{key}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        log_progress(
            "[tabnn data] encoding key=%s workers=%d cat=%d num=%d hidden=%d",
            key,
            num_workers,
            len(schema.categorical),
            len(schema.numerical),
            len(schema.hidden_states),
        )
        preprocessor = TabularPreprocessor(
            categorical_columns=preprocessing["categorical_columns"] or None,
            numeric_columns=preprocessing["numeric_columns"] or None,
            **dict(preprocessor_kwargs or {}),
        )
        preprocessor.fit(
            [str(path) for path in train_source.files], num_workers=num_workers
        )
        splits = {}
        for name, source in (("train", train_source), ("valid", valid_source)):
            directory = staging / name
            directory.mkdir()
            files = [str(path) for path in source.files]
            # One worker writes one file and is handed that file; several write
            # one part each and are handed the directory. The number of parts
            # follows the number of source files either way.
            destination = (
                str(directory)
                if num_workers > 1
                else str(directory / "part-00000.parquet")
            )
            preprocessor.transform(
                files,
                destination,
                identity_cols=list(identity),
                output="packed",
                num_workers=num_workers,
            )
            parts, rows = _written_parts(directory)
            splits[name] = ProcessedSplit(path=root / name, rows=rows, parts=parts)

        (staging / PREPROCESSOR_NAME).write_text(
            yaml.safe_dump(preprocessor.dump(), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        payload = {
            "complete": True,
            "key": key,
            "contract_version": CONTRACT_VERSION,
            "vocab_size": int(preprocessor.vocab_size),
            "num_numerical": len(schema.numerical),
            "hidden_states": dict(schema.hidden_states),
            "identity_columns": list(identity),
            "target_column": external(config.target_column),
            "preprocessing": preprocessing,
            "schema": schema.to_dict(),
            "sources": {
                "train": [dict(item) for item in train_manifest],
                "valid": [dict(item) for item in valid_manifest],
            },
            "train": {
                "rows": splits["train"].rows,
                "parts": list(splits["train"].parts),
            },
            "valid": {
                "rows": splits["valid"].rows,
                "parts": list(splits["valid"].parts),
            },
        }
        # Written last, and only after every parquet file is closed: this is
        # the marker that makes the directory usable.
        (staging / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        if root.exists() and _read_manifest(root) is None:
            # A directory without a marker is an interrupted encoding, and by
            # construction unusable. Publication is a rename, so nothing else
            # can have left one here mid-flight.
            shutil.rmtree(root, ignore_errors=True)
        try:
            os.rename(staging, root)
        except OSError:
            # Someone else published this key while we were encoding it. Their
            # directory is as good as ours by construction; drop ours.
            if _read_manifest(root) is None:
                raise
            shutil.rmtree(staging, ignore_errors=True)
            return prepare_processed_data(
                config=config,
                schema=schema,
                train_source=train_source,
                valid_source=valid_source,
                train_manifest=train_manifest,
                valid_manifest=valid_manifest,
                to_external=names,
                num_workers=num_workers,
                preprocessor_kwargs=preprocessor_kwargs,
            )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    _update_index(
        processed_root,
        key,
        {
            "train_rows": splits["train"].rows,
            "valid_rows": splits["valid"].rows,
            "train_source": [item["path"] for item in train_manifest],
            "valid_source": [item["path"] for item in valid_manifest],
        },
    )
    log_progress(
        "[tabnn data] published key=%s train_rows=%d valid_rows=%d parts=%d/%d",
        key,
        splits["train"].rows,
        splits["valid"].rows,
        len(splits["train"].parts),
        len(splits["valid"].parts),
    )
    return ProcessedData(
        root=root,
        key=key,
        train=splits["train"],
        valid=splits["valid"],
        preprocessor_path=root / PREPROCESSOR_NAME,
        vocab_size=int(preprocessor.vocab_size),
        num_numerical=len(schema.numerical),
        hidden_states=dict(schema.hidden_states),
        identity_columns=identity,
        target_column=external(config.target_column),
    )


def clear_processed_data(
    processed_data_path: str | os.PathLike, key: str | None = None
) -> int:
    """Delete encoded data. Deletion is a call, never a side effect.

    Args:
        processed_data_path: Directory holding the content-addressed caches.
        key: One key to remove; ``None`` removes all of them.

    Returns:
        How many directories were removed.
    """
    root = Path(processed_data_path)
    if not root.is_dir():
        return 0
    removed = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if key is not None and child.name != key:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    if key is None:
        (root / INDEX_NAME).unlink(missing_ok=True)
    else:
        index_path = root / INDEX_NAME
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                index = {}
            index.pop(key, None)
            index_path.write_text(
                json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
            )
    return removed
