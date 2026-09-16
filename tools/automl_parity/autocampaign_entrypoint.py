"""Launch autocampaignxfm with parity-only ranges and safe Windows paths."""

from __future__ import annotations

import builtins
import io
import os
import runpy
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ENCODING = {"<": "__ac_lt__", ">": "__ac_gt__"}


def _encode_path(path: Any) -> Any:
    """Replace Windows-forbidden angle brackets in path-like values."""
    if isinstance(path, int):
        return path
    value = os.fspath(path)
    if not isinstance(value, str):
        return path
    for character, replacement in _ENCODING.items():
        value = value.replace(character, replacement)
    return value


def _decode_name(name: Any) -> Any:
    """Restore autocampaignxfm's logical filename returned by ``os.listdir``."""
    if not isinstance(name, str):
        return name
    for character, replacement in _ENCODING.items():
        name = name.replace(replacement, character)
    return name


@contextmanager
def windows_filename_compatibility() -> Iterator[None]:
    """Virtualize angle brackets in filenames for one Windows reference process."""
    if os.name != "nt":
        yield
        return
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_listdir = os.listdir

    def compatible_builtin_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        return original_builtin_open(_encode_path(file), *args, **kwargs)

    def compatible_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        return original_io_open(_encode_path(file), *args, **kwargs)

    def compatible_listdir(path: Any = ".") -> list[Any]:
        return [_decode_name(name) for name in original_listdir(_encode_path(path))]

    builtins.open = compatible_builtin_open
    io.open = compatible_io_open
    os.listdir = compatible_listdir
    try:
        yield
    finally:
        builtins.open = original_builtin_open
        io.open = original_io_open
        os.listdir = original_listdir


@contextmanager
def configured_optuna_ranges() -> Iterator[None]:
    """Make the reference booster consume ranges from its native runtime YAML.

    ``autocampaignxfm`` exposes ``optuna_ranges`` in Hydra configuration, but
    its learner factory does not attach that configuration to
    ``BoosterWrapper``.  The parity process supplies the missing connection so
    test-only ranges exercise the real entrypoints without modifying the
    reference repository.
    """
    try:
        config_path_index = sys.argv.index("--config-path") + 1
        config_name_index = sys.argv.index("--config-name") + 1
    except ValueError:
        yield
        return

    from omegaconf import OmegaConf
    from uplift_metalearner.boosting.booster import BoosterWrapper

    config_directory = Path(sys.argv[config_path_index])
    config_name = Path(sys.argv[config_name_index]).stem
    config_file = config_directory / f"{config_name}.yaml"
    config = OmegaConf.load(config_file)
    ranges = OmegaConf.to_container(OmegaConf.select(config, "optuna_ranges") or {}, resolve=True)
    original = BoosterWrapper._get_optuna_ranges

    def get_ranges(self: Any, boosting_type: str) -> dict[str, Any]:
        selected = ranges.get(boosting_type, {}) if isinstance(ranges, dict) else {}
        return dict(selected) if isinstance(selected, dict) else {}

    BoosterWrapper._get_optuna_ranges = get_ranges
    try:
        yield
    finally:
        BoosterWrapper._get_optuna_ranges = original


def main() -> None:
    """Execute the requested autocampaignxfm script with its original CLI arguments."""
    if len(sys.argv) < 2:
        msg = "Usage: autocampaign_entrypoint.py SCRIPT [HYDRA_ARGS ...]"
        raise SystemExit(msg)
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    with windows_filename_compatibility(), configured_optuna_ranges():
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
