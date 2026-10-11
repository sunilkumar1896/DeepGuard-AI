"""
configs/__init__.py
-------------------
Configuration loader for the Deepfake Detection Platform.
Loads config.yaml and exposes a typed config object.
"""
import os
from pathlib import Path
import yaml
from typing import Any, Dict


class DotDict(dict):
    """Dictionary subclass that allows dot-notation access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key: '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str | Path | None = None) -> DotDict:
    """
    Load the YAML configuration file and return a DotDict.

    Args:
        config_path: Path to config.yaml. Defaults to
                     <project_root>/configs/config.yaml.

    Returns:
        DotDict with nested dot-notation access.
    """
    if config_path is None:
        # Resolve relative to this file's grandparent (project root)
        config_path = Path(__file__).parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    return _to_dotdict(raw)


def _to_dotdict(d: Any) -> Any:
    """Recursively convert dicts to DotDicts."""
    if isinstance(d, dict):
        return DotDict({k: _to_dotdict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_dotdict(i) for i in d]
    return d


# ── Singleton config instance ──────────────────────────────────
# Import and use `cfg` anywhere in the project:
#   from configs import cfg
cfg = load_config()
