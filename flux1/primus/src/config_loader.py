"""
Lightweight YAML config loader with shell-style ${VAR:default} expansion.

Mirrors the Primus config format used by GPT-OSS so that the YAML is the
single source of truth for all training parameters.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_PATTERN = re.compile(r"\$\{(\w+):([^}]*)\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR:default} patterns using environment variables."""
    def _replace(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default)
    return _ENV_PATTERN.sub(_replace, value)


def _walk_and_expand(obj: Any) -> Any:
    """Recursively expand env vars in a parsed YAML tree."""
    if isinstance(obj, str):
        expanded = _expand_env(obj)
        if expanded.lower() in ("true", "false"):
            return expanded.lower() == "true"
        if expanded.lower() == "null":
            return None
        try:
            return int(expanded)
        except ValueError:
            pass
        try:
            return float(expanded)
        except ValueError:
            pass
        return expanded
    if isinstance(obj, dict):
        return {k: _walk_and_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_expand(v) for v in obj]
    return obj


def load_flux_config(config_path: str) -> Dict[str, Any]:
    """Load a Flux YAML config and return the expanded overrides dict.

    Returns a flat dict of all keys under ``modules.diffusion_trainer.overrides``
    with ``${VAR:default}`` patterns resolved from environment variables.
    Top-level keys (``exp_name``, ``workspace``) are also included.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    expanded = _walk_and_expand(raw)

    result: Dict[str, Any] = {}

    for key in ("work_group", "user_name", "exp_name", "workspace"):
        if key in expanded:
            result[key] = expanded[key]

    trainer = expanded.get("modules", {}).get("diffusion_trainer", {})
    overrides = trainer.get("overrides", {})
    result.update(overrides)

    return result
