from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand(value: str, ctx: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in ctx:
            return str(ctx[key])
        return os.environ.get(key, match.group(0))

    return _ENV_PATTERN.sub(repl, value)


def _resolve(obj: Any, ctx: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            resolved = _resolve(v, ctx)
            out[k] = resolved
            if isinstance(resolved, str) and not resolved.startswith("${"):
                ctx[k] = resolved
        return out
    if isinstance(obj, list):
        return [_resolve(v, ctx) for v in obj]
    if isinstance(obj, str):
        return _expand(obj, ctx)
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    root = os.environ.get("CODE_JEPA_ROOT")
    ctx: dict[str, Any] = {}
    if root:
        ctx["storage_root"] = root
        raw["storage_root"] = root

    cfg = _resolve(raw, ctx)
    for key in ("data_dir", "cache_dir", "models_dir", "runs_dir"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    return cfg
