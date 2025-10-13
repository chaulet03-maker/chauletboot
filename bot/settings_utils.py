from __future__ import annotations

import os
import re
from typing import Any

import yaml

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


def read_config_raw(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception:
        return {}


def _norm_key(key: str) -> str:
    key = key.replace("-", "_").strip()
    key = re.sub(r"\s+", "_", key)
    return key


def get_val(S: Any, raw_cfg: dict, *keys, default=None):
    """Busca en S como atributo y en el YAML como clave normalizando nombres."""

    for key in keys:
        norm_key = _norm_key(key)
        if hasattr(S, norm_key):
            value = getattr(S, norm_key)
            if value is not None:
                return value
        if norm_key in raw_cfg and raw_cfg[norm_key] is not None:
            return raw_cfg[norm_key]
    return default
