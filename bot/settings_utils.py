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


def get_val(settings: Any, config: dict, *keys, default: Any = None) -> Any:
    """
    Intenta obtener un valor usando una lista de posibles nombres de clave,
    buscando en el objeto Settings (S) y luego en el diccionario de configuración (config).
    Soporta rutas anidadas usando notación de puntos (ej: 'order_sizing.default_pct').
    """

    for key in keys:
        # 1. Buscar en el objeto Settings (S) cuando no es ruta anidada
        if "." not in key:
            attr_name = _norm_key(key)
            if hasattr(settings, attr_name):
                value = getattr(settings, attr_name)
                if value is not None:
                    return value

        # 2. Buscar en el diccionario de configuración (config)
        path = key.split(".")
        current = config
        found = True
        for part in path:
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                    continue

                norm_part = _norm_key(part)
                if norm_part in current:
                    current = current[norm_part]
                    continue

            found = False
            break

        if found and current is not None:
            return current

    return default
