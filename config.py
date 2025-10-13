from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _candidate_paths(path: Optional[str]) -> list[Path]:
    candidates: list[str] = []
    env_path = os.getenv("CONFIG_PATH")
    if path:
        candidates.append(path)
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "config.yaml",
        "config.yml",
        os.path.join("config", "config.yaml"),
        os.path.join("config", "config.yml"),
    ])
    seen: set[Path] = set()
    out: list[Path] = []
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand).expanduser().resolve()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _resolve_config_path(path: Optional[str] = None) -> Path:
    for candidate in _candidate_paths(path):
        if candidate.exists():
            return candidate
    # fallback to first candidate even if missing
    candidates = _candidate_paths(path)
    if candidates:
        return candidates[0]
    return Path("config.yaml").resolve()


def load_raw_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = _resolve_config_path(path)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    if not isinstance(data, dict):
        raise TypeError("El archivo de configuración debe contener un objeto YAML (dict).")
    return data


@dataclass
class Settings:
    trading_mode: str
    start_equity: float
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None

    @property
    def PAPER(self) -> bool:
        return str(self.trading_mode).lower() in {"simulado", "paper", "test", "demo"}

    @property
    def LIVE(self) -> bool:
        return not self.PAPER


def load_settings(path: Optional[str] = None) -> Settings:
    cfg = load_raw_config(path)
    trading_mode = str(cfg.get("trading_mode", "simulado"))
    start_equity = float(cfg.get("start_equity", 1000.0) or 0.0)
    binance_cfg = cfg.get("binance", {})
    if not isinstance(binance_cfg, dict):
        binance_cfg = {}
    api_key = (
        os.getenv("BINANCE_API_KEY")
        or os.getenv("BINANCE_API_KEY_REAL")
        or os.getenv("BINANCE_API_KEY_TEST")
        or binance_cfg.get("api_key")
    )
    api_secret = (
        os.getenv("BINANCE_API_SECRET")
        or os.getenv("BINANCE_API_SECRET_REAL")
        or os.getenv("BINANCE_API_SECRET_TEST")
        or binance_cfg.get("api_secret")
    )
    settings = Settings(
        trading_mode=trading_mode,
        start_equity=start_equity,
        binance_api_key=api_key,
        binance_api_secret=api_secret,
    )
    output_timezone = cfg.get("output_timezone")
    setattr(settings, "output_timezone", output_timezone)
    return settings


def _nested_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


RAW_CONFIG: Dict[str, Any] = load_raw_config()
S: Settings = load_settings()


def get_telegram_token(default: Optional[str] = None) -> Optional[str]:
    env = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if env:
        return env
    token = _nested_get(RAW_CONFIG, "telegram", "token")
    if token:
        return str(token)
    return default


def get_telegram_chat_id(default: Optional[str] = None) -> Optional[str]:
    env = os.getenv("TELEGRAM_CHAT_ID")
    if env:
        return env
    chat_id = _nested_get(RAW_CONFIG, "telegram", "chat_id")
    if chat_id:
        return str(chat_id)
    return default


__all__ = [
    "Settings",
    "S",
    "RAW_CONFIG",
    "load_raw_config",
    "load_settings",
    "get_telegram_token",
    "get_telegram_chat_id",
]
