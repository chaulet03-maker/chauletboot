from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _resolve_bot_id() -> str:
    try:
        cwd = os.path.abspath(os.getcwd())
    except Exception:
        return "bot"
    base = os.path.basename(cwd) or "bot"
    return base.replace(" ", "_")


BOT_ID = _resolve_bot_id()


def _resolve_base_dir() -> Path:
    raw = os.getenv("APP_ROOT")
    if raw:
        base = Path(raw).expanduser()
        if not base.is_absolute():
            base = (Path.cwd() / base).resolve()
        return base
    return Path("/app")


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    raw = os.getenv("DATA_DIR")
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            base = _resolve_base_dir()
            candidate = (base / candidate).resolve()
    else:
        candidate = _resolve_base_dir() / "data"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _resolve_under_data(raw_path: Optional[str], default_name: str) -> Path:
    data_dir = get_data_dir()
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (data_dir / candidate).resolve()
    else:
        candidate = data_dir / default_name
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


@lru_cache(maxsize=1)
def get_paper_store_path() -> Path:
    import os as _os

    bot_name = _os.path.basename(_os.getcwd())
    return _resolve_under_data(
        _os.getenv("PAPER_STORE_PATH"), f"paper_store_{bot_name}.json"
    )


@lru_cache(maxsize=1)
def get_live_store_path() -> Path:
    return _resolve_under_data(
        os.getenv("LIVE_STORE_PATH"), f"live_bot_position_{BOT_ID}.json"
    )


@lru_cache(maxsize=1)
def get_runtime_dir() -> Path:
    path = get_data_dir() / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_runtime_path(name: str) -> Path:
    return (get_runtime_dir() / name).resolve()
