from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from paths import get_runtime_path


def _resolve_bot_id() -> str:
    """Deriva un identificador estable para el bot según la carpeta actual."""

    try:
        cwd = os.path.abspath(os.getcwd())
    except Exception:
        return "bot"
    base = os.path.basename(cwd) or "bot"
    return base.replace(" ", "_")


BOT_ID = _resolve_bot_id()
STATE_PATH = str(get_runtime_path(f"state_{BOT_ID}.json"))


def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)


def get_mode(default: str = "simulado") -> str:
    st = _load_state()
    mode = str(st.get("mode") or default).lower()
    if mode.lower() in ("paper", "simulated", "simul", "simulado"):
        return "simulado"
    return "real" if mode in {"real", "live"} else "simulado"


def set_mode(mode: str) -> None:
    st = _load_state()
    st["mode"] = "real" if str(mode).lower() in {"real", "live"} else "simulado"
    _save_state(st)


def get_equity_sim() -> float:
    try:
        return float(_load_state().get("equity_sim", 0.0) or 0.0)
    except Exception:
        return 0.0


def set_equity_sim(value: float) -> None:
    st = _load_state()
    try:
        st["equity_sim"] = float(value)
    except Exception:
        st["equity_sim"] = 0.0
    _save_state(st)


def get_protection_defaults(symbol: Optional[str] = None) -> Dict[str, Any]:
    st = _load_state()
    prot = st.get("protections") or {}
    if symbol:
        return dict(prot.get(str(symbol), {}))
    return dict(prot)


def update_protection_defaults(symbol: str, **kwargs: Any) -> Dict[str, Any]:
    st = _load_state()
    prot = st.get("protections") or {}
    sym = str(symbol)
    data = dict(prot.get(sym, {}))
    for k in ("SL", "TP", "TP_PCT", "SL_PCT"):
        if k in kwargs and kwargs[k] is not None:
            try:
                data[k] = float(kwargs[k])
            except Exception:
                pass
    prot[sym] = data
    st["protections"] = prot
    _save_state(st)
    return data
