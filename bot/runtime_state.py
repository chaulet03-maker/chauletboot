from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


STATE_PATH = "data/runtime/state.json"
DEFAULT_SL_PRICE_PCT = -10.0
DEFAULT_TP_PRICE_PCT = 10.0
os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)


def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"mode": "paper", "equity_sim": 0.0, "updated": int(time.time())}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("runtime state must be a dict")
    except Exception:
        data = {"mode": "paper", "equity_sim": 0.0, "updated": int(time.time())}
    data.setdefault("mode", "paper")
    data.setdefault("equity_sim", 0.0)
    data.setdefault("updated", int(time.time()))
    return data


def _save_state(payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated"] = int(time.time())
    tmp_path = f"{STATE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp_path, STATE_PATH)


def get_mode() -> str:
    """Return the persisted trading mode (`paper` | `live`)."""

    mode = str(_load_state().get("mode", "paper")).lower()
    return "live" if mode == "live" else "paper"


def set_mode(mode: str) -> None:
    """Persist the trading mode (`paper` | `live`)."""

    state = _load_state()
    state["mode"] = "live" if str(mode).lower() in {"live", "real"} else "paper"
    _save_state(state)


def get_equity_sim() -> float:
    """Return the simulated equity base used for manual actions."""

    try:
        return float(_load_state().get("equity_sim", 0.0) or 0.0)
    except Exception:
        return 0.0


def set_equity_sim(value: float) -> None:
    """Persist the simulated equity base used for manual actions."""

    state = _load_state()
    try:
        state["equity_sim"] = float(value)
    except Exception:
        state["equity_sim"] = 0.0
    _save_state(state)


def _symbol_key(symbol: str) -> str:
    return str(symbol or "BTCUSDT").replace("/", "").upper()


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_protection_defaults(symbol: str) -> Dict[str, Any]:
    """Return persisted TP/SL defaults for the given symbol (if any)."""

    state = _load_state()
    protections = state.get("protections", {})
    if not isinstance(protections, dict):
        return {
            "sl_last_kind": "pct",
            "sl_pct_equity": DEFAULT_SL_PRICE_PCT,
            "tp_last_kind": "pct",
            "tp_pct_equity": DEFAULT_TP_PRICE_PCT,
        }
    entry_raw = protections.get(_symbol_key(symbol), {})
    entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}

    sl_kind = str(entry.get("sl_last_kind") or "").lower()
    if sl_kind not in {"pct", "price"}:
        sl_kind = "pct"
    entry["sl_last_kind"] = "price" if sl_kind == "price" else "pct"
    if entry["sl_last_kind"] == "pct":
        sl_pct = _coerce_float(entry.get("sl_pct_equity"))
        entry["sl_pct_equity"] = sl_pct if sl_pct is not None else DEFAULT_SL_PRICE_PCT
    else:
        sl_price = _coerce_float(entry.get("sl_price"))
        if sl_price is not None:
            entry["sl_price"] = sl_price

    tp_kind = str(entry.get("tp_last_kind") or "").lower()
    if tp_kind not in {"pct", "price"}:
        tp_kind = "pct"
    entry["tp_last_kind"] = "price" if tp_kind == "price" else "pct"
    if entry["tp_last_kind"] == "pct":
        tp_pct = _coerce_float(entry.get("tp_pct_equity"))
        entry["tp_pct_equity"] = tp_pct if tp_pct is not None else DEFAULT_TP_PRICE_PCT
    else:
        tp_price = _coerce_float(entry.get("tp_price"))
        if tp_price is not None:
            entry["tp_price"] = tp_price

    return entry


def update_protection_defaults(symbol: str, **changes: Any) -> Dict[str, Any]:
    """Persist TP/SL defaults for the next manual trade."""

    state = _load_state()
    protections = state.setdefault("protections", {})
    if not isinstance(protections, dict):
        protections = {}
        state["protections"] = protections

    key = _symbol_key(symbol)
    current = dict(protections.get(key) or {})

    mutated = False
    for field, value in changes.items():
        if value is None:
            if field in current:
                current.pop(field, None)
                mutated = True
            continue
        try:
            current[field] = float(value) if isinstance(value, (int, float)) else value
        except Exception:
            current[field] = value
        mutated = True

    if current:
        protections[key] = current
    elif key in protections:
        protections.pop(key, None)
        mutated = True

    if mutated:
        _save_state(state)
    return dict(protections.get(key, {}))


__all__ = [
    "get_mode",
    "set_mode",
    "get_equity_sim",
    "set_equity_sim",
    "get_protection_defaults",
    "update_protection_defaults",
]
