"""Simple TTL-based snapshot cache for Telegram status responses."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict


_DEFAULT_SNAPSHOT: Dict[str, Any] = {
    "mode_display": "…",
    "mode_human": "…",
    "symbol": "…",
    "ccxt_auth": None,
    "price": "—",
    "price_value": None,
    "equity": None,
    "equity_usdt": "—",
    "funding_rate": None,
    "funding_text": "—",
    "position": {
        "has_position": False,
        "symbol": None,
        "side": "FLAT",
        "qty": 0.0,
        "entry_price": None,
        "mark_price": None,
        "pnl": None,
        "roe": None,
        "leverage": None,
    },
    "position_text": "No hay posiciones abiertas.",
    "collected_at": None,
}


class StateCache:
    """Stores the latest status snapshot with a configurable TTL."""

    def __init__(self, ttl: float = 5.0) -> None:
        self.ttl = float(ttl)
        self._last: Dict[str, Any] | None = None
        self._ts: float = 0.0
        self._lock = threading.Lock()

    def put(self, snapshot: Dict[str, Any]) -> None:
        """Save a snapshot copy and update the timestamp."""

        if not isinstance(snapshot, dict):
            return
        data = deepcopy(snapshot)
        ts = float(data.get("collected_at") or time.time())
        data["collected_at"] = ts
        with self._lock:
            self._last = data
            self._ts = ts

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Return the last snapshot or a default placeholder copy."""

        with self._lock:
            current = deepcopy(self._last) if isinstance(self._last, dict) else None
        if current is None:
            return deepcopy(_DEFAULT_SNAPSHOT)
        # Ensure we always hand back a dictionary with the expected keys
        merged = deepcopy(_DEFAULT_SNAPSHOT)
        merged.update({k: v for k, v in current.items() if k != "position"})
        if isinstance(current.get("position"), dict):
            merged_pos = deepcopy(_DEFAULT_SNAPSHOT["position"])
            merged_pos.update(current["position"])
            merged["position"] = merged_pos
        return merged

    def is_stale(self) -> bool:
        """Return True when the cached value is older than the TTL."""

        if self._ts <= 0:
            return True
        return (time.time() - self._ts) > self.ttl

