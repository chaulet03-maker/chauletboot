from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict

from paths import get_data_dir, get_paper_store_path

log = logging.getLogger(__name__)

DEFAULT_STATE: Dict[str, Any] = {
    "equity": None,
    "pos_qty": 0.0,
    "avg_price": 0.0,
    "realized_pnl": 0.0,
    "fees": 0.0,
    "mark": 0.0,
    "tp": None,
    "sl": None,
    "leverage": 1,
    "updated": 0,
}


class PaperStore:
    """Thread-safe JSON store to persist simulated trading state."""

    def __init__(self, path: str | Path | None = None, start_equity: float = 1000.0):
        self.start_equity = float(start_equity)
        self.lock = RLock()

        data_dir = get_data_dir()
        if path is None:
            candidate = get_paper_store_path()
        else:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = (data_dir / candidate).resolve()

        self.path = candidate
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            initial_state = dict(DEFAULT_STATE)
            initial_state["equity"] = self.start_equity
            self._write(initial_state)
            log.info("PaperStore creado: %s (equity=%.2f)", self.path, self.start_equity)

        self.state = self._safe_read()
        if self.state.get("equity") is None:
            self.state["equity"] = self.start_equity
            self._write(self.state)
        log.info(
            "PaperStore cargado: %s (pos_qty=%.6f, avg=%.2f, updated=%s)",
            self.path,
            float(self.state.get("pos_qty", 0.0) or 0.0),
            float(self.state.get("avg_price", 0.0) or 0.0),
            self.state.get("updated"),
        )

    # --- private helpers -------------------------------------------------

    def _safe_read(self) -> Dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            data = {}
        except json.JSONDecodeError:
            data = {}
        state = dict(DEFAULT_STATE)
        for key, value in (data or {}).items():
            if key in DEFAULT_STATE or key in {"tp", "sl"}:
                state[key] = value
        if state.get("equity") is None:
            state["equity"] = self.start_equity
        return state

    def _write(self, state: Dict[str, Any]) -> None:
        state = dict(state)
        state["updated"] = int(time.time())
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
        tmp_path.replace(self.path)

    # --- public api ------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        with self.lock:
            self.state = self._safe_read()
            return dict(self.state)

    def save(self, **changes: Any) -> Dict[str, Any]:
        with self.lock:
            state = dict(self.state)
            state.update(changes)
            if state.get("equity") is None:
                state["equity"] = self.start_equity
            self._write(state)
            self.state = state
            return dict(state)


__all__ = ["PaperStore", "DEFAULT_STATE"]
