from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from bot.runtime_state import get_mode as runtime_get_mode

STATE_PATH = os.getenv("BOT_STATE_PATH", "./state.json")


@dataclass
class Position:
    id: str
    symbol: str
    side: str  # "LONG" | "SHORT"
    qty: float
    entry_price: float
    leverage: float
    tp: Optional[float] = None
    sl: Optional[float] = None
    opened_at: float = time.time()
    closed_at: Optional[float] = None
    status: str = "open"  # "open" | "closed"
    realized_pnl: float = 0.0
    mode: str = "paper"  # "paper" | "live"
    fees: float = 0.0
    gross_pnl: float = 0.0


def _runtime_is_paper() -> bool:
    try:
        return (runtime_get_mode() or "paper").lower() not in {"real", "live"}
    except Exception:
        return True


def _atomic_write(path: str, data: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"open_positions": {}, "closed_positions": []}
    with open(STATE_PATH, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def save_state(state: Dict[str, Any]) -> None:
    payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    _atomic_write(STATE_PATH, payload.encode("utf-8"))


def create_position(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    leverage: float,
    tp: Optional[float] = None,
    sl: Optional[float] = None,
    mode: str = "paper",
    fee: float = 0.0,
) -> Position:
    return Position(
        id=str(uuid.uuid4()),
        symbol=symbol,
        side=side.upper(),
        qty=float(qty),
        entry_price=float(entry_price),
        leverage=float(leverage),
        tp=tp,
        sl=sl,
        mode=mode,
        fees=float(fee or 0.0),
    )


def persist_open(pos: Position, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state_obj = state or load_state()
    state_obj.setdefault("open_positions", {})[pos.symbol] = asdict(pos)
    state_obj.setdefault("closed_positions", [])
    save_state(state_obj)
    return state_obj


def persist_close(
    symbol: str,
    exit_price: float,
    realized_pnl: float,
    *,
    fee: float = 0.0,
    gross_pnl: Optional[float] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state_obj = state or load_state()
    open_positions = state_obj.setdefault("open_positions", {})
    pos = open_positions.pop(symbol, None)
    if pos is None and "/" in symbol:
        pos = open_positions.pop(symbol.replace("/", ""), None)
        symbol = symbol.replace("/", "") if pos is not None else symbol
    if pos:
        pos["closed_at"] = time.time()
        pos["status"] = "closed"
        pos["exit_price"] = float(exit_price)
        existing_fees = float(pos.get("fees") or 0.0)
        total_fees = existing_fees + float(fee or 0.0)
        pos["fees"] = total_fees
        if gross_pnl is not None:
            pos["gross_pnl"] = float(gross_pnl)
        pos["realized_pnl"] = float(realized_pnl)
        state_obj.setdefault("closed_positions", []).append(pos)
        save_state(state_obj)
    return state_obj


def update_open_position(symbol: str, **changes: Any) -> bool:
    """Actualiza campos simples (tp/sl/lev) de la posición abierta."""

    if not changes:
        return False

    state = load_state()
    open_positions = state.setdefault("open_positions", {})

    candidates = []
    if symbol:
        candidates.append(symbol)
        normalized = symbol.replace("/", "")
        if normalized != symbol:
            candidates.append(normalized)

    for key in candidates:
        pos = open_positions.get(key)
        if pos is None and "/" in key:
            alt_key = key.replace("/", "")
            pos = open_positions.get(alt_key)
            key = alt_key if pos is not None else key
        if pos is None:
            continue
        updated = dict(pos)
        for field, value in changes.items():
            if value is None:
                updated.pop(field, None)
                continue
            try:
                updated[field] = float(value)
            except (TypeError, ValueError):
                updated[field] = value
        open_positions[key] = updated
        save_state(state)
        return True
    return False


def broker_get_open_position(symbol: str) -> Optional[Dict[str, Any]]:
    """Obtiene una posición abierta directamente desde el broker si existe."""

    try:
        from trading import POSITION_SERVICE, ensure_initialized
    except Exception:
        return None

    try:
        ensure_initialized()
    except Exception:
        return None

    svc = POSITION_SERVICE
    if svc is None:
        return None

    try:
        status = svc.get_status()
    except Exception:
        return None

    if not status or str(status.get("side", "FLAT")).upper() == "FLAT":
        return None

    qty = float(status.get("qty") or status.get("pos_qty") or 0.0)
    if qty == 0:
        return None

    side = str(status.get("side", "FLAT")).upper()
    amount = qty if side == "LONG" else -qty
    entry_price = status.get("entry_price") or status.get("avg_price")
    leverage = status.get("leverage") or 1.0

    try:
        entry_price_f = float(entry_price) if entry_price is not None else 0.0
    except Exception:
        entry_price_f = 0.0
    try:
        leverage_f = float(leverage) if leverage not in (None, "") else 1.0
    except Exception:
        leverage_f = 1.0

    return {
        "amount": amount,
        "entry_price": entry_price_f,
        "leverage": leverage_f if leverage_f > 0 else 1.0,
    }


def on_open_filled(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    lev: float,
    tp: Optional[float] = None,
    sl: Optional[float] = None,
    mode: str = "paper",
    fee: float = 0.0,
) -> None:
    # En REAL no persistimos posición del BOT en disco
    if not _runtime_is_paper():
        return
    pos = create_position(symbol, side, qty, price, lev, tp, sl, mode, fee)
    persist_open(pos)


def on_close_filled(symbol: str, exit_price: float, fee: float = 0.0) -> None:
    # En REAL no persistimos posición del BOT en disco
    if not _runtime_is_paper():
        return
    state = load_state()
    open_positions = state.get("open_positions", {})
    pos = open_positions.get(symbol)
    key_used = symbol
    if pos is None and "/" in symbol:
        alt = symbol.replace("/", "")
        pos = open_positions.get(alt)
        if pos is not None:
            key_used = alt
    if pos is None and symbol.replace("/", "") in open_positions:
        key_used = symbol.replace("/", "")
        pos = open_positions.get(key_used)
    if not pos:
        return
    side = pos.get("side", "LONG").upper()
    sign = 1 if side == "LONG" else -1
    try:
        entry = float(pos.get("entry_price", 0.0) or 0.0)
    except Exception:
        entry = 0.0
    try:
        qty = float(pos.get("qty", 0.0) or 0.0)
    except Exception:
        qty = 0.0
    try:
        leverage = float(pos.get("leverage", 1.0) or 1.0)
    except Exception:
        leverage = 1.0
    existing_fees = float(pos.get("fees") or 0.0)
    gross = (float(exit_price) - entry) * sign * qty
    total_fees = existing_fees + float(fee or 0.0)
    realized = gross - total_fees
    persist_close(
        key_used,
        exit_price,
        realized,
        fee=total_fees,
        gross_pnl=gross,
        state=state,
    )


def position_status(symbol: str, mode: str) -> str:
    state = load_state()
    pos = state.get("open_positions", {}).get(symbol)
    if pos:
        return (
            f"Posición ABIERTA: {pos['side']} {symbol} @ {pos['entry_price']} "
            f"x{pos['leverage']}, qty {pos['qty']}, tp: {pos.get('tp')}, sl: {pos.get('sl')}"
        )

    if mode == "live":
        broker_pos = broker_get_open_position(symbol)
        if broker_pos and abs(float(broker_pos.get("amount", 0.0))) > 0:
            reconstructed = create_position(
                symbol=symbol,
                side="LONG" if float(broker_pos["amount"]) > 0 else "SHORT",
                qty=abs(float(broker_pos["amount"])),
                entry_price=float(broker_pos.get("entry_price", 0.0) or 0.0),
                leverage=float(broker_pos.get("leverage", 1.0) or 1.0),
                mode="live",
            )
            persist_open(reconstructed)
            pos = load_state().get("open_positions", {}).get(symbol)
            if pos:
                return (
                    f"Posición ABIERTA (sincronizada): {pos['side']} {symbol} @ {pos['entry_price']} "
                    f"x{pos['leverage']}, qty {pos['qty']}"
                )
    return "SIN POSICIÓN"


__all__ = [
    "Position",
    "STATE_PATH",
    "load_state",
    "save_state",
    "create_position",
    "persist_open",
    "persist_close",
    "update_open_position",
    "on_open_filled",
    "on_close_filled",
    "position_status",
    "broker_get_open_position",
]
