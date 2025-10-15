from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from paper_store import PaperStore

logger = logging.getLogger(__name__)

_RAW_DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data")).expanduser()
if not _RAW_DATA_DIR.is_absolute():
    _RAW_DATA_DIR = (Path("/app") / _RAW_DATA_DIR).resolve()
_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

_raw_store_path = os.getenv("PAPER_STORE_PATH")
if _raw_store_path:
    candidate = Path(_raw_store_path)
    if not candidate.is_absolute():
        candidate = (_RAW_DATA_DIR / candidate).resolve()
else:
    candidate = (_RAW_DATA_DIR / "paper_state.json").resolve()

PAPER_STORE_PATH = candidate

ACTIVE_PAPER_STORE: PaperStore | None = None
ACTIVE_LIVE_CLIENT: Any | None = None


class SimBroker:
    """Simula fills y persiste el estado en disco."""

    def __init__(self, store: PaperStore, fee_rate: float = 0.0):
        self.store = store
        self.fee_rate = max(0.0, float(fee_rate))

    def _apply_fill(self, side: str, qty: float, price: float) -> dict[str, Any]:
        state = self.store.load()
        pos_qty = float(state.get("pos_qty", 0.0) or 0.0)
        avg_price = float(state.get("avg_price", 0.0) or 0.0)
        dq = float(qty) if side.upper() == "BUY" else -float(qty)
        fee = abs(float(price) * float(qty)) * self.fee_rate

        updates: dict[str, Any] = {}
        if pos_qty == 0.0 or pos_qty * dq >= 0.0:
            # Abrir o aumentar posición en la misma dirección.
            new_qty = pos_qty + dq
            if abs(new_qty) < 1e-12:
                new_avg = 0.0
            elif pos_qty == 0.0:
                new_avg = float(price)
            else:
                total_notional = abs(pos_qty) * avg_price + abs(dq) * float(price)
                new_avg = total_notional / max(abs(new_qty), 1e-12)
            updates.update({
                "pos_qty": new_qty,
                "avg_price": new_avg,
            })
        else:
            # Cerrando posición parcial o completamente.
            closing_qty = min(abs(pos_qty), abs(dq))
            realized = 0.0
            if closing_qty > 0:
                if pos_qty > 0:
                    realized = (float(price) - avg_price) * closing_qty
                else:
                    realized = (avg_price - float(price)) * closing_qty
            remaining_qty = pos_qty + dq
            updates["pos_qty"] = remaining_qty
            if abs(remaining_qty) < 1e-12:
                updates["avg_price"] = 0.0
            elif abs(abs(dq) - closing_qty) > 1e-12:
                # Revirtió y abrió nueva en dirección contraria.
                updates["avg_price"] = float(price)
            else:
                updates["avg_price"] = avg_price
            updates["realized_pnl"] = state.get("realized_pnl", 0.0) + realized

        if fee > 0:
            updates["fees"] = state.get("fees", 0.0) + fee

        return self.store.save(**updates)

    def place_order(self, side: str, qty: float, price: float | None, **kwargs: Any) -> dict[str, Any]:
        fill_price: float | None = None if price is None else float(price)
        if fill_price is None:
            state = self.store.load()
            fallback = state.get("mark") or state.get("avg_price")
            if fallback is None:
                raise ValueError("SimBroker requiere un precio para simular fills")
            fill_price = float(fallback)

        oid = f"sim-{int(time.time() * 1000)}"
        new_state = self._apply_fill(side, qty, float(fill_price))
        logger.info("PAPER ORDER %s %.6f @ %.2f → FILLED [%s]", side, qty, float(fill_price), oid)
        payload: dict[str, Any] = {
            "orderId": oid,
            "status": "FILLED",
            "price": float(fill_price),
            "executedQty": float(qty),
            "side": str(side).upper(),
            "sim": True,
            "state": new_state,
        }
        symbol = kwargs.get("symbol")
        if symbol:
            payload["symbol"] = symbol
        if "sl" in kwargs:
            payload["sl"] = kwargs["sl"]
        if "tp" in kwargs:
            payload["tp"] = kwargs["tp"]
        return payload


class BinanceBroker:
    """Envuelve al cliente real SOLO en live."""

    def __init__(self, client: Any):
        self.client = client

    def place_order(self, side: str, qty: float, price: float | None, **kwargs: Any) -> Any:
        symbol = kwargs.get("symbol")
        if not symbol:
            raise ValueError("BinanceBroker requiere 'symbol' para enviar la orden.")
        params = {k: v for k, v in kwargs.items() if k not in {"symbol", "sl", "tp"}}
        explicit_type = params.pop("order_type", None)
        reduce_only = params.pop("reduce_only", False)
        if explicit_type is None:
            order_type = "MARKET" if price in (None, 0, "0") else "LIMIT"
        else:
            order_type = str(explicit_type).upper()

        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Orden LIMIT requiere precio")
            params["price"] = float(price)
        else:
            params.pop("price", None)

        if reduce_only:
            params["reduceOnly"] = True

        return self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=qty,
            **params,
        )


def _build_paper_store(start_equity: float) -> PaperStore:
    path = PAPER_STORE_PATH
    if path.is_dir():
        path = path / "paper_state.json"
    return PaperStore(path, start_equity=start_equity)


def build_broker(settings, client_factory: Callable[..., Any]):
    global ACTIVE_PAPER_STORE, ACTIVE_LIVE_CLIENT
    if settings.PAPER:
        ACTIVE_PAPER_STORE = _build_paper_store(settings.start_equity)
        logger.warning(
            "MODO: 🧪 SIMULADO | equity inicial: %.2f USDT | store=%s",
            settings.start_equity,
            ACTIVE_PAPER_STORE.path,
        )
        fee_rate = float(os.getenv("PAPER_FEE_RATE", "0") or 0.0)
        return SimBroker(ACTIVE_PAPER_STORE, fee_rate=fee_rate)

    assert settings.binance_api_key, "Falta BINANCE_API_KEY (modo real)."
    assert settings.binance_api_secret, "Falta BINANCE_API_SECRET (modo real)."
    # Siempre recrear el cliente al cambiar a REAL para evitar credenciales viejas
    ACTIVE_LIVE_CLIENT = None
    client = client_factory(api_key=settings.binance_api_key, secret=settings.binance_api_secret)
    ACTIVE_LIVE_CLIENT = client
    logger.warning("MODO: 🔴 REAL | Binance listo.")
    return BinanceBroker(client)


__all__ = [
    "SimBroker",
    "BinanceBroker",
    "build_broker",
    "ACTIVE_PAPER_STORE",
    "ACTIVE_LIVE_CLIENT",
    "PAPER_STORE_PATH",
]
