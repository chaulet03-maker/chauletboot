from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

from paper_store import DEFAULT_STATE, PaperStore
from bot.storage.order_store import OrderStore


class PaperAccount:
    """Pequeña fachada sobre ``PaperStore`` para exponer equity/posiciones."""

    def __init__(self, start_equity: float = 1000.0):
        self.store = PaperStore(start_equity=start_equity)

    def get_state(self) -> Dict[str, Any]:
        try:
            return self.store.get_state()
        except Exception:
            return dict(DEFAULT_STATE)

    def get_equity(self) -> float:
        state = self.get_state()
        try:
            eq = float(state.get("equity") or 0.0)
        except Exception:
            eq = 0.0
        if eq <= 0:
            try:
                eq = float(self.store.start_equity)
            except Exception:
                eq = 1000.0
        return eq

    def open_position(
        self,
        *,
        qty: float,
        side: str,
        entry: float,
        leverage: float = 1.0,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
        symbol: Optional[str] = None,
        mark: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            current_state = self.get_state()
            sym = symbol or str(current_state.get("symbol") or "")
        except Exception:
            sym = symbol or ""

        try:
            entry_price = float(entry)
            qty_float = float(qty)
            pos = {
                "symbol": sym,
                "side": side.upper(),
                "qty": qty_float,
                "entry": float(entry_price),
                "timestamp": time.time(),
                "mode": "SIM",
            }

            store = OrderStore()
            store.save_position(pos)

            return self.store.set_position(
                symbol=sym,
                qty=qty_float,
                side=side,
                entry=entry_price,
                leverage=float(leverage),
                tp=tp if tp is None else float(tp),
                sl=sl if sl is None else float(sl),
                mark=mark if mark is None else float(mark),
            )
        except Exception:
            return self.get_state()


@dataclass
class Fill:
    price: float
    qty: float
    side: str
    ts: float


class PaperExchange:
    def __init__(self, fees: dict, slippage_bps: int = 5):
        self.fees = fees or {"taker": 0.0002, "maker": 0.0002}
        self.slippage_bps = int(slippage_bps)
        self._stops = {}
        self._takes = {}

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            self._lev = getattr(self, "_lev", {})
            self._lev[symbol] = leverage
        except Exception:
            pass
        return True

    async def market_order(self, symbol: str, side: str, qty: float, ref_price: float):
        slip = self.slippage_bps / 10000.0
        price = ref_price * (1 + slip) if side == "long" else ref_price * (1 - slip)
        fee = abs(price * qty) * self.fees.get("taker", 0.0002)
        return Fill(price=price, qty=qty, side=side, ts=time.time()), fee

    # --- MÉTODOS INTERNOS DE SIMULACIÓN (añadidos para completitud) ---
    def _place_stop_loss(self, symbol, side, qty, sl_price):
        """Simula el almacenamiento de una orden Stop Loss."""
        # En una simulación real, guardarías esto para verificarlo en cada tick.
        print(f"PAPER: Registrando SL para {symbol} ({side} {qty}) @ {sl_price}")
        self._stops[symbol] = {"side": side, "price": sl_price}

    def _place_take_profit(self, symbol, side, qty, tp_price):
        """Simula el almacenamiento de una orden Take Profit."""
        # En una simulación real, guardarías esto para verificarlo en cada tick.
        print(f"PAPER: Registrando TP para {symbol} ({side} {qty}) @ {tp_price}")
        self._takes[symbol] = {"side": side, "price": tp_price}

    async def place_protections(self, symbol, side, qty, sl, tp1=None, tp2=None):
        """PAPER: coloca un único SL y un único TP (sin parciales)."""

        tp_final = tp2 if tp2 is not None else tp1
        self._place_stop_loss(symbol, side, qty, sl)
        if tp_final is not None:
            self._place_take_profit(symbol, side, qty, tp_final)
        return True


__all__ = ["PaperAccount", "PaperExchange", "Fill"]
