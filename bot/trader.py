import asyncio
import logging
import math
from typing import Any, Dict, Optional

from config import S
import trading


class Trader:
    def __init__(self, cfg):
        self.config = cfg
        trading.ensure_initialized()
        default_balance = float(self.config.get('balance', S.start_equity))
        if S.PAPER:
            try:
                default_balance = float(getattr(trading.BROKER, "equity", default_balance))
            except Exception:
                pass
        self._balance = default_balance
        self._open_position: Optional[Dict[str, Any]] = None
        self._last_mode = "paper" if S.PAPER else "real"

    def reset_caches(self):
        """Limpiar caches de balance/posición al cambiar de modo."""
        self._balance = float(self.config.get('balance', S.start_equity))
        self._open_position = None
        self._last_mode = "paper" if S.PAPER else "real"

    def _ensure_mode_consistency(self):
        curr = "paper" if S.PAPER else "real"
        if curr != getattr(self, "_last_mode", curr):
            self.reset_caches()

    async def get_balance(self, exchange=None) -> float:
        """Devuelve el balance actual de la cuenta."""
        self._ensure_mode_consistency()
        if S.PAPER:
            try:
                self._balance = float(self.equity())
                return self._balance
            except Exception:
                logging.debug("No se pudo refrescar equity en modo paper.", exc_info=True)

        if exchange and getattr(exchange, 'client', None):
            try:
                balance = await asyncio.to_thread(exchange.client.fetch_balance)
                usdt_info = balance.get('USDT') if isinstance(balance, dict) else None
                if isinstance(usdt_info, dict):
                    total = usdt_info.get('total') or usdt_info.get('free')
                    if total is not None:
                        self._balance = float(total)
            except Exception as exc:
                logging.warning("No se pudo actualizar el balance desde el exchange: %s", exc)
        return self._balance

    def equity(self, force_refresh: bool = False) -> float:
        """Devuelve el equity actual (USDT) usando las fuentes disponibles."""
        self._ensure_mode_consistency()
        trading.ensure_initialized()

        if not force_refresh and math.isfinite(self._balance):
            cached = float(self._balance)
        else:
            cached = 0.0

        equity: Optional[float] = None

        service = getattr(trading, "POSITION_SERVICE", None)
        if service is not None:
            try:
                status = service.get_status() or {}
                raw_equity = status.get("equity")
                if raw_equity is not None:
                    equity = float(raw_equity)
            except Exception:
                logging.debug("No se pudo obtener equity desde PositionService.", exc_info=True)

        if equity is None and S.PAPER:
            store = None
            broker = getattr(trading, "BROKER", None)
            if broker is not None:
                store = getattr(broker, "store", None)
            if store is None:
                store = getattr(trading, "ACTIVE_PAPER_STORE", None)
            if store is not None:
                try:
                    state = store.load()
                    base_equity = float(state.get("equity") or 0.0)
                    realized = float(state.get("realized_pnl") or 0.0)
                    fees = float(state.get("fees") or 0.0)
                    mark = float(state.get("mark") or 0.0)
                    pos_qty = float(state.get("pos_qty") or 0.0)
                    avg_price = float(state.get("avg_price") or 0.0)
                    unreal = 0.0
                    if pos_qty != 0.0 and avg_price > 0.0 and mark > 0.0:
                        delta = mark - avg_price
                        if pos_qty < 0:
                            delta = -delta
                        unreal = abs(pos_qty) * delta
                    equity = base_equity + realized + unreal - fees
                except Exception:
                    logging.debug("No se pudo calcular equity desde el PaperStore.", exc_info=True)

        if equity is None:
            equity = cached

        if not math.isfinite(equity):
            equity = 0.0

        self._balance = float(equity)
        return self._balance

    async def check_open_position(self, exchange=None) -> Optional[Dict[str, Any]]:
        """Devuelve la posición abierta (si la hay) y cachea el resultado."""
        self._ensure_mode_consistency()

        if self._open_position:
            return self._open_position

        # 1) PAPER: leer del PositionService (persistente)
        if S.PAPER and trading.POSITION_SERVICE is not None:
            try:
                st = trading.POSITION_SERVICE.get_status()
                side = (st.get("side") or "FLAT").upper()
                if side != "FLAT":
                    self._open_position = {
                        "symbol": st.get("symbol", self.config.get("symbol", "BTC/USDT")),
                        "side": side,
                        "contracts": float(st.get("qty") or st.get("size") or 0.0),
                        "entryPrice": float(st.get("entry_price") or 0.0),
                        "markPrice": float(st.get("mark") or 0.0),
                    }
                    return self._open_position
                self._open_position = None
            except Exception as exc:
                logging.debug("PAPER check_open_position fallo: %s", exc)
            return self._open_position

        # 2) LIVE: si tenemos exchange, tratemos de obtener la posición real
        if exchange and hasattr(exchange, "fetch_positions"):
            try:
                positions = await exchange.fetch_positions(self.config.get('symbol', 'BTC/USDT'))
                if positions:
                    self._open_position = positions[0]
                    return self._open_position
            except Exception as exc:
                logging.debug("No se pudo obtener la posición desde el exchange: %s", exc)
        return self._open_position

    async def set_position(self, position_data: Optional[Dict[str, Any]]) -> None:
        """Actualiza el estado de la posición almacenada."""
        self._open_position = position_data
        logging.info("Nuevo estado de posición: %s", position_data)
