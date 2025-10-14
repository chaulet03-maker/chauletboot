import asyncio
import logging
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

    async def get_balance(self, exchange=None) -> float:
        """Devuelve el balance actual de la cuenta."""
        if S.PAPER:
            try:
                self._balance = float(getattr(trading.BROKER, "equity", self._balance))
                return self._balance
            except Exception:
                pass

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

    async def check_open_position(self, exchange=None) -> Optional[Dict[str, Any]]:
        """Devuelve la posición abierta (si la hay) y cachea el resultado."""

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
