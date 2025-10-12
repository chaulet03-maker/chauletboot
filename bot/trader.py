import asyncio
import logging
from typing import Any, Dict, Optional

from config import S
from trading import BROKER, POSITION_SERVICE


class Trader:
    def __init__(self, cfg):
        self.config = cfg
        default_balance = float(self.config.get('balance', S.start_equity))
        if S.PAPER:
            try:
                default_balance = float(getattr(BROKER, "equity", default_balance))
            except Exception:
                pass
        self._balance = default_balance
        self._open_position: Optional[Dict[str, Any]] = None

    async def get_balance(self, exchange=None) -> float:
        """Devuelve el balance actual de la cuenta."""
        if S.PAPER:
            try:
                self._balance = float(getattr(BROKER, "equity", self._balance))
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
        """
        Devuelve la posición abierta (si la hay) y cachea el resultado.
        En PAPER lee de PositionService (paper store). En LIVE consulta exchange/ccxt.
        """

        # 1) PAPER: leer del PositionService (persistente)
        if S.PAPER and POSITION_SERVICE is not None:
            try:
                st = POSITION_SERVICE.get_status()
                side = (st.get("side") or "FLAT").upper()
                if side != "FLAT":
                    self._open_position = {
                        "symbol": st.get("symbol", self.config.get("symbol", "BTC/USDT")),
                        "side": side,
                        "contracts": st.get("qty") or st.get("size") or 0.0,
                        "entryPrice": st.get("entry_price") or 0.0,
                        "markPrice": st.get("mark") or 0.0,
                    }
                    return self._open_position
                self._open_position = None
            except Exception as exc:
                logging.debug("PAPER check_open_position fallo: %s", exc)

        # 2) LIVE: si tenemos exchange, tratemos de obtener la posición real
        if exchange and getattr(exchange, 'client', None):
            try:
                positions = await asyncio.to_thread(
                    exchange.client.fetch_positions,
                    [self.config.get('symbol', 'BTC/USDT')]
                )
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
