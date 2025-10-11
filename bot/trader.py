import asyncio
import logging
from typing import Any, Dict, Optional


class Trader:
    def __init__(self, cfg):
        self.config = cfg
        self._balance = float(self.config.get('balance', 1000))
        self._open_position: Optional[Dict[str, Any]] = None

    async def get_balance(self, exchange=None) -> float:
        """Devuelve el balance actual de la cuenta."""
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
        """Revisa si hay una posición abierta."""
        if exchange and getattr(exchange, 'client', None):
            try:
                positions = await asyncio.to_thread(exchange.client.fetch_positions, [self.config.get('symbol', 'BTC/USDT')])
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
