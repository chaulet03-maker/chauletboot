import asyncio
import logging
from typing import Any, Dict, List, Optional

import ccxt


class Exchange:
    def __init__(self, cfg):
        self.config = cfg
        self.client = self._setup_client()

    def _setup_client(self):
        """Configura e inicializa el cliente del exchange (ccxt)."""
        mode = self.config.get('trading_mode', 'simulado')
        logging.info(f"Configurando el exchange en modo: {mode.upper()}")

        api_key_env = 'binance_api_key_real' if mode == 'real' else 'binance_api_key_test'
        api_secret_env = 'binance_api_secret_real' if mode == 'real' else 'binance_api_secret_test'

        client = ccxt.binance({
            'apiKey': self.config.get(api_key_env),
            'secret': self.config.get(api_secret_env),
            'options': {
                'defaultType': 'future',
            },
        })

        if mode != 'real':
            client.set_sandbox_mode(True)

        logging.info("Cliente de CCXT para Binance inicializado correctamente.")
        return client

    async def get_klines(self, timeframe: str = '1h', symbol: Optional[str] = None, limit: int = 300) -> List[List[Any]]:
        """Obtiene las velas (klines) de un par."""
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        ccxt_symbol = symbol.replace('/', '')
        logging.debug("Solicitando klines %s para %s", timeframe, ccxt_symbol)
        return await asyncio.to_thread(self.client.fetch_ohlcv, ccxt_symbol, timeframe=timeframe, limit=limit)

    async def get_current_price(self, symbol: Optional[str] = None) -> float:
        """Obtiene el último precio de un par."""
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        ticker = await asyncio.to_thread(self.client.fetch_ticker, symbol)
        price = float(ticker.get('last') or 0.0)
        logging.debug("Precio actual de %s: %s", symbol, price)
        return price

    async def set_leverage(self, leverage: float, symbol: Optional[str] = None) -> None:
        """Establece el apalancamiento para un par."""
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        ccxt_symbol = symbol.replace('/', '')
        await asyncio.to_thread(self.client.set_leverage, leverage, ccxt_symbol)
        logging.info("Apalancamiento establecido en x%s para %s", leverage, symbol)

    async def create_order(self, side: str, quantity: float, sl_price: float, tp_price: float, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Crea una nueva orden de mercado con SL y TP (simulada por ahora)."""
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        logging.info("Creando orden %s para %s unidades de %s...", side, quantity, symbol)
        # Aquí iría la lógica real de creación de orden con stop-loss y take-profit.
        # Por ahora devolvemos una estructura simulada para mantener la compatibilidad.
        return {
            "status": "ok",
            "side": side,
            "quantity": quantity,
            "symbol": symbol,
            "sl": sl_price,
            "tp": tp_price,
        }
