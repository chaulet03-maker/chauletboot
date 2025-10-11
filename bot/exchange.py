import asyncio
import logging
from typing import Any, Dict, List, Optional

import ccxt


class Exchange:
    def __init__(self, cfg):
        self.config = cfg
        self.public_client = None
        self.client = self._setup_client()
        if self.public_client is None:
            self.public_client = self.client

    def _setup_client(self):
        """ Configura e inicializa el cliente del exchange. """
        mode = self.config.get('trading_mode', 'simulado')
        self.is_authenticated = False  # Bandera de estado

        # 1. Intento de Conexión Autenticada (para Trading)
        try:
            if mode == 'real':
                api_key = self.config.get('binance_api_key_real')
                secret = self.config.get('binance_api_secret_real')
            else:
                api_key = self.config.get('binance_api_key_test')
                secret = self.config.get('binance_api_secret_test')

            client = ccxt.binance({
                'apiKey': api_key,
                'secret': secret,
                'options': {'defaultType': 'future'},
            })

            if mode != 'real':
                client.set_sandbox_mode(True)

            # Intentamos verificar si las credenciales funcionan cargando mercados
            client.load_markets()
            self.is_authenticated = True
            logging.info("Cliente de CCXT AUTENTICADO e inicializado correctamente.")
            return client

        except Exception as e:
            # 2. FALLBACK: Si las claves fallan, inicializar cliente PÚBLICO (solo lectura)
            logging.warning(
                f"ADVERTENCIA: Fallo de autenticación. El bot operará en modo SÓLO LECTURA. Error: {e}"
            )
            logging.warning("Verifique las claves de Testnet en su archivo .env.")

            # Creamos un cliente que NO necesita claves para obtener precios públicos
            public_client = ccxt.binance({
                'options': {'defaultType': 'future'},
            })
            self.public_client = public_client
            return public_client

    async def get_klines(self, timeframe: str = '1h', symbol: Optional[str] = None, limit: int = 300) -> List[List[Any]]:
        """Obtiene las velas (klines) de un par."""
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        ccxt_symbol = symbol.replace('/', '')
        logging.debug("Solicitando klines %s para %s", timeframe, ccxt_symbol)
        return await asyncio.to_thread(self.client.fetch_ohlcv, ccxt_symbol, timeframe=timeframe, limit=limit)

    async def get_current_price(self, symbol=None):
        """ Obtiene el último precio de un par de forma resiliente. """
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        # Priorizamos el cliente autenticado si existe, sino el público de respaldo.
        client_to_use = self.client if self.client is not None else self.public_client

        try:
            # Usamos el método fetch_ticker que es universal
            ticker = await client_to_use.fetch_ticker(symbol)

            # El precio se encuentra en la clave 'last'
            price = ticker.get('last')
            if price is not None:
                return price
            else:
                # Si 'last' no existe por alguna razón, devolvemos None.
                logging.warning(f"El ticker no devolvió el precio 'last' para {symbol}.")
                return None

        except Exception as e:
            # Captura cualquier error de conexión o API y lo informa sin romper el bot.
            logging.error(f"FALLA CRÍTICA: No se pudo obtener el precio de {symbol}: {e}")
            return None

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
