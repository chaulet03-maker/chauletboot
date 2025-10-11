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
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        logging.debug("Solicitando klines %s para %s", timeframe, sym)
        try:
            return await asyncio.to_thread(self.client.fetch_ohlcv, sym, timeframe=timeframe, limit=limit)
        except Exception:
            if sym.endswith('/USDT') and ':USDT' not in sym:
                fut_symbol = f"{sym}:USDT"
                logging.debug("Fallo al obtener klines para %s, probando %s", sym, fut_symbol)
                return await asyncio.to_thread(self.client.fetch_ohlcv, fut_symbol, timeframe=timeframe, limit=limit)
            raise

    async def get_current_price(self, symbol=None):
        """ Obtiene el último precio de un par de forma resiliente. """
        base_symbol = symbol or self.config.get('symbol', 'BTC/USDT')
        candidates = [base_symbol]
        if base_symbol.endswith('/USDT') and ':USDT' not in base_symbol:
            candidates.append(f"{base_symbol}:USDT")
        stripped = base_symbol
        if stripped not in candidates:
            candidates.append(stripped)

        clients: List[Any] = []
        if self.client is not None:
            clients.append(self.client)
        if self.public_client is not None and self.public_client is not self.client:
            clients.append(self.public_client)

        for candidate in candidates:
            for client in clients:
                try:
                    ticker = await asyncio.to_thread(client.fetch_ticker, candidate)
                    if ticker and ticker.get('last') is not None:
                        return ticker['last']
                except Exception:
                    continue

        logging.warning(f"No pude obtener precio para {base_symbol}")
        return None

    async def set_leverage(self, leverage: float, symbol: Optional[str] = None) -> None:
        """Establece el apalancamiento para un par."""
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        if sym.endswith('/USDT') and ':USDT' not in sym:
            sym = f"{sym}:USDT"

        await asyncio.to_thread(self.client.set_leverage, leverage, sym)
        logging.info("Apalancamiento establecido en x%s para %s", leverage, sym)

    async def fetch_current_funding_rate(self, symbol: Optional[str] = None) -> Optional[float]:
        """Obtiene el funding rate actual en formato decimal (por intervalo de funding)."""
        base_symbol = symbol or self.config.get('symbol', 'BTC/USDT')
        candidates = [base_symbol]
        if base_symbol.endswith('/USDT') and ':USDT' not in base_symbol:
            candidates.append(f"{base_symbol}:USDT")
        stripped = base_symbol
        if stripped not in candidates:
            candidates.append(stripped)

        clients: List[Any] = []
        if self.client is not None:
            clients.append(self.client)
        if self.public_client is not None and self.public_client is not self.client:
            clients.append(self.public_client)

        for candidate in candidates:
            for client in clients:
                try:
                    fr = await asyncio.to_thread(client.fetch_funding_rate, candidate)
                    if fr is None:
                        continue
                    rate = fr.get('fundingRate')
                    if rate is None:
                        continue
                    rate = float(rate)
                    return rate
                except Exception:
                    continue

        logging.warning(f"No pude obtener funding rate para {base_symbol}")
        return None

    async def get_current_funding_rate_bps(self, symbol: Optional[str] = None) -> Optional[float]:
        """Mantiene compatibilidad devolviendo el funding rate en basis points."""
        rate_dec = await self.fetch_current_funding_rate(symbol)
        if rate_dec is None:
            return None
        return rate_dec * 10000.0

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
