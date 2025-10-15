import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import ccxt

from config import S
from trading import place_order_safe


logger = logging.getLogger(__name__)


class Exchange:
    def __init__(self, cfg):
        self.config = cfg
        self.public_client = None
        self.client = self._setup_client()
        if self.public_client is None:
            self.public_client = self.client
        self.is_authenticated = getattr(self, "is_authenticated", False)

    def _credentials_available(self) -> bool:
        return (
            (not S.PAPER)
            and bool(getattr(S, "binance_api_key", None))
            and bool(getattr(S, "binance_api_secret", None))
        )

    def _new_usdm_client(self):
        params = {"enableRateLimit": True, "options": {"defaultType": "future"}}
        client = ccxt.binanceusdm(params)
        use_testnet = os.getenv("BINANCE_UMFUTURES_TESTNET", "false").lower() == "true"
        if (S.PAPER or use_testnet) and hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
        return client

    async def upgrade_to_real_if_needed(self):
        """Si cambiaste a REAL y este Exchange sigue 'público', lo reautentico en caliente."""
        if not self._credentials_available():
            return
        if self.is_authenticated and getattr(self.client, "apiKey", None):
            return
        try:
            client = self._new_usdm_client()
            client.apiKey = S.binance_api_key
            client.secret = S.binance_api_secret
            await asyncio.to_thread(client.load_markets)
            self.client = client
            self.public_client = client
            self.is_authenticated = True
            logger.info("Cliente CCXT actualizado a AUTENTICADO tras cambio de modo.")
        except Exception:
            logger.warning("No pude reautenticar CCXT tras cambio a REAL.", exc_info=True)

    def _setup_client(self):
        """ Configura e inicializa el cliente del exchange. """
        self.is_authenticated = False  # Bandera de estado

        # 1. Intento de Conexión Autenticada (para Trading)
        params = {
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        }
        use_testnet = os.getenv('BINANCE_UMFUTURES_TESTNET', 'false').lower() == 'true'

        try:
            client = ccxt.binanceusdm(params)

            if not S.PAPER:
                client.apiKey = S.binance_api_key
                client.secret = S.binance_api_secret
            if (S.PAPER or use_testnet) and hasattr(client, 'set_sandbox_mode'):
                client.set_sandbox_mode(True)

            # Intentamos verificar si las credenciales funcionan cargando mercados
            client.load_markets()
            self.is_authenticated = True
            logger.info("Cliente de CCXT AUTENTICADO e inicializado correctamente.")
            return client

        except Exception as e:
            # 2. FALLBACK: Si las claves fallan, inicializar cliente PÚBLICO (solo lectura)
            logger.warning(
                f"ADVERTENCIA: Fallo de autenticación. El bot operará en modo SÓLO LECTURA. Error: {e}"
            )
            logger.warning("Verifique las claves de Testnet en su archivo .env.")

            # Creamos un cliente que NO necesita claves para obtener precios públicos
            public_client = ccxt.binanceusdm(params)
            if (S.PAPER or use_testnet) and hasattr(public_client, 'set_sandbox_mode'):
                public_client.set_sandbox_mode(True)
            self.public_client = public_client
            return public_client

    async def _ensure_auth_for_private(self):
        if S.PAPER:
            return
        await self.upgrade_to_real_if_needed()

    async def get_klines(self, timeframe: str = '1h', symbol: Optional[str] = None, limit: int = 300) -> List[List[Any]]:
        """Obtiene las velas (klines) de un par."""
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        logger.debug("Solicitando klines %s para %s", timeframe, sym)
        try:
            return await asyncio.to_thread(self.client.fetch_ohlcv, sym, timeframe=timeframe, limit=limit)
        except Exception:
            if sym.endswith('/USDT') and ':USDT' not in sym:
                fut_symbol = f"{sym}:USDT"
                logger.debug("Fallo al obtener klines para %s, probando %s", sym, fut_symbol)
                return await asyncio.to_thread(self.client.fetch_ohlcv, fut_symbol, timeframe=timeframe, limit=limit)
            raise

    async def get_current_price(self, symbol: Optional[str] = None) -> Optional[float]:
        """Obtiene el precio actual del símbolo configurado."""

        base_symbol = symbol or self.config.get('symbol', 'BTC/USDT')
        candidates = [base_symbol]
        if base_symbol.endswith('/USDT') and ':USDT' not in base_symbol:
            candidates.append(f"{base_symbol}:USDT")
        if base_symbol.replace('/', '') not in candidates:
            candidates.append(base_symbol.replace('/', ''))

        clients: List[Any] = []
        if self.public_client is not None:
            clients.append(self.public_client)
        if self.client is not None and self.client is not self.public_client:
            clients.append(self.client)

        for candidate in candidates:
            for client in clients:
                try:
                    data = await asyncio.to_thread(client.fetch_ticker, candidate)
                except Exception:
                    continue

                if not data:
                    continue

                price = (
                    data.get('last')
                    or data.get('close')
                    or data.get('ask')
                    or data.get('bid')
                )
                if price is not None:
                    try:
                        return float(price)
                    except Exception:
                        continue

        logger.warning(f"No pude obtener precio para {base_symbol}")
        return None

    async def set_leverage(self, leverage: float, symbol: Optional[str] = None) -> None:
        """Establece el apalancamiento para un par."""
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        if sym.endswith('/USDT') and ':USDT' not in sym:
            sym = f"{sym}:USDT"

        if S.PAPER:
            logger.info("PAPER: skipping private endpoint set_leverage(%s, %s)", leverage, sym)
            return

        await self._ensure_auth_for_private()
        await asyncio.to_thread(self.client.set_leverage, leverage, sym)
        logger.info("Apalancamiento establecido en x%s para %s", leverage, sym)

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

        logger.warning(f"No pude obtener funding rate para {base_symbol}")
        return None

    async def get_current_funding_rate_bps(self, symbol: Optional[str] = None) -> Optional[float]:
        """Mantiene compatibilidad devolviendo el funding rate en basis points."""
        rate_dec = await self.fetch_current_funding_rate(symbol)
        if rate_dec is None:
            return None
        return rate_dec * 10000.0

    async def fetch_positions(self, symbol: Optional[str] = None):
        """Obtiene posiciones abiertas evitando llamadas privadas en paper."""
        if self.client is None:
            return []
        if S.PAPER:
            return []
        await self._ensure_auth_for_private()
        try:
            if symbol:
                return await asyncio.to_thread(self.client.fetch_positions, [symbol])
            return await asyncio.to_thread(self.client.fetch_positions)
        except Exception:
            logger.debug("fetch_positions falló (symbol=%s)", symbol, exc_info=True)
            return []

    async def create_order(
        self,
        side: str,
        quantity: float,
        sl_price: float,
        tp_price: float,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Crea una orden pasando SIEMPRE por la ruta segura de trading."""

        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        logger.info("Creando orden %s para %.6f unidades de %s...", side, quantity, symbol)

        price = await self.get_current_price(symbol)
        if price is None:
            raise RuntimeError(f"No se pudo obtener precio para ejecutar la orden de {symbol}.")

        order = await asyncio.to_thread(
            place_order_safe,
            side,
            quantity,
            float(price),
            symbol=symbol,
            sl=sl_price,
            tp=tp_price,
        )

        logger.info("Orden ejecutada vía broker seguro: %s", order)
        return order
