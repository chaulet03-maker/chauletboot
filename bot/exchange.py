import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import ccxt

from config import S
from trading import place_order_safe


logger = logging.getLogger(__name__)


class Exchange:
    def __init__(self, cfg):
        self.config = cfg
        self._hedge_mode = self._determine_hedge_mode()
        self.public_client = None
        self.client = self._setup_client()
        if self.public_client is None:
            self.public_client = self.client
        self.is_authenticated = getattr(self, "is_authenticated", False)
        self._price_cache: Dict[str, Dict[str, float]] = {}
        self._price_lock = threading.Lock()
        self._price_stream = None
        self._start_price_stream()

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").lower()

    def _start_price_stream(self):
        symbol = self.config.get("symbol", "BTC/USDT")
        ws_symbol = self._normalize_symbol(symbol)
        try:
            from binance.websocket.um_futures.websocket_client import (  # type: ignore
                UMFuturesWebsocketClient,
            )
        except Exception:
            logger.debug("UMFuturesWebsocketClient no disponible, fallback REST")
            return

        def handle(message):
            if not isinstance(message, dict):
                return
            price = (
                message.get("p")
                or message.get("price")
                or message.get("markPrice")
                or message.get("c")
            )
            if price is None:
                return
            try:
                px = float(price)
            except Exception:
                return
            ts = float(message.get("E") or message.get("eventTime") or time.time()) / 1000.0
            with self._price_lock:
                self._price_cache[symbol] = {"price": px, "ts": ts}

        try:
            self._price_stream = UMFuturesWebsocketClient(on_message=handle)
            self._price_stream.start()
            self._price_stream.book_ticker(symbol=ws_symbol)
        except Exception:
            logger.warning("No se pudo iniciar websocket de precios", exc_info=True)
            self._price_stream = None

    def _stop_price_stream(self):
        stream = getattr(self, "_price_stream", None)
        if not stream:
            return
        try:
            stream.stop()
        except Exception:
            pass
        self._price_stream = None

    def _determine_hedge_mode(self) -> bool:
        cfg: Mapping[str, Any] = (
            self.config if isinstance(self.config, Mapping) else {}
        )

        exchange_cfg = cfg.get("exchange")
        if isinstance(exchange_cfg, Mapping):
            options = exchange_cfg.get("options")
            if isinstance(options, Mapping):
                if "hedgeMode" in options:
                    return bool(options.get("hedgeMode"))
                if "hedge_mode" in options:
                    return bool(options.get("hedge_mode"))
            if "hedgeMode" in exchange_cfg:
                return bool(exchange_cfg.get("hedgeMode"))
            if "hedge_mode" in exchange_cfg:
                return bool(exchange_cfg.get("hedge_mode"))

        limits_cfg = cfg.get("limits")
        if isinstance(limits_cfg, Mapping) and "no_hedge" in limits_cfg:
            return not bool(limits_cfg.get("no_hedge"))

        return True

    def _credentials_available(self) -> bool:
        return (
            (not S.PAPER)
            and bool(getattr(S, "binance_api_key", None))
            and bool(getattr(S, "binance_api_secret", None))
        )

    def _new_usdm_client(self):
        exchange_cfg = self.config.get("exchange") if isinstance(self.config, Mapping) else None

        params: Dict[str, Any] = {"enableRateLimit": True}
        options: Dict[str, Any] = {}

        if isinstance(exchange_cfg, Mapping):
            user_params = exchange_cfg.get("params") or exchange_cfg.get("ccxt_params")
            if isinstance(user_params, Mapping):
                params.update(user_params)
            user_options = exchange_cfg.get("options")
            if isinstance(user_options, Mapping):
                options.update(user_options)

        options.setdefault("defaultType", "future")
        if self._hedge_mode:
            options["hedgeMode"] = True

        params["options"] = options

        client = ccxt.binanceusdm(params)
        use_testnet = os.getenv("BINANCE_UMFUTURES_TESTNET", "false").lower() == "true"
        if use_testnet and hasattr(client, "set_sandbox_mode"):
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
            self._stop_price_stream()
            self._start_price_stream()
        except Exception:
            logger.warning("No pude reautenticar CCXT tras cambio a REAL.", exc_info=True)

    async def downgrade_to_paper(self):
        """Vuelve a modo paper: cliente público sin credenciales."""
        try:
            client = self._new_usdm_client()
            self.client = client
            self.public_client = client
            self.is_authenticated = False
            logger.info("Cliente CCXT cambiado a PÚBLICO (paper).")
            self._stop_price_stream()
            self._start_price_stream()
        except Exception:
            logger.warning("No pude pasar exchange a paper.", exc_info=True)

    def _setup_client(self):
        """ Configura e inicializa el cliente del exchange. """
        self.is_authenticated = False  # Bandera de estado

        if not self._credentials_available():
            client = self._new_usdm_client()
            self.public_client = client
            return client

        try:
            client = self._new_usdm_client()
            client.apiKey = S.binance_api_key
            client.secret = S.binance_api_secret
            client.load_markets()
            self.is_authenticated = True
            logger.info("Cliente de CCXT AUTENTICADO e inicializado correctamente.")
            return client

        except Exception as e:
            logger.warning(
                "ADVERTENCIA: Fallo de autenticación. El bot operará en modo SÓLO LECTURA. Error: %s",
                e,
            )
            logger.warning("Verifique las claves de Testnet en su archivo .env.")

            public_client = self._new_usdm_client()
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

    def get_price_age_sec(self, symbol: Optional[str] = None) -> float:
        """Edad (segundos) del último precio cacheado del WS; grande => stream frío."""

        base_symbol = symbol or self.config.get('symbol', 'BTC/USDT')
        with self._price_lock:
            cached = self._price_cache.get(base_symbol)
        if not cached:
            return float("inf")
        ts = float(cached.get("ts", 0.0))
        return max(0.0, time.time() - ts)

    async def get_current_price(self, symbol: Optional[str] = None) -> Optional[float]:
        """Obtiene el precio actual del símbolo configurado."""

        base_symbol = symbol or self.config.get('symbol', 'BTC/USDT')
        with self._price_lock:
            cached = self._price_cache.get(base_symbol)
        age = self.get_price_age_sec(base_symbol)
        if cached and age <= 2.0:
            return float(cached.get("price", 0.0))
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

    async def set_leverage(self, leverage: float | int, symbol: Optional[str] = None) -> None:
        """Establece el apalancamiento para un par."""
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        # CCXT acepta 'BTC/USDT' y lo mapea a 'BTCUSDT'. No sumar ':USDT'.
        lev_int = int(float(str(leverage).lower().replace("x", "").strip()))

        if S.PAPER:
            logger.info("PAPER: skipping set_leverage(%s, %s)", lev_int, sym)
            return

        await self._ensure_auth_for_private()
        await asyncio.to_thread(self.client.set_leverage, lev_int, sym)
        logger.info("Apalancamiento establecido en x%s para %s", lev_int, sym)

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
