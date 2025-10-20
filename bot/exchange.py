import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import ccxt
from ccxt.base.errors import OperationRejected

from config import S
from trading import place_order_safe


logger = logging.getLogger(__name__)


class Exchange:
    def __init__(self, cfg):
        self.config = cfg
        self._hedge_mode = self._determine_hedge_mode()
        self.public_client = None
        self.client = self._setup_client()
        # Asegurar que CCXT apunte a Futuros USD-M
        try:
            opts = getattr(self.client, "options", {}) or {}
            if opts.get("defaultType") not in {"future", "swap"}:
                opts["defaultType"] = "future"
            # Hedge mode si lo usás
            if "hedgeMode" not in opts:
                opts["hedgeMode"] = True
            self.client.options = opts
        except Exception:
            pass
        if self.public_client is None:
            self.public_client = self.client
        self.is_authenticated = getattr(self, "is_authenticated", False)
        self._price_cache: Dict[str, Dict[str, float]] = {}
        self._price_lock = threading.Lock()
        self._price_stream = None
        self._start_price_stream()

    # --------- NUEVO: posiciones abiertas (CCXT) ---------
    async def get_open_position(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Devuelve una sola posición (la del símbolo configurado), o None si no hay.
        """
        sym = symbol or self.config.get("symbol", "BTC/USDT")
        target = sym.replace("/", "").upper()
        ccxt_client = getattr(self, "client", None)
        if ccxt_client is None or not hasattr(ccxt_client, "fetch_positions"):
            return None
        try:
            pos_list = await asyncio.to_thread(ccxt_client.fetch_positions, [sym])
        except Exception:
            return None
        for entry in pos_list or []:
            info = entry.get("info") or {}
            exch_sym = str(info.get("symbol") or entry.get("symbol") or "").upper()
            if exch_sym.replace("/", "") != target:
                continue
            raw_amt = info.get("positionAmt") or info.get("positionamt")
            if raw_amt is None:
                raw_amt = entry.get("contracts") or entry.get("size") or 0
            amt_f = float(raw_amt or "0")
            if abs(amt_f) == 0.0:
                return None
            side = "LONG" if amt_f > 0 else "SHORT"
            entry_price = float(info.get("entryPrice") or entry.get("entryPrice") or 0.0)
            mark_price = float(
                info.get("markPrice") or info.get("markprice")
                or entry.get("markPrice") or entry.get("markprice") or 0.0
            )
            return {
                "symbol": sym,
                "side": side,
                "contracts": abs(amt_f),
                "entryPrice": entry_price,
                "markPrice": mark_price,
            }
        return None

    async def list_open_positions(self) -> List[Dict[str, Any]]:
        """
        Devuelve todas las posiciones abiertas del account (formato unificado).
        """
        out: List[Dict[str, Any]] = []
        ccxt_client = getattr(self, "client", None)
        # 1) CCXT en futures
        if ccxt_client is not None and hasattr(ccxt_client, "fetch_positions"):
            try:
                pos_list = await asyncio.to_thread(
                    ccxt_client.fetch_positions, None, {"type": "future"}
                )
                for entry in pos_list or []:
                    info = entry.get("info") or {}
                    raw_amt = (
                        info.get("positionAmt")
                        or entry.get("contracts")
                        or entry.get("size")
                        or 0
                    )
                    amt_f = float(raw_amt or 0)
                    if abs(amt_f) == 0.0:
                        continue
                    side = "LONG" if amt_f > 0 else "SHORT"
                    sym_raw = str(info.get("symbol") or entry.get("symbol") or "")

                    def fmt(s: str) -> str:
                        return s if "/" in s else (s[:-4] + "/USDT" if s.endswith("USDT") else s)

                    out.append(
                        {
                            "symbol": fmt(sym_raw),
                            "side": side,
                            "contracts": abs(amt_f),
                            "entryPrice": float(
                                info.get("entryPrice")
                                or entry.get("entryPrice")
                                or 0.0
                            ),
                            "markPrice": float(
                                info.get("markPrice")
                                or entry.get("markPrice")
                                or 0.0
                            ),
                        }
                    )
                if out:
                    return out
            except Exception:
                pass

        # 2) Fallback nativo (python-binance)
        try:
            from brokers import ACTIVE_LIVE_CLIENT

            nat = ACTIVE_LIVE_CLIENT
            if nat:
                acct = await asyncio.to_thread(nat.futures_account)
                for pos in acct.get("positions", []):
                    sym = str(pos.get("symbol") or "")
                    amt = float(pos.get("positionAmt") or 0.0)
                    if abs(amt) == 0.0:
                        continue
                    side = "LONG" if amt > 0 else "SHORT"
                    entry = float(pos.get("entryPrice") or 0.0)
                    mark = float(pos.get("markPrice") or 0.0)
                    symbol_fmt = (
                        sym
                        if "/" in sym
                        else (sym[:-4] + "/USDT" if sym.endswith("USDT") else sym)
                    )
                    out.append(
                        {
                            "symbol": symbol_fmt,
                            "side": side,
                            "contracts": abs(amt),
                            "entryPrice": entry,
                            "markPrice": mark,
                        }
                    )
        except Exception:
            return out
        return out

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").lower()

    async def fetch_balance_usdt(self) -> float:
        # 1) CCXT en Futuros USD-M
        try:
            bal = await asyncio.to_thread(self.client.fetch_balance, {"type": "future"})
            usdt = bal.get("USDT") or (bal.get("total", {}) or {}).get("USDT")
            if isinstance(usdt, dict):
                # En CCXT para futures, 'total' refleja wallet; si no, suma free+used
                total = usdt.get("total")
                if total is not None:
                    return float(total)
                return float((usdt.get("free", 0.0) or 0.0) + (usdt.get("used", 0.0) or 0.0))
            return float(usdt or 0.0)
        except Exception:
            pass

        # 2) Fallback nativo (python-binance): wallet de USD-M
        try:
            from brokers import ACTIVE_LIVE_CLIENT  # client python-binance si estás en REAL

            nat = ACTIVE_LIVE_CLIENT
            if nat:
                # futures_account_balance devuelve lista de assets de la wallet USDM
                data = await asyncio.to_thread(nat.futures_account_balance)
                for a in data or []:
                    if str(a.get("asset")) == "USDT":
                        # usar 'balance' / 'walletBalance'
                        return float(a.get("balance") or a.get("walletBalance") or 0.0)
        except Exception:
            pass
        return 0.0

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

        default_type = str(options.get("defaultType") or options.get("default_type") or "").lower()
        if default_type not in {"future", "swap"}:
            default_type = "future"
        options["defaultType"] = default_type
        options["hedgeMode"] = bool(self._hedge_mode)

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

    @property
    def hedge_mode(self) -> bool:
        """Indica si el exchange opera en modo hedge (dual side)."""

        return bool(self._hedge_mode)

    async def set_position_mode(self, one_way: Optional[bool] = None) -> None:
        """Configura el modo de posición (one-way o hedge) en el exchange."""

        if one_way is None:
            one_way = not self.hedge_mode

        if self.client is None:
            return

        if S.PAPER:
            logger.debug(
                "PAPER: omitiendo set_position_mode(one_way=%s) (no aplica en paper)",
                one_way,
            )
            return

        await self._ensure_auth_for_private()

        hedged = not bool(one_way)
        try:
            client = self.client
            if hasattr(client, "set_position_mode"):
                await asyncio.to_thread(client.set_position_mode, hedged)
                return
            if hasattr(client, "setPositionMode"):
                await asyncio.to_thread(client.setPositionMode, hedged)
                return

            request = {"dualSidePosition": "true" if hedged else "false"}
            rest_method = getattr(client, "fapiPrivate_post_positionside_dual", None)
            if rest_method is None:
                rest_method = getattr(client, "fapiPrivatePostPositionSideDual", None)
            if rest_method is None:
                raise AttributeError("Método PositionSideDual no disponible en el cliente CCXT")
            await asyncio.to_thread(rest_method, request)
        except OperationRejected as exc:
            message = str(exc)
            if "-4059" in message or "No need to change position side" in message:
                logger.info(
                    "set_position_mode(one_way=%s) omitido: ya estaba configurado.",
                    one_way,
                )
                return
            raise
        except Exception:
            logger.warning(
                "set_position_mode(one_way=%s) falló", one_way, exc_info=True
            )

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

        order_kwargs = dict(symbol=symbol, sl=sl_price, tp=tp_price, order_type="MARKET")
        if self.hedge_mode:
            side_upper = str(side).upper()
            order_kwargs["positionSide"] = (
                "SHORT" if side_upper in {"SELL", "SHORT"} else "LONG"
            )

        order = await asyncio.to_thread(
            place_order_safe,
            side,
            quantity,
            None,
            **order_kwargs,
        )

        logger.info("Orden ejecutada vía broker seguro: %s", order)
        return order
