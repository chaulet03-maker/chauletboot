import asyncio
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

import inspect

import ccxt.async_support as ccxt
from ccxt.base.errors import ExchangeError, NetworkError

from bot.exchanges.binance_filters import build_filters
from bot.exchange_client import ensure_position_mode, get_ccxt, reset_ccxt_client, normalize_symbol
from bot.runtime_state import get_mode as runtime_get_mode
from bot.price_ws import PriceStream


def _get_trading_module():
    import trading

    return trading


def _get_brokers_module():
    import brokers

    return brokers


logger = logging.getLogger(__name__)

async def create_order_smart(
    symbol: str,
    side: str,
    qty: float,
    *,
    price=None,
    client=None,
    **extra_params,
):
    """
    Crea orden MARKET/limit de forma tolerante al modo:
    - Intenta SIN positionSide (ONE-WAY compatible).
    - Si Binance devuelve -4061 / mismatch => reintenta con positionSide=LONG/SHORT.
    """

    try:
        ex = client or get_ccxt()
    except Exception as exc:
        raise RuntimeError("Faltan credenciales o CCXT no disponible") from exc

    side_up = str(side).upper()
    order_type = "market" if price is None else "limit"
    params = dict(extra_params or {})

    try:
        return await ex.create_order(symbol, order_type, side_up, qty, price, params)
    except NetworkError as e:
        raise RuntimeError("Faltan credenciales (ccxt) o sin conectividad") from e
    except ExchangeError as e:
        message = str(e)
        if ("-4061" in message) or ("position side does not match" in message.lower()):
            ps = "LONG" if side_up in ("BUY", "LONG") else "SHORT"
            retry_params = dict(params)
            retry_params["positionSide"] = ps
            logger.info("Reintento con positionSide=%s (cuenta en HEDGE)", ps)
            return await ex.create_order(
                symbol, order_type, side_up, qty, price, retry_params
            )
        raise


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

    def _cache_price(
        self,
        symbol: Optional[str],
        price: float,
        ts: Optional[float] = None,
    ) -> None:
        try:
            value = float(price)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value <= 0:
            return

        base_symbol = symbol or self.config.get("symbol", "BTC/USDT")
        if not isinstance(base_symbol, str):
            base_symbol = str(base_symbol)

        now_ts = float(ts) if ts is not None else time.time()
        keys = {base_symbol}

        try:
            cfg_symbol = str(self.config.get("symbol", base_symbol))
            keys.add(cfg_symbol)
        except Exception:
            pass

        for key in list(keys):
            if not key:
                continue
            no_slash = key.replace("/", "")
            keys.add(no_slash)
            if key.endswith("/USDT"):
                keys.add(f"{key}:USDT")
            if no_slash.endswith("USDT"):
                keys.add(f"{no_slash}:USDT")

        with self._price_lock:
            for key in keys:
                if not key:
                    continue
                self._price_cache[key] = {"price": value, "ts": now_ts}

    def _on_price_stream_tick(
        self, symbol: str, price: float, event_ts: Optional[float] = None
    ) -> None:
        self._cache_price(symbol, price, event_ts)

    def _get_cached_price(
        self, key: str, *, max_age: float = 2.0
    ) -> Optional[float]:
        with self._price_lock:
            cached = self._price_cache.get(key)
        if not cached:
            return None
        try:
            ts = float(cached.get("ts", 0.0))
        except Exception:
            return None
        age = max(0.0, time.time() - ts)
        if age > max_age:
            return None
        try:
            return float(cached.get("price", 0.0))
        except (TypeError, ValueError):
            return None

    def _is_paper_runtime(self) -> bool:
        """
        Evalúa el modo efectivo en tiempo real, sin depender de S.PAPER.
        Retorna True si el runtime no está en 'live' o si no hay credenciales
        aplicadas al cliente actual.
        """

        mode = (runtime_get_mode() or "paper").lower()
        if mode not in {"real", "live"}:
            return True
        # En modo REAL/LIVE consideramos el runtime como live aunque CCXT todavía
        # no tenga credenciales disponibles; se usan fallbacks nativos en las
        # llamadas que lo requieran.
        return False

    def is_paper(self) -> bool:
        return self._is_paper_runtime()

    # --------- NUEVO: posiciones abiertas (CCXT) ---------
    def _maybe_persist_live_position(self, symbol: str, position: Dict[str, Any]) -> None:
        try:
            from state_store import create_position, persist_open  # import lazy para evitar ciclos

            qty = float(position.get("contracts") or 0.0)
            if qty <= 0:
                return
            entry = float(position.get("entryPrice") or 0.0)
            side = str(position.get("side") or "").upper()
            if not side:
                return
            leverage_cfg = self.config.get("leverage")
            try:
                leverage_val = float(leverage_cfg) if leverage_cfg is not None else 1.0
            except Exception:
                leverage_val = 1.0
            leverage_val = leverage_val if leverage_val > 0 else 1.0
            pos = create_position(
                symbol=position.get("symbol") or symbol,
                side=side,
                qty=qty,
                entry_price=entry,
                leverage=leverage_val,
                mode="live",
            )
            persist_open(pos)
        except Exception:
            logger.debug(
                "No se pudo persistir posición live detectada automáticamente.",
                exc_info=True,
            )

    async def get_open_position(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Devuelve una sola posición (la del símbolo configurado), o None si no hay.
        """
        # En modo PAPER nunca interrogamos posiciones “live”.
        if self.is_paper():
            try:
                trading = _get_trading_module()

                svc = getattr(trading, "POSITION_SERVICE", None)
                if svc is None:
                    return None
                st = svc.get_status() or {}
                side = str(st.get("side", "FLAT")).upper()
                qty = float(st.get("qty") or st.get("pos_qty") or 0.0)
                if side == "FLAT" or qty == 0.0:
                    return None
                return {
                    "symbol": symbol or self.config.get("symbol", "BTC/USDT"),
                    "side": side,
                    "contracts": float(abs(qty)),
                    "entryPrice": float(st.get("entry_price") or st.get("avg_price") or 0.0),
                    "markPrice": float(st.get("mark") or st.get("entry_price") or 0.0),
                }
            except Exception:
                return None
        sym = symbol or self.config.get("symbol", "BTC/USDT")
        target = sym.replace("/", "").upper()
        ccxt_client = getattr(self, "client", None)
        store_has_position = False
        sym_key = symbol or self.config.get("symbol", "BTC/USDT")
        try:
            from state_store import load_state  # import local para evitar ciclos

            st = await load_state() or {}
            open_pos = (st.get("open_positions") or {}).get(sym_key)
            qty_st = float((open_pos or {}).get("qty") or 0.0)
            if open_pos and abs(qty_st) > 0.0:
                store_has_position = True
                side_st = str(open_pos.get("side", "")).upper()
                entry_st = float(open_pos.get("entry_price") or 0.0)
                try:
                    mark_st = await self.get_current_price(sym_key)
                except Exception:
                    mark_st = entry_st
                return {
                    "symbol": normalize_symbol(sym_key) or sym_key,
                    "side": side_st,
                    "contracts": abs(qty_st),
                    "entryPrice": entry_st,
                    "markPrice": float(mark_st or entry_st),
                }
        except Exception:
            store_has_position = False

        pos_list: Optional[List[Dict[str, Any]]] = None
        if ccxt_client is not None and hasattr(ccxt_client, "fetch_positions"):
            try:
                pos_list = await ccxt_client.fetch_positions([sym])
            except Exception as exc:
                logger.debug("fetch_positions fallo, intento fallback: %s", exc)
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
                continue
            side = "LONG" if amt_f > 0 else "SHORT"
            entry_price = float(info.get("entryPrice") or entry.get("entryPrice") or 0.0)
            mark_price = float(
                info.get("markPrice") or info.get("markprice")
                or entry.get("markPrice") or entry.get("markprice") or 0.0
            )
            position = {
                "symbol": normalize_symbol(sym),
                "side": side,
                "contracts": abs(amt_f),
                "entryPrice": entry_price,
                "markPrice": mark_price,
            }
            if not store_has_position:
                self._maybe_persist_live_position(sym_key, position)
            return position
        # Fallback nativo (python-binance)
        try:
            brokers = _get_brokers_module()

            nat = getattr(brokers, "ACTIVE_LIVE_CLIENT", None)
            if nat and not self.is_paper():
                acct = await asyncio.to_thread(nat.futures_account)
                for pos in acct.get("positions", []):
                    sym_raw = str(pos.get("symbol") or "")
                    if sym_raw.replace("/", "") != target:
                        continue
                    amt = float(pos.get("positionAmt") or 0.0)
                    if abs(amt) == 0.0:
                        continue
                    side = "LONG" if amt > 0 else "SHORT"
                    entry = float(pos.get("entryPrice") or 0.0)
                    mark = float(pos.get("markPrice") or 0.0)
                    position = {
                        "symbol": normalize_symbol(sym),
                        "side": side,
                        "contracts": abs(amt),
                        "entryPrice": entry,
                        "markPrice": mark,
                    }
                    if not store_has_position:
                        self._maybe_persist_live_position(sym_key, position)
                    return position
        except Exception as exc:
            logger.debug("Fallback python-binance fallo: %s", exc)
        return None

    async def list_open_positions(self) -> List[Dict[str, Any]]:
        """
        Devuelve todas las posiciones abiertas del account (formato unificado).
        """
        # En modo PAPER no listamos posiciones de la cuenta real.
        if self.is_paper():
            # Si más adelante querés listar “otras” posiciones simuladas, se podría
            # extender para leerlas de un store múltiple. Hoy el bot sólo gestiona 1.
            return []

        out: List[Dict[str, Any]] = []
        ccxt_client = getattr(self, "client", None)
        # 1) CCXT en futures
        if ccxt_client is not None and hasattr(ccxt_client, "fetch_positions"):
            try:
                pos_list = await ccxt_client.fetch_positions(None, {"type": "future"})
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

                    symbol_fmt = normalize_symbol(sym_raw) or sym_raw

                    out.append(
                        {
                            "symbol": symbol_fmt,
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
            brokers = _get_brokers_module()

            nat = getattr(brokers, "ACTIVE_LIVE_CLIENT", None)
            if nat and not self.is_paper():
                acct = await asyncio.to_thread(nat.futures_account)
                for pos in acct.get("positions", []):
                    sym = str(pos.get("symbol") or "")
                    amt = float(pos.get("positionAmt") or 0.0)
                    if abs(amt) == 0.0:
                        continue
                    side = "LONG" if amt > 0 else "SHORT"
                    entry = float(pos.get("entryPrice") or 0.0)
                    mark = float(pos.get("markPrice") or 0.0)
                    symbol_fmt = normalize_symbol(sym) or sym
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

    async def _fetch_mark_price_rest(
        self, clients: List[Any], base_symbol: str
    ) -> Optional[float]:
        if not base_symbol:
            return None

        sym_clean = base_symbol.replace("/", "").upper()
        if not sym_clean:
            return None

        request = {"symbol": sym_clean}
        keys = ("markPrice", "markprice", "indexPrice", "indexprice")

        for client in clients:
            if client is None:
                continue

            for attr in ("public_get_premiumindex", "fapiPublicGetPremiumIndex"):
                method = getattr(client, attr, None)
                if method is None:
                    continue

                try:
                    result = method(request)
                    if inspect.isawaitable(result):
                        result = await result
                except Exception:
                    continue

                if isinstance(result, dict):
                    payloads = [result]
                elif isinstance(result, list):
                    payloads = [r for r in result if isinstance(r, dict)]
                else:
                    continue

                for payload in payloads:
                    # si viene una lista, priorizamos el símbolo exacto
                    sym_match = payload.get("symbol") or payload.get("pair")
                    if sym_match and str(sym_match).upper() != sym_clean:
                        continue
                    for key in keys:
                        value = payload.get(key)
                        if value is None:
                            continue
                        try:
                            return float(value)
                        except Exception:
                            continue

        return None

    async def fetch_balance_usdt(self) -> float:
        # En modo PAPER devolvemos el equity del PaperStore/pos service.
        if self.is_paper():
            try:
                trading = _get_trading_module()

                svc = getattr(trading, "POSITION_SERVICE", None)
                if svc is not None:
                    st = svc.get_status() or {}
                    eq = st.get("equity")
                    if eq is not None:
                        return float(eq)
                # Fallback: intentar leer el store activo si existe
                store = None
                try:
                    store = getattr(svc, "store", None)
                except Exception:
                    store = None
                if store is not None:
                    s = store.load() or {}
                    base = float(s.get("equity") or 0.0)
                    realized = float(s.get("realized_pnl") or 0.0)
                    fees = float(s.get("fees") or 0.0)
                    # mark podría venir nulo; no sumamos no-realizado para saldo
                    return float(base + realized - fees)
            except Exception:
                pass

        # 1) CCXT en Futuros USD-M
        try:
            ccxt_client = get_ccxt()
        except Exception:
            ccxt_client = None

        if ccxt_client is not None and hasattr(ccxt_client, "fetch_balance"):
            try:
                bal = await ccxt_client.fetch_balance({"type": "future"})
                total = (bal.get("total") or {}).get("USDT")
                if total is not None:
                    return float(total)

                usdt = bal.get("USDT") or bal.get("USDT:USDT") or {}
                if isinstance(usdt, dict):
                    t = usdt.get("total")
                    if t is not None:
                        return float(t)
                    return float((usdt.get("free", 0.0) or 0.0) + (usdt.get("used", 0.0) or 0.0))
                return float(usdt or 0.0)
            except Exception:
                try:
                    bal = await ccxt_client.fetch_balance()
                    total = (bal.get("total") or {}).get("USDT")
                    if total is not None:
                        return float(total)
                    usdt = bal.get("USDT") or {}
                    if isinstance(usdt, dict):
                        t = usdt.get("total")
                        if t is not None:
                            return float(t)
                        return float(
                            (usdt.get("free", 0.0) or 0.0)
                            + (usdt.get("used", 0.0) or 0.0)
                        )
                    if usdt not in (None, ""):
                        return float(usdt)
                except Exception:
                    pass

        # 2) Fallback nativo (python-binance): wallet de USD-M (solo si runtime live)
        try:
            brokers = _get_brokers_module()  # client python-binance si estás en REAL

            nat = getattr(brokers, "ACTIVE_LIVE_CLIENT", None)
            if nat and not self.is_paper():
                # futures_account_balance devuelve lista de assets de la wallet USDM
                data = await asyncio.to_thread(nat.futures_account_balance)
                for a in data or []:
                    if str(a.get("asset")) == "USDT":
                        # usar 'balance' / 'walletBalance'
                        return float(a.get("balance") or a.get("walletBalance") or 0.0)
        except Exception:
            pass
        return 0.0

    async def update_protections(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> dict[str, Any]:
        trading = _get_trading_module()

        broker = getattr(trading, "BROKER", None)
        if broker is None:
            return {"ok": False, "reason": "broker unavailable"}

        norm_symbol = str(symbol or self.config.get("symbol", "BTC/USDT")).replace("/", "")
        return await asyncio.to_thread(
            broker.update_protections,
            norm_symbol,
            side,
            float(qty),
            tp,
            sl,
        )

    def _start_price_stream(self):
        self._stop_price_stream()

        cfg: Mapping[str, Any]
        if isinstance(self.config, Mapping):
            cfg = self.config
        else:
            cfg = {}

        price_ws_cfg = cfg.get("price_ws", {}) if isinstance(cfg, Mapping) else {}
        enabled = False
        if isinstance(price_ws_cfg, Mapping):
            enabled = bool(price_ws_cfg.get("enabled", True))
        elif isinstance(price_ws_cfg, bool):
            enabled = price_ws_cfg
            price_ws_cfg = {}
        elif price_ws_cfg:
            enabled = bool(price_ws_cfg)
            price_ws_cfg = {}
        else:
            price_ws_cfg = {}

        runtime_mode = (runtime_get_mode() or "paper").lower()
        if not enabled:
            logger.info(
                "WS de precios deshabilitado por configuración; se utilizará REST para obtener cotizaciones."
            )
            return
        if runtime_mode not in {"real", "live"}:
            logger.info(
                "WS de precios sólo disponible en modo REAL. Runtime=%s; se utilizará REST.",
                runtime_mode,
            )
            return

        symbol = str(cfg.get("symbol", "BTC/USDT")) if cfg else "BTC/USDT"
        base_url = str(price_ws_cfg.get("url") or price_ws_cfg.get("base_url") or "")
        stream_name = str(price_ws_cfg.get("stream") or "mark").lower()
        interval = str(price_ws_cfg.get("interval") or "1s")

        use_testnet = (
            os.getenv("BINANCE_UMFUTURES_TESTNET", "false").lower() == "true"
        )
        if not base_url:
            base_url = (
                "wss://stream.binancefuture.com/ws"
                if use_testnet
                else "wss://fstream.binance.com/ws"
            )

        try:
            stream = PriceStream(
                symbol=symbol,
                callback=self._on_price_stream_tick,
                base_url=base_url,
                stream=stream_name,
                interval=interval,
            )
        except Exception:
            logger.exception("No se pudo configurar el WS de precios. Se usará REST.")
            return

        if use_testnet:
            logger.info("Iniciando WS de precios (testnet) para %s", symbol)
        else:
            logger.info("Iniciando WS de precios (live) para %s", symbol)

        try:
            stream.start()
        except Exception:
            logger.exception(
                "Fallo al iniciar el WS de precios; se continuará con REST."
            )
            return

        self._price_stream = stream

    def _stop_price_stream(self):
        stream = getattr(self, "_price_stream", None)
        if not stream:
            return
        try:
            stop = getattr(stream, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass
        self._price_stream = None

    def get_price_source(self) -> str:
        return "websocket" if getattr(self, "_price_stream", None) else "rest"

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

        if "timeout" not in params:
            params["timeout"] = 15000

        default_type = str(options.get("defaultType") or options.get("default_type") or "").lower()
        if default_type not in {"future", "swap"}:
            default_type = "future"
        options["defaultType"] = default_type
        options["hedgeMode"] = bool(self._hedge_mode)

        params["options"] = options

        requested_retries = params.get("retries")

        client = ccxt.binanceusdm(params)
        use_testnet = os.getenv("BINANCE_UMFUTURES_TESTNET", "false").lower() == "true"
        if use_testnet and hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
        try:
            if requested_retries is None:
                client.retries = 2
        except Exception:
            logger.debug("No se pudo fijar retries en cliente CCXT", exc_info=True)
        return client

    def _apply_client_options(self, client: Any) -> None:
        if client is None:
            return
        try:
            opts = getattr(client, "options", {}) or {}
            opts["defaultType"] = "future"
            opts["adjustForTimeDifference"] = True
            opts["hedgeMode"] = bool(self._hedge_mode)
            client.options = opts
        except Exception:
            logger.debug("No se pudieron aplicar opciones al cliente CCXT", exc_info=True)

    async def upgrade_to_real_if_needed(self):
        """Si cambiaste a REAL y este Exchange sigue 'público', lo reautentico en caliente."""
        mode = (runtime_get_mode() or "paper").lower()
        if mode not in {"real", "live"}:
            return
        if self.is_authenticated and getattr(self.client, "apiKey", None):
            return
        try:
            client = get_ccxt()
        except Exception:
            logger.warning("No pude reautenticar CCXT tras cambio a REAL.", exc_info=True)
            return

        self._apply_client_options(client)
        self.client = client
        self.public_client = client
        self.is_authenticated = True
        logger.info("Cliente CCXT actualizado a AUTENTICADO tras cambio de modo.")
        self._stop_price_stream()
        self._start_price_stream()

    async def downgrade_to_paper(self):
        """Vuelve a modo paper: cliente público sin credenciales."""
        try:
            reset_ccxt_client()
            client = self._new_usdm_client()
            self._apply_client_options(client)
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

        if self.is_paper():
            client = self._new_usdm_client()
            self._apply_client_options(client)
            self.public_client = client
            return client

        try:
            client = get_ccxt()
        except Exception as exc:
            logger.warning(
                "ADVERTENCIA: Fallo de autenticación CCXT. El bot operará en modo SÓLO LECTURA. Error: %s",
                exc,
            )
            logger.warning("Verifique las claves de Testnet en su archivo .env.")
            client = self._new_usdm_client()
            self._apply_client_options(client)
            self.public_client = client
            return client

        self._apply_client_options(client)
        self.public_client = client
        self.is_authenticated = True
        return client

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

        if self.is_paper():
            logger.debug(
                "PAPER: omitiendo set_position_mode(one_way=%s) (no aplica en paper)",
                one_way,
            )
            return

        hedged = not bool(one_way)
        try:
            ensure_position_mode(self.client, hedged)
        except Exception:
            logger.debug(
                "ensure_position_mode(one_way=%s) falló (no crítico)",
                one_way,
                exc_info=True,
            )

    async def _ensure_auth_for_private(self):
        if self.is_paper():
            return
        await self.upgrade_to_real_if_needed()

    async def get_klines(self, timeframe: str = '1h', symbol: Optional[str] = None, limit: int = 300) -> List[List[Any]]:
        """Obtiene las velas (klines) de un par."""
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        logger.debug("Solicitando klines %s para %s", timeframe, sym)
        try:
            return await self.client.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        except Exception:
            alt = sym.replace('/', '')
            if alt != sym:
                logger.debug("Fallo al obtener klines para %s, probando %s", sym, alt)
                return await self.client.fetch_ohlcv(alt, timeframe=timeframe, limit=limit)
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

        for key in (base_symbol, base_symbol.replace('/', ''), f"{base_symbol}:USDT"):
            px = self._get_cached_price(key)
            if px is not None:
                return px

        candidates = [base_symbol]
        no_slash = base_symbol.replace('/', '')
        if no_slash not in candidates:
            candidates.append(no_slash)
        if base_symbol.endswith('/USDT') and f"{base_symbol}:USDT" not in candidates:
            candidates.append(f"{base_symbol}:USDT")

        clients: List[Any] = []
        if self.public_client is not None:
            clients.append(self.public_client)
        if self.client is not None and self.client is not self.public_client:
            clients.append(self.client)

        mark_price = await self._fetch_mark_price_rest(clients, base_symbol)
        if mark_price is not None:
            self._cache_price(base_symbol, mark_price)
            return mark_price

        for candidate in candidates:
            for client in clients:
                fetch = getattr(client, "fetch_ticker", None)
                if fetch is None:
                    continue
                try:
                    data = fetch(candidate)
                    if inspect.isawaitable(data):
                        data = await data
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
                        value = float(price)
                        self._cache_price(base_symbol, value)
                        return value
                    except Exception:
                        continue

        logger.warning(f"No pude obtener precio para {base_symbol}")
        return None

    async def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        sym_clean = sym.replace('/', '').upper()

        client = getattr(self, 'client', None)
        if client is None:
            return {}

        markets: Dict[str, Any] = {}
        try:
            load = getattr(client, "load_markets", None)
            if callable(load):
                maybe_markets = load()
                if inspect.isawaitable(maybe_markets):
                    maybe_markets = await maybe_markets
                if isinstance(maybe_markets, dict):
                    markets = maybe_markets
        except Exception:
            markets = {}

        candidates = [sym, sym.upper(), sym_clean]
        if sym_clean.endswith('USDT') and not sym.endswith('/USDT'):
            candidates.append(f"{sym_clean[:-4]}/USDT")
        market = None
        for key in candidates:
            if not key:
                continue
            market = markets.get(key)
            if market:
                break

        if market is None:
            try:
                market = client.market(sym)
            except Exception:
                market = None

        if market is None and sym_clean.endswith('USDT'):
            alt = f"{sym_clean[:-4]}/USDT"
            market = markets.get(alt)

        if market is None:
            return {}

        filters = build_filters(sym_clean, market)
        return {
            "stepSize": float(filters.step_size),
            "minQty": float(filters.min_qty),
            "minNotional": float(filters.min_notional),
        }

    async def set_leverage(self, leverage: float | int, symbol: Optional[str] = None) -> None:
        """Establece el apalancamiento para un par."""
        sym = symbol or self.config.get('symbol', 'BTC/USDT')
        # CCXT acepta 'BTC/USDT' y lo mapea a 'BTCUSDT'. No sumar ':USDT'.
        lev_int = int(float(str(leverage).lower().replace("x", "").strip()))

        if self.is_paper():
            logger.info("PAPER: skipping set_leverage(%s, %s)", lev_int, sym)
            return

        await self._ensure_auth_for_private()
        await self.client.set_leverage(lev_int, sym)
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
                fetch = getattr(client, "fetch_funding_rate", None)
                if fetch is None:
                    continue
                try:
                    fr = fetch(candidate)
                    if inspect.isawaitable(fr):
                        fr = await fr
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
        if self._is_paper_runtime():
            return []
        await self._ensure_auth_for_private()
        try:
            if symbol:
                return await self.client.fetch_positions([symbol])
            return await self.client.fetch_positions()
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
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Crea una orden pasando SIEMPRE por la ruta segura de trading."""

        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')

        logger.info("Creando orden %s para %.6f unidades de %s...", side, quantity, symbol)

        price = await self.get_current_price(symbol)
        if price is None:
            raise RuntimeError(f"No se pudo obtener precio para ejecutar la orden de {symbol}.")

        if not self.is_paper():
            try:
                ensure_position_mode(self.client, self.hedge_mode)
            except Exception:
                logger.warning("No se pudo asegurar el modo de posición antes de abrir", exc_info=True)

        extra_params = dict(params or {})
        close_bot = bool(extra_params.pop("close_bot", False))

        order_kwargs = dict(symbol=symbol, sl=sl_price, tp=tp_price, order_type="MARKET")
        if self.hedge_mode:
            side_upper = str(side).upper()
            order_kwargs["positionSide"] = (
                "SHORT" if side_upper in {"SELL", "SHORT"} else "LONG"
            )

        if close_bot:
            order_kwargs.setdefault("reduceOnly", True)
            order_kwargs.setdefault("reduce_only", True)
            options = None
            params_options = extra_params.get("options") if isinstance(extra_params, dict) else None
            if isinstance(params_options, Mapping):
                options = params_options
            elif isinstance(self.config, Mapping):
                exchange_cfg = self.config.get("exchange")
                if isinstance(exchange_cfg, Mapping):
                    opt_cfg = exchange_cfg.get("options")
                    if isinstance(opt_cfg, Mapping):
                        options = opt_cfg
            if options is None:
                try:
                    client_options = getattr(self.client, "options", None)
                    if isinstance(client_options, Mapping):
                        options = client_options
                except Exception:
                    options = None
            is_hedge = False
            if isinstance(options, Mapping):
                try:
                    is_hedge = bool(options.get("hedgeMode"))
                except Exception:
                    is_hedge = False
            if not is_hedge:
                order_kwargs.pop("positionSide", None)
                if isinstance(extra_params, dict):
                    extra_params.pop("positionSide", None)

        if extra_params:
            order_kwargs.update(extra_params)

        from trading import place_order_safe

        order = await asyncio.to_thread(
            place_order_safe,
            side,
            quantity,
            None,
            **order_kwargs,
        )

        if not close_bot:
            try:
                trading_mod = _get_trading_module()
                infer_fn = getattr(trading_mod, "_infer_fill_price", None)
                entry_price = None
                if callable(infer_fn):
                    try:
                        entry_price = infer_fn(order, None)
                    except Exception:
                        entry_price = None
                if entry_price is not None and math.isfinite(entry_price):
                    qty_val = float(quantity)
                    if qty_val > 0:
                        side_label = (
                            "LONG" if str(side).upper() in {"BUY", "LONG"} else "SHORT"
                        )
                        try:
                            from bot.telemetry import telegram_bot as tg_bot

                            cur = tg_bot.sl_default_get()
                            if cur["mode"] == "abs":
                                sl_price_default = float(cur["value"])
                            else:
                                pct = float(cur["value"]) / 100.0
                                sl_price_default = (
                                    entry_price * (1 - pct)
                                    if side_label == "LONG"
                                    else entry_price * (1 + pct)
                                )
                            ok, msg = await tg_bot.trading_update_stop_loss(
                                sl_price_default,
                                side=side_label,
                                qty=qty_val if side_label == "LONG" else -qty_val,
                                symbol=symbol,
                            )
                            if not ok:
                                logger.warning("No pude setear SL inicial: %s", msg)
                        except Exception:
                            logger.exception("Fallo al setear SL inicial")
            except Exception:
                logger.exception("Fallo al setear SL inicial")

        logger.info("Orden ejecutada vía broker seguro: %s", order)
        return order
