from __future__ import annotations

import logging
import math
import os
import time
from threading import Lock
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from bot.mode_manager import Mode, ModeResult, get_mode
from config import RAW_CONFIG, S
from brokers import ACTIVE_LIVE_CLIENT, ACTIVE_PAPER_STORE, build_broker
from binance_client import client_factory
from position_service import PositionService, EPS_QTY
from paper_store import PaperStore
from state_store import on_close_filled, on_open_filled
from bot.telemetry.metrics import get_metrics
from bot.runtime_state import get_mode as runtime_get_mode
from paths import get_data_dir, get_paper_store_path
from bot.logger import _warn

logger = logging.getLogger(__name__)

BROKER: Any | None = None
POSITION_SERVICE: PositionService | None = None
PUBLIC_CCXT_CLIENT: Optional[Any] = None
ACTIVE_MODE: Mode = "simulado"
ACTIVE_DATA_DIR: Optional[Path] = None  # type: ignore[name-defined]
ACTIVE_STORE_PATH: Optional[Path] = None  # type: ignore[name-defined]
LAST_MODE_CHANGE_SOURCE: str = "startup"
_INITIALIZED: bool = False

_SYMBOL_RULE_CACHE: dict[tuple[int, str], dict[str, float]] = {}
_METRICS = get_metrics()

_ENTRY_MUTEX = Lock()

_LAST_REAL_SYNC: tuple[float, float] | None = None


def _normalize_symbol(symbol: str | None) -> str:
    if not symbol:
        return ""
    value = str(symbol).replace("/", "")
    return value.upper()


def get_live_client() -> Optional[Any]:
    broker = BROKER
    client = getattr(broker, "client", None) if broker is not None else None
    if client is None:
        client = ACTIVE_LIVE_CLIENT
    return client


def _extract_symbol_rules(client: Any, symbol: str) -> dict[str, float]:
    if client is None:
        raise RuntimeError("No hay cliente LIVE disponible para obtener filtros.")
    norm = _normalize_symbol(symbol)
    if not norm:
        raise ValueError("Símbolo inválido para obtener reglas")
    cache_key = (id(client), norm)
    cached = _SYMBOL_RULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    price_precision = 2
    qty_precision = 3
    step_size = 0.0
    min_qty = 0.0
    min_notional = 0.0

    try:
        info = client.futures_exchange_info()
    except Exception:
        info = None

    symbols = []
    if isinstance(info, dict):
        symbols = info.get("symbols") or []
    if not isinstance(symbols, list):
        symbols = []

    for entry in symbols:
        if not isinstance(entry, dict):
            continue
        entry_symbol = str(entry.get("symbol") or "").upper()
        if entry_symbol != norm:
            continue
        try:
            price_precision = int(entry.get("pricePrecision", price_precision))
        except Exception:
            pass
        qty_prec = entry.get("quantityPrecision")
        if qty_prec is None:
            qty_prec = entry.get("qtyPrecision")
        try:
            if qty_prec is not None:
                qty_precision = int(qty_prec)
        except Exception:
            pass
        filters = entry.get("filters") or []
        if isinstance(filters, list):
            for raw_filter in filters:
                if not isinstance(raw_filter, dict):
                    continue
                ftype = str(raw_filter.get("filterType") or "").upper()
                if ftype in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
                    step_val = raw_filter.get("stepSize")
                    min_qty_val = raw_filter.get("minQty")
                    try:
                        if step_val is not None:
                            step_size = float(step_val)
                    except Exception:
                        pass
                    try:
                        if min_qty_val is not None:
                            min_qty = float(min_qty_val)
                    except Exception:
                        pass
                if ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
                    try:
                        candidate = raw_filter.get("notional") or raw_filter.get("minNotional")
                        if candidate is not None:
                            min_notional = float(candidate)
                    except Exception:
                        pass
        break

    rules = {
        "price_precision": int(price_precision),
        "qty_precision": int(qty_precision),
        "step_size": float(step_size or 0.0),
        "min_qty": float(min_qty or 0.0),
        "min_notional": float(min_notional or 0.0),
    }
    _SYMBOL_RULE_CACHE[cache_key] = rules
    return rules


def get_symbol_rules(symbol: str) -> dict[str, float]:
    client = get_live_client()
    if client is None:
        raise RuntimeError("No hay cliente LIVE disponible")
    return _extract_symbol_rules(client, symbol)


def _quantize_qty(raw_qty: float, rules: Mapping[str, float]) -> float:
    try:
        qty = float(raw_qty)
    except Exception:
        return 0.0
    if qty <= 0:
        return 0.0
    step = float(rules.get("step_size") or 0.0)
    precision = int(rules.get("qty_precision") or 0)
    if step > 0:
        qty = math.floor((qty / step) + 1e-9) * step
    if precision >= 0:
        qty = round(qty, precision)
    return qty


def _fetch_live_position(client: Any, symbol_norm: str) -> Optional[Dict[str, Any]]:
    payload = None
    if hasattr(client, "futures_position_information"):
        try:
            payload = client.futures_position_information(symbol=symbol_norm)
        except Exception:
            payload = None
    if payload is None:
        if hasattr(client, "fapiPrivate_get_positionrisk"):
            try:
                payload = client.fapiPrivate_get_positionrisk({"symbol": symbol_norm})
            except Exception:
                payload = None
        elif hasattr(client, "fapiPrivateGetPositionRisk"):
            try:
                payload = client.fapiPrivateGetPositionRisk({"symbol": symbol_norm})
            except Exception:
                payload = None

    entries: list[Dict[str, Any]] = []
    if isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
    elif isinstance(payload, dict):
        entries = [payload]

    for entry in entries:
        entry_symbol = str(entry.get("symbol") or entry.get("symbolName") or "").upper()
        if entry_symbol != symbol_norm:
            continue
        try:
            amount = float(
                entry.get("positionAmt")
                or entry.get("position_amt")
                or entry.get("amount")
                or entry.get("qty")
                or 0.0
            )
        except Exception:
            amount = 0.0
        if abs(amount) <= EPS_QTY:
            continue
        side = "LONG" if amount > 0 else "SHORT"
        try:
            entry_price = float(entry.get("entryPrice") or entry.get("avgPrice") or 0.0)
        except Exception:
            entry_price = 0.0
        try:
            mark_price = float(entry.get("markPrice") or entry.get("mark_price") or entry_price)
        except Exception:
            mark_price = entry_price
        try:
            leverage = float(entry.get("leverage") or 1.0)
        except Exception:
            leverage = 1.0
        return {
            "side": side,
            "qty": abs(amount),
            "entry_price": entry_price,
            "mark_price": mark_price,
            "leverage": leverage,
            "raw": entry,
        }
    return None


def fetch_live_position(symbol: str) -> Optional[Dict[str, Any]]:
    client = get_live_client()
    if client is None:
        return None
    return _fetch_live_position(client, _normalize_symbol(symbol))


def fetch_futures_usdt_balance() -> Optional[float]:
    client = get_live_client()
    if client is None:
        return None
    try:
        balances = client.futures_account_balance()
    except Exception:
        balances = None
    if isinstance(balances, list):
        for entry in balances:
            if not isinstance(entry, dict):
                continue
            asset = str(entry.get("asset") or "").upper()
            if asset != "USDT":
                continue
            for key in ("balance", "crossWalletBalance", "walletBalance", "availableBalance"):
                if entry.get(key) not in (None, ""):
                    try:
                        return float(entry[key])
                    except Exception:
                        continue
    return None


def get_latest_price(symbol: str) -> Optional[float]:
    symbol_norm = _normalize_symbol(symbol)
    ccxt_symbol = symbol if "/" in symbol else f"{symbol[:-4]}/USDT" if symbol.upper().endswith("USDT") else symbol
    if PUBLIC_CCXT_CLIENT is not None and hasattr(PUBLIC_CCXT_CLIENT, "fetch_ticker"):
        try:
            ticker = PUBLIC_CCXT_CLIENT.fetch_ticker(ccxt_symbol)
            for key in ("last", "mark", "close"):
                if ticker.get(key) not in (None, ""):
                    return float(ticker[key])
            info = ticker.get("info") if isinstance(ticker, dict) else {}
            if isinstance(info, dict):
                candidate = info.get("markPrice") or info.get("price")
                if candidate not in (None, ""):
                    return float(candidate)
        except Exception:
            pass
    client = get_live_client()
    if client is not None:
        try:
            data = client.futures_mark_price(symbol=symbol_norm)
            price = data.get("markPrice") if isinstance(data, dict) else None
            if price not in (None, ""):
                return float(price)
        except Exception:
            pass
        try:
            ticker = client.futures_symbol_ticker(symbol=symbol_norm)
            if isinstance(ticker, dict):
                for key in ("price", "lastPrice", "markPrice"):
                    value = ticker.get(key)
                    if value not in (None, ""):
                        return float(value)
        except Exception:
            pass
    return None


def close_position_hard(
    exchange: Any,
    symbol: str,
    side_pos: str,
    pos_qty: float,
    price_precision: Optional[int] = None,
    qty_precision: Optional[int] = None,
) -> Dict[str, Any]:
    norm_symbol = _normalize_symbol(symbol)
    if exchange is None:
        return {"ok": False, "msg": "No hay exchange configurado"}
    if abs(float(pos_qty or 0.0)) < 1e-12:
        return {"ok": True, "msg": "No hay posición para cerrar."}

    rules = _extract_symbol_rules(exchange, norm_symbol)
    if qty_precision is None:
        qty_precision = int(rules.get("qty_precision") or 0)
    if price_precision is None:
        price_precision = int(rules.get("price_precision") or 2)

    raw_qty = abs(float(pos_qty or 0.0))
    qty = _quantize_qty(raw_qty, rules)
    if qty <= 0:
        return {"ok": False, "msg": "Qty redondeada a 0; revisar lotes/qty_precision."}

    close_side = "BUY" if str(side_pos).upper() == "SHORT" else "SELL"
    client_oid = f"bot-close-{int(time.time())}"
    try:
        order = exchange.futures_create_order(
            symbol=norm_symbol,
            side=close_side,
            type="MARKET",
            quantity=qty,
            reduceOnly=True,
            newClientOrderId=client_oid,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    attempts: list[Dict[str, Any]] = []
    residual_error: Optional[str] = None

    try:
        remaining = _fetch_live_position(exchange, norm_symbol)
    except Exception:
        remaining = None

    tolerance = max(float(rules.get("step_size") or 0.0), float(rules.get("min_qty") or 0.0))
    if tolerance <= 0:
        tolerance = 1e-12

    if remaining is not None:
        rem_side = str(remaining.get("side") or "").upper()
        rem_qty = _quantize_qty(float(remaining.get("qty") or 0.0), rules)
        if rem_side == str(side_pos).upper() and rem_qty > tolerance:
            try:
                attempts.append(
                    exchange.futures_create_order(
                        symbol=norm_symbol,
                        side=close_side,
                        type="MARKET",
                        quantity=rem_qty,
                        reduceOnly=True,
                        newClientOrderId=f"{client_oid}-r",
                    )
                )
                remaining = _fetch_live_position(exchange, norm_symbol)
            except Exception as exc:  # pragma: no cover - dependiente del exchange
                residual_error = str(exc)

    summary: Dict[str, Any] = {
        "ok": True,
        "symbol": norm_symbol,
        "side": close_side,
        "qty": float(qty),
        "order": order,
        "msg": f"Enviada orden MARKET {close_side} reduceOnly qty={qty}",
    }
    price = _infer_fill_price(order, None)
    if price is not None:
        summary["price"] = round(float(price), price_precision)
    if attempts:
        summary["extra_orders"] = attempts
    if residual_error:
        summary["residual_error"] = residual_error
    if remaining and float(remaining.get("qty") or 0.0) > tolerance:
        summary["residual_qty"] = float(remaining.get("qty") or 0.0)
        summary["residual_side"] = str(remaining.get("side") or "")
    return summary


def bootstrap_real_state(
    exchange: Optional[Any] = None,
    store: Optional[Any] = None,
    symbol: Optional[str] = None,
) -> None:
    runtime_mode = (runtime_get_mode() or "paper").lower()
    if runtime_mode not in {"real", "live"}:
        return

    client = exchange or get_live_client()
    if client is None:
        logger.debug("bootstrap_real_state: no hay cliente live disponible")
        return

    target_symbol = symbol or getattr(S, "symbol", None) or "BTC/USDT"
    rules = _extract_symbol_rules(client, target_symbol)
    live = _fetch_live_position(client, _normalize_symbol(target_symbol))

    active_store = store
    if active_store is None and POSITION_SERVICE is not None:
        active_store = getattr(POSITION_SERVICE, "store", None)

    global _LAST_REAL_SYNC

    if live:
        side_live = str(live.get("side") or "LONG").upper()
        qty_live = float(live.get("qty") or 0.0)
        signed_qty = _quantize_qty(qty_live, rules)
        if side_live == "SHORT":
            signed_qty = -signed_qty
        avg_price = float(live.get("entry_price") or 0.0)
        lev = float(live.get("leverage") or 1.0)

        if active_store is not None:
            try:
                active_store.save(pos_qty=float(signed_qty), avg_price=avg_price)
            except Exception as exc:
                _warn(
                    "TRADING",
                    "bootstrap_real_state: no se pudo actualizar el store",
                    exc=exc,
                    level="debug",
                )

        try:
            on_open_filled(
                target_symbol,
                side_live,
                abs(float(signed_qty)),
                avg_price,
                lev,
                mode="live",
            )
        except Exception as exc:
            _warn(
                "TRADING",
                "bootstrap_real_state: no se pudo persistir posición live",
                exc=exc,
                level="debug",
            )

        current_sync = (abs(float(signed_qty)), avg_price)
        if _LAST_REAL_SYNC != current_sync:
            logger.info(
                "Sincronizado estado REAL con exchange: qty=%.6f, avg=%.2f",
                *current_sync,
            )
            _LAST_REAL_SYNC = current_sync
        else:
            logger.debug("Sincronizado estado REAL con exchange: sin cambios")
    else:
        if active_store is not None:
            try:
                active_store.save(pos_qty=0.0, avg_price=0.0)
            except Exception as exc:
                _warn(
                    "TRADING",
                    "bootstrap_real_state: no se pudo limpiar el store",
                    exc=exc,
                    level="debug",
                )
        _LAST_REAL_SYNC = None
        logger.debug("Sincronizado estado REAL con exchange: sin posición abierta.")


def _config_uses_hedge(raw_config: Mapping[str, Any] | None) -> bool:
    """Determina si la configuración global habilita el hedge mode."""

    if not isinstance(raw_config, Mapping):
        return True

    exchange_cfg = raw_config.get("exchange")
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

    limits_cfg = raw_config.get("limits")
    if isinstance(limits_cfg, Mapping):
        if "no_hedge" in limits_cfg:
            return not bool(limits_cfg.get("no_hedge"))

    return True


def _build_public_ccxt() -> Optional[Any]:
    """
    Crea SIEMPRE binanceusdm (UM Futures).
    En REAL: setea apiKey/secret y sandbox según BINANCE_UMFUTURES_TESTNET.
    En SIM: público (sin keys), pero sigue siendo UM Futures.
    """
    try:
        import ccxt  # type: ignore
    except ImportError:
        logger.warning("ccxt no está disponible. Sin precios/privado por ccxt.")
        return None

    try:
        try:
            import certifi

            cert_path = certifi.where()
            if os.path.exists(cert_path):
                for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
                    current = os.getenv(env_var)
                    if not current or not os.path.exists(current):
                        os.environ[env_var] = cert_path
            else:  # pragma: no cover - defensive log
                logger.debug("Certifi path not found: %s", cert_path)
        except ImportError as exc:
            _warn(
                "TRADING",
                "certifi no está disponible; se usa la configuración TLS por defecto",
                exc=exc,
                level="debug",
            )

        options: dict[str, Any] = {"defaultType": "future"}
        if _config_uses_hedge(RAW_CONFIG):
            options["hedgeMode"] = True

        exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": options})

        from config import S

        use_testnet = os.getenv("BINANCE_UMFUTURES_TESTNET", "false").lower() == "true"
        runtime_mode = runtime_get_mode()
        if ((runtime_mode == "paper") or use_testnet) and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)

        if runtime_mode != "paper":
            exchange.apiKey = S.binance_api_key
            exchange.secret = S.binance_api_secret
        return exchange
    except Exception as exc:
        logger.warning("No se pudo construir ccxt/binanceusdm: %s", exc)
        return None


def force_refresh_clients():
    global PUBLIC_CCXT_CLIENT
    try:
        PUBLIC_CCXT_CLIENT = _build_public_ccxt()
    except Exception as exc:  # pragma: no cover - defensive
        _warn("TRADING", "No se pudo recrear ccxt", exc=exc, level="debug")


def _sync_settings_mode(mode: Mode) -> None:
    setattr(S, "trading_mode", mode)
    if mode == "real":
        key = (
            os.getenv("BINANCE_KEY")
            or os.getenv("BINANCE_API_KEY")
            or os.getenv("BINANCE_API_KEY_REAL")
            or getattr(S, "binance_api_key", None)
        )
        secret = (
            os.getenv("BINANCE_SECRET")
            or os.getenv("BINANCE_API_SECRET")
            or os.getenv("BINANCE_API_SECRET_REAL")
            or getattr(S, "binance_api_secret", None)
        )
        if key:
            setattr(S, "binance_api_key", key)
        if secret:
            setattr(S, "binance_api_secret", secret)


def rebuild(mode: Mode) -> None:
    global BROKER, POSITION_SERVICE, PUBLIC_CCXT_CLIENT, ACTIVE_MODE, _INITIALIZED, ACTIVE_DATA_DIR, ACTIVE_STORE_PATH
    ACTIVE_MODE = mode
    _sync_settings_mode(mode)
    PUBLIC_CCXT_CLIENT = _build_public_ccxt()
    BROKER = build_broker(S, client_factory)

    ACTIVE_DATA_DIR = get_data_dir()
    if mode == "simulado":
        bot_store: Optional[PaperStore] = ACTIVE_PAPER_STORE or PaperStore(
            path=get_paper_store_path(), start_equity=S.start_equity
        )
    else:
        bot_store = ACTIVE_PAPER_STORE

    shared_store = getattr(BROKER, "store", None)
    if shared_store is not None:
        bot_store = shared_store

    if bot_store is not None:
        try:
            ACTIVE_STORE_PATH = Path(bot_store.path).resolve()
        except Exception:
            ACTIVE_STORE_PATH = None
    else:
        ACTIVE_STORE_PATH = None

    POSITION_SERVICE = PositionService(
        paper_store=bot_store,
        live_client=ACTIVE_LIVE_CLIENT,
        ccxt_client=PUBLIC_CCXT_CLIENT,
        symbol=S.symbol if hasattr(S, "symbol") else "BTC/USDT",
        mode=mode,
    )
    if mode != "simulado":
        try:
            bootstrap_real_state(symbol=getattr(POSITION_SERVICE, "symbol", None))
        except Exception as exc:
            _warn("TRADING", "bootstrap_real_state falló durante rebuild", exc=exc, level="debug")
    logger.debug(
        "Trading stack reconstruido para modo %s (data_dir=%s, store=%s)",
        mode.upper(),
        ACTIVE_DATA_DIR,
        ACTIVE_STORE_PATH,
    )
    _INITIALIZED = True


def ensure_initialized(mode: Mode | None = None) -> None:
    """Inicializa el stack de trading solo una vez (o para un modo específico)."""

    global _INITIALIZED, ACTIVE_MODE

    target_mode = mode or get_mode()
    if _INITIALIZED and mode is None and ACTIVE_MODE == target_mode:
        return

    rebuild(target_mode)


def position_status() -> dict[str, Any]:
    ensure_initialized()
    if POSITION_SERVICE is None:
        return {"side": "FLAT"}
    try:
        return POSITION_SERVICE.get_status()
    except Exception as exc:
        _warn("TRADING", "position_status falló", exc=exc, level="debug")
        return {"side": "FLAT"}


def set_trading_mode(new_mode: Mode, *, source: str = "unknown") -> ModeResult:
    from bot.mode_manager import safe_switch

    normalized_source = str(source or "unknown")
    logger.info(
        "Cambio de modo solicitado: %s (source=%s)",
        new_mode,
        normalized_source,
    )

    class _Services:
        @staticmethod
        def position_status() -> dict[str, Any]:
            return position_status()

        @staticmethod
        def rebuild(mode: Mode) -> None:
            rebuild(mode)

    result = safe_switch(new_mode, _Services)
    if result.ok:
        global LAST_MODE_CHANGE_SOURCE
        LAST_MODE_CHANGE_SOURCE = normalized_source
        logger.info(
            "Modo activo: %s (source=%s)",
            result.mode or new_mode,
            normalized_source,
        )
    else:
        logger.warning(
            "No se pudo cambiar a %s (source=%s): %s",
            new_mode,
            normalized_source,
            result.msg,
        )
    return result


def switch_mode(new_mode: Mode) -> ModeResult:
    return set_trading_mode(new_mode, source="legacy")


def _extract_filled_qty(order_result: Any) -> float:
    if not isinstance(order_result, dict):
        return 0.0
    for key in (
        "executedQty",
        "filled",
        "size",
        "qty",
        "quantity",
        "amount",
    ):
        value = order_result.get(key)
        if value in (None, ""):
            continue
        try:
            qty_val = float(value)
        except Exception:
            continue
        if qty_val > 0:
            return qty_val
    fills = order_result.get("fills")
    if isinstance(fills, list):
        total = 0.0
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            raw = (
                fill.get("qty")
                or fill.get("quantity")
                or fill.get("size")
                or fill.get("amount")
                or fill.get("executedQty")
            )
            if raw in (None, ""):
                continue
            try:
                total += float(raw)
            except Exception:
                continue
        if total > 0:
            return total
    return 0.0


def place_order_safe(side: str, qty: float, price: float | None = None, **kwargs):
    ensure_initialized()
    with _ENTRY_MUTEX:
        try:
            status = POSITION_SERVICE.get_status() if POSITION_SERVICE else None
        except Exception:
            status = None
        if status and str(status.get("side", "FLAT")).upper() != "FLAT":
            raise RuntimeError(
                "Bloqueado: ya hay una posición abierta por el bot. Cerrá antes de abrir otra."
            )
        logger.info(
            "ORDER PATH: %s",
            "PAPER/SimBroker" if ACTIVE_MODE == "simulado" else "LIVE/Binance",
        )
        if BROKER is None:
            raise RuntimeError("Broker no inicializado")

        desired_type = kwargs.get("order_type")
        if desired_type is not None:
            desired_type = str(desired_type).upper()
            kwargs["order_type"] = desired_type
            if desired_type == "MARKET":
                price = None
        elif price in (None, 0, "0"):
            kwargs["order_type"] = "MARKET"
            price = None

        result = BROKER.place_order(side, qty, price, **kwargs)
        inferred_price = _infer_fill_price(result, price)
        symbol = kwargs.get("symbol") or getattr(S, "symbol", None) or "BTC/USDT"
        try:
            lev_source = (
                kwargs.get("leverage")
                or kwargs.get("lev")
                or getattr(S, "leverage", None)
                or getattr(S, "default_leverage", None)
                or getattr(S, "leverage_default", None)
                or 1.0
            )
            lev_value = float(lev_source)
        except Exception:
            lev_value = 1.0
        tp_value = kwargs.get("tp")
        sl_value = kwargs.get("sl")
        mode_label = "live" if str(ACTIVE_MODE).lower() == "real" else "paper"
        try:
            if POSITION_SERVICE is not None and getattr(POSITION_SERVICE, "store", None):
                if isinstance(result, dict) and result.get("sim"):
                    try:
                        POSITION_SERVICE.refresh()
                        logging.info("paper refresh -> %s", POSITION_SERVICE.get_status())
                    except Exception as exc:
                        _warn("TRADING", "paper refresh falló (store)", exc=exc)
                else:
                    fill_price = _infer_fill_price(result, price)
                    if fill_price is not None:
                        fill_side = "LONG" if str(side).upper() in {"BUY", "LONG"} else "SHORT"
                        POSITION_SERVICE.apply_fill(fill_side, float(qty), float(fill_price))
                        logging.info(
                            "apply_fill(open): side=%s qty=%.6f price=%.2f -> %s",
                            fill_side,
                            float(qty),
                            float(fill_price),
                            POSITION_SERVICE.get_status(),
                        )
                        inferred_price = fill_price
                    if inferred_price is not None:
                        fee_paid = _extract_order_fee(result)
                        try:
                            on_open_filled(
                                symbol,
                                "LONG" if str(side).upper() in {"BUY", "LONG"} else "SHORT",
                                float(qty),
                                float(inferred_price),
                                lev_value,
                                tp=tp_value,
                                sl=sl_value,
                                mode=mode_label,
                                fee=fee_paid,
                            )
                        except Exception as exc:
                            _warn(
                                "TRADING",
                                "No se pudo persistir estado en state_store al abrir.",
                                exc=exc,
                                level="debug",
                            )
        except Exception as exc:
            if isinstance(result, dict) and result.get("sim"):
                _warn("TRADING", "PAPER: no se pudo reflejar estado tras abrir.", exc=exc)
            else:
                _warn(
                    "TRADING", "No se pudo reflejar fill en store tras abrir.", exc=exc, level="debug"
                )
        metrics = _METRICS
        if metrics is not None:
            metrics.record_order_sent()
            filled_qty = _extract_filled_qty(result)
            if filled_qty > 0:
                metrics.record_order_filled(filled_qty)
        return result


def close_now(symbol: str | None = None):
    """Cierra la posición del BOT con orden MARKET reduce-only usando la qty propia."""

    ensure_initialized()
    if POSITION_SERVICE is None or BROKER is None:
        raise RuntimeError("No hay servicios activos para cerrar.")

    status = POSITION_SERVICE.get_status() or {}
    side = (status.get("side") or "FLAT").upper()
    qty = float(status.get("qty") or status.get("pos_qty") or 0.0)
    qty = abs(qty)

    if side == "FLAT" or qty <= 0:
        return {"status": "noop", "msg": "Sin posición para cerrar"}

    target_symbol = (
        symbol
        or status.get("symbol")
        or getattr(S, "symbol", None)
        or "BTC/USDT"
    )

    close_side = "SELL" if side == "LONG" else "BUY"

    hedged = _config_uses_hedge(RAW_CONFIG)
    broker_client = getattr(BROKER, "client", None)
    client_options = getattr(broker_client, "options", None) if broker_client else None
    if isinstance(client_options, dict):
        if "hedgeMode" in client_options:
            hedged = bool(client_options.get("hedgeMode"))
        elif "hedge_mode" in client_options:
            hedged = bool(client_options.get("hedge_mode"))
    order_kwargs = {
        "order_type": "market",
        "symbol": target_symbol,
        "reduceOnly": True,
        "reduce_only": True,
        "newOrderRespType": "RESULT",
    }
    if hedged and side in {"LONG", "SHORT"}:
        order_kwargs["positionSide"] = side

    result_payload = BROKER.place_order(close_side, qty, None, **order_kwargs)

    fallback_px = get_latest_price(target_symbol)
    if fallback_px is None:
        fallback_px = status.get("mark") or status.get("mark_price")
    close_price = _infer_fill_price(result_payload, fallback_px)
    effective_qty = qty

    try:
        if POSITION_SERVICE is not None and getattr(POSITION_SERVICE, "store", None):
            is_sim = isinstance(result_payload, dict) and result_payload.get("sim")
            if is_sim:
                try:
                    POSITION_SERVICE.refresh()
                except Exception as exc:
                    _warn("TRADING", "PAPER: refresh tras cierre falló", exc=exc)
            elif close_price is not None:
                bot_side = "SHORT" if side == "LONG" else "LONG"
                POSITION_SERVICE.apply_fill(
                    bot_side,
                    float(effective_qty),
                    float(close_price),
                )
        fee_paid = _extract_order_fee(result_payload)
        if close_price is not None:
            try:
                on_close_filled(str(target_symbol), float(close_price), fee=fee_paid)
            except Exception as exc:
                _warn("TRADING", "No se pudo persistir cierre en state_store.", exc=exc, level="debug")
    except Exception as exc:
        _warn("TRADING", "No se pudo reflejar cierre en store.", exc=exc, level="debug")

    summary: dict[str, Any] = {
        "status": "ok",
        "order": result_payload,
        "symbol": target_symbol,
        "side": side,
        "qty": float(effective_qty),
    }
    if close_price is not None:
        summary["close_price"] = float(close_price)
    return summary


def _infer_fill_price(order_result: Any, fallback: float | None = None) -> float | None:
    """Intenta inferir el precio de fill a partir de la respuesta del broker."""
    if fallback is not None:
        try:
            fallback_f = float(fallback)
        except Exception:
            fallback_f = None
        else:
            if fallback_f and fallback_f > 0:
                return fallback_f
    if isinstance(order_result, dict):
        for key in ("avgPrice", "avg_price", "avgExecutionPrice", "price", "fills_price"):
            value = order_result.get(key)
            if value not in (None, ""):
                try:
                    candidate = float(value)
                except Exception:
                    continue
                if candidate and candidate > 0:
                    return candidate
        fills = order_result.get("fills")
        if isinstance(fills, list) and fills:
            total_qty = 0.0
            total_notional = 0.0
            for fill in fills:
                try:
                    f_price = float(fill.get("price"))
                    qty_candidate = (
                        fill.get("qty")
                        or fill.get("quantity")
                        or fill.get("size")
                        or fill.get("amount")
                        or fill.get("executedQty")
                        or fill.get("filled")
                    )
                    f_qty = float(qty_candidate)
                except Exception:
                    continue
                if f_price <= 0 or abs(f_qty) <= 0:
                    continue
                total_qty += abs(f_qty)
                total_notional += abs(f_qty) * f_price
            if total_qty > 0:
                return total_notional / total_qty
    return None


def _extract_order_fee(order_result: Any) -> float:
    """Extrae el fee total (en USDT) de la respuesta del broker."""

    total_fee = 0.0
    if not isinstance(order_result, dict):
        return total_fee

    direct_keys = ("fee", "commission", "commissionAmount", "commission_amount")
    for key in direct_keys:
        if key in order_result and order_result[key] not in (None, ""):
            try:
                total_fee += abs(float(order_result[key]))
            except Exception:
                continue

    fees_field = order_result.get("fees")
    if isinstance(fees_field, (list, tuple)):
        for entry in fees_field:
            if isinstance(entry, dict):
                for key in ("fee", "commission", "cost"):
                    if entry.get(key) not in (None, ""):
                        try:
                            total_fee += abs(float(entry[key]))
                        except Exception:
                            continue
            else:
                try:
                    total_fee += abs(float(entry))
                except Exception:
                    continue

    fills = order_result.get("fills")
    if isinstance(fills, list):
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            for key in ("commission", "fee", "cost"):
                if fill.get(key) in (None, ""):
                    continue
                try:
                    total_fee += abs(float(fill.get(key)))
                except Exception:
                    continue

    return total_fee


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace(":USDT", "").upper()


def close_bot_position_market() -> dict[str, Any]:
    """Cierra la posición del bot mediante una orden MARKET reduce-only."""

    ensure_initialized()
    if BROKER is None:
        raise RuntimeError("Broker no inicializado.")

    status = POSITION_SERVICE.get_status() if POSITION_SERVICE else None
    if not status:
        return {"status": "noop", "reason": "no_position"}

    side = str(status.get("side", "FLAT")).upper()
    qty_raw = status.get("qty") or status.get("pos_qty") or 0.0
    try:
        qty_now = abs(float(qty_raw))
    except Exception as exc:
        raise RuntimeError("Cantidad inválida para cerrar posición.") from exc
    if side == "FLAT" or qty_now <= 0:
        return {"status": "noop", "reason": "no_position"}

    symbol_conf = status.get("symbol") or getattr(S, "symbol", None) or "BTC/USDT"
    mark_price = status.get("mark") or status.get("mark_price")
    hedged = _config_uses_hedge(RAW_CONFIG)

    kwargs = {
        "symbol": symbol_conf,
        "reduceOnly": True,
        "reduce_only": True,
        "newOrderRespType": "RESULT",
    }
    if hedged and side in {"LONG", "SHORT"}:
        kwargs["positionSide"] = side

    close_side = "SELL" if side == "LONG" else "BUY"

    try:
        if hasattr(BROKER, "create_order"):
            result = BROKER.create_order(close_side, qty_now, None, **kwargs)
        else:
            result = BROKER.place_order(close_side, qty_now, None, **kwargs)
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}

    fallback_px = get_latest_price(symbol_conf)
    if fallback_px is None:
        fallback_px = mark_price
    close_price = _infer_fill_price(result, fallback_px)
    summary = {
        "status": "closed",
        "side": side,
        "qty": float(qty_now),
        "price": close_price,
        "order": result,
        "symbol": symbol_conf,
        "symbol_normalized": _normalize_symbol(symbol_conf),
    }
    return summary
