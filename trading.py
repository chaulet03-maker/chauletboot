from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from bot.mode_manager import Mode, ModeResult, get_mode
from config import RAW_CONFIG, S
from brokers import ACTIVE_LIVE_CLIENT, ACTIVE_PAPER_STORE, build_broker
from binance_client import client_factory
from position_service import PositionService
from paper_store import PaperStore
from state_store import on_close_filled, on_open_filled

logger = logging.getLogger(__name__)

BROKER: Any | None = None
POSITION_SERVICE: PositionService | None = None
PUBLIC_CCXT_CLIENT: Optional[Any] = None
ACTIVE_MODE: Mode = "simulado"
_INITIALIZED: bool = False


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
        except ImportError:
            logger.debug("certifi no está disponible; se usa la configuración TLS por defecto")

        options: dict[str, Any] = {"defaultType": "future"}
        if _config_uses_hedge(RAW_CONFIG):
            options["hedgeMode"] = True

        exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": options})

        from config import S

        use_testnet = os.getenv("BINANCE_UMFUTURES_TESTNET", "false").lower() == "true"
        if (S.PAPER or use_testnet) and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)

        if not S.PAPER:
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
        logger.debug('No se pudo recrear ccxt: %s', exc)


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
    global BROKER, POSITION_SERVICE, PUBLIC_CCXT_CLIENT, ACTIVE_MODE, _INITIALIZED
    ACTIVE_MODE = mode
    _sync_settings_mode(mode)
    PUBLIC_CCXT_CLIENT = _build_public_ccxt()
    BROKER = build_broker(S, client_factory)
    bot_store = ACTIVE_PAPER_STORE
    if mode != "simulado":
        try:
            os.makedirs("data", exist_ok=True)
        except Exception:
            pass
        live_store_path = os.path.join(os.getcwd(), "data", "live_bot_position.json")
        bot_store = PaperStore(path=live_store_path, start_equity=S.start_equity)

    shared_store = getattr(BROKER, "store", None)
    if shared_store is not None:
        bot_store = shared_store

    POSITION_SERVICE = PositionService(
        paper_store=bot_store,
        live_client=ACTIVE_LIVE_CLIENT,
        ccxt_client=PUBLIC_CCXT_CLIENT,
        symbol=S.symbol if hasattr(S, "symbol") else "BTC/USDT",
    )
    logger.info("Trading stack reconstruido para modo %s", mode.upper())
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
        logger.debug("position_status falló: %s", exc)
        return {"side": "FLAT"}


def switch_mode(new_mode: Mode) -> ModeResult:
    from bot.mode_manager import safe_switch

    class _Services:
        @staticmethod
        def position_status() -> dict[str, Any]:
            return position_status()

        @staticmethod
        def rebuild(mode: Mode) -> None:
            rebuild(mode)

    return safe_switch(new_mode, _Services)


def place_order_safe(side: str, qty: float, price: float | None = None, **kwargs):
    ensure_initialized()
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
                except Exception:
                    logging.warning("paper refresh falló (store)", exc_info=True)
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
                    except Exception:
                        logger.debug(
                            "No se pudo persistir estado en state_store al abrir.",
                            exc_info=True,
                        )
    except Exception:
        if isinstance(result, dict) and result.get("sim"):
            logger.warning("PAPER: no se pudo reflejar estado tras abrir.", exc_info=True)
        else:
            logger.debug("No se pudo reflejar fill en store tras abrir.", exc_info=True)
    return result


def close_now(symbol: str | None = None):
    """Cierra la posición actual de inmediato usando orden MARKET compatible con hedge."""

    ensure_initialized()
    if POSITION_SERVICE is None or BROKER is None:
        raise RuntimeError("No hay servicios activos para cerrar.")

    status = POSITION_SERVICE.get_status() or {}
    side = (status.get("side") or "FLAT").upper()
    if side == "FLAT":
        return {"status": "noop", "msg": "Sin posición para cerrar"}

    qty = float(status.get("qty") or status.get("pos_qty") or 0.0)
    if qty <= 0:
        return {"status": "noop", "msg": "Qty=0"}

    close_side = "SELL" if side == "LONG" else "BUY"
    target_symbol = symbol or status.get("symbol")
    hedged = _config_uses_hedge(RAW_CONFIG)
    # Siempre cerrar con reduceOnly; en hedge además explicitar el lado actual
    kwargs = dict(order_type="market", symbol=target_symbol, reduce_only=True)
    if hedged and side in {"LONG", "SHORT"}:
        kwargs["positionSide"] = side

    # Sincronizar con posición viva si hay cliente activo (importante en REAL)
    if ACTIVE_LIVE_CLIENT is not None:
        try:
            sym_id = (target_symbol or "BTC/USDT").replace("/", "")
            live = ACTIVE_LIVE_CLIENT.futures_position_information(symbol=sym_id)
            live_amt = 0.0
            for p in live or []:
                if str(p.get("symbol") or "").upper() == sym_id.upper():
                    live_amt = float(p.get("positionAmt") or 0.0)
                    break
            live_qty = abs(live_amt)
            if live_qty <= 0.0:
                return {"status": "noop", "msg": "No live position"}
            if qty > live_qty + 1e-12:
                qty = live_qty
        except Exception:
            # si falla el fetch, seguimos con qty local (mejor que no cerrar nada)
            pass

    result = BROKER.place_order(close_side, qty, None, **kwargs)
    close_price = _infer_fill_price(result, status.get("mark"))
    try:
        if POSITION_SERVICE is not None and getattr(POSITION_SERVICE, "store", None):
            if isinstance(result, dict) and result.get("sim"):
                try:
                    POSITION_SERVICE.refresh()
                except Exception:
                    logger.warning("PAPER: refresh tras cierre falló", exc_info=True)
            else:
                bot_side = "SHORT" if side == "LONG" else "LONG"
                if close_price is not None:
                    POSITION_SERVICE.apply_fill(bot_side, float(qty), float(close_price))
        fee_paid = _extract_order_fee(result)
        if close_price is not None:
            try:
                target = target_symbol or symbol or getattr(S, "symbol", None) or "BTC/USDT"
                on_close_filled(str(target), float(close_price), fee=fee_paid)
            except Exception:
                logger.debug("No se pudo persistir cierre en state_store.", exc_info=True)
    except Exception:
        if isinstance(result, dict) and result.get("sim"):
            logger.warning("PAPER: no se pudo reflejar cierre en store.", exc_info=True)
        else:
            logger.debug("No se pudo reflejar cierre en store.", exc_info=True)
    summary: dict[str, Any] = {
        "status": "ok",
        "order": result,
        "symbol": target_symbol,
        "side": side,
        "qty": float(qty),
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

    close_price = _infer_fill_price(result, mark_price)
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
