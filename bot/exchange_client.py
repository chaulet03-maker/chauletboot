"""Shared CCXT client helpers and position mode utilities."""

import logging
import os
import time
from typing import Any, Dict

import ccxt
from ccxt.base.errors import AuthenticationError

from config import S

logger = logging.getLogger(__name__)

_CCXT = None
_POSMODE_CACHE = {"known": None, "ts": 0.0, "ttl": 60.0}


def _clean(s):
    if s is None:
        return ""
    return str(s).strip().strip('"').strip("'").replace("\r", "").replace("\n", "")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def reset_ccxt_client() -> None:
    """Reset the shared CCXT client (mainly for tests or mode switches)."""

    global _CCXT
    _CCXT = None


def get_ccxt():
    """Return a shared CCXT client configured for USD-M futures."""

    global _CCXT
    if _CCXT is not None:
        return _CCXT

    api_key = _clean(
        os.getenv("BINANCE_API_KEY")
        or os.getenv("BINANCE_FUTURES_API_KEY")
        or os.getenv("BINANCE_API_KEY_REAL")
        or getattr(S, "binance_api_key", "")
    )
    secret = _clean(
        os.getenv("BINANCE_API_SECRET")
        or os.getenv("BINANCE_FUTURES_API_SECRET")
        or os.getenv("BINANCE_API_SECRET_REAL")
        or getattr(S, "binance_api_secret", "")
    )
    if not api_key or not secret:
        raise RuntimeError("Faltan credenciales BINANCE_API_KEY / BINANCE_API_SECRET")

    options: Dict[str, Any] = {
        "defaultType": "future",
        "adjustForTimeDifference": True,
        "recvWindow": 60000,
    }

    hedge_hint = getattr(S, "hedge_mode", None)
    if hedge_hint is None:
        hedge_hint = getattr(S, "hedgeMode", None)
    if hedge_hint is not None:
        try:
            options["hedgeMode"] = bool(_to_bool(hedge_hint))
        except Exception:
            options["hedgeMode"] = bool(hedge_hint)

    ex = ccxt.binanceusdm(
        {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "timeout": 20000,
            "options": options,
        }
    )
    try:
        ex.has["fetchCurrencies"] = False
    except Exception:
        pass

    use_testnet = _to_bool(
        os.getenv("BINANCE_UMFUTURES_TESTNET")
        or getattr(S, "binance_umfutures_testnet", None)
    )
    if use_testnet and hasattr(ex, "set_sandbox_mode"):
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            logger.debug("No pude activar sandbox_mode en CCXT", exc_info=True)

    try:
        try:
            ex.load_time_difference()
        except Exception:
            logger.debug("load_time_difference falló; continúo.", exc_info=True)
        ex.load_markets(reload=True)
        logger.info("Cliente CCXT (binanceusdm) inicializado OK.")
    except Exception:
        logger.exception("No pude inicializar CCXT binanceusdm")
        raise

    _CCXT = ex
    return _CCXT


def get_position_mode_cached():
    """True=HEDGE, False=ONE-WAY, None=desconocido (no romper si falla)."""

    now = time.time()
    cached = _POSMODE_CACHE.get("known")
    cached_ts = float(_POSMODE_CACHE.get("ts", 0.0))
    ttl = float(_POSMODE_CACHE.get("ttl", 60.0))
    if cached is not None and (now - cached_ts) < ttl:
        return cached

    ex = get_ccxt()
    try:
        resp = ex.fapiPrivateGetPositionSideDual()
        cur = resp.get("dualSidePosition") if isinstance(resp, dict) else None
        if isinstance(cur, bool):
            current = cur
        else:
            current = str(cur).lower() == "true"
        _POSMODE_CACHE.update(known=current, ts=now)
        return current
    except AuthenticationError as e:
        logger.warning("No puedo leer PositionSideDual (auth). No toco nada. %s", e)
    except Exception:
        logger.debug("No pude leer PositionSideDual", exc_info=True)

    _POSMODE_CACHE.update(known=None, ts=now)
    return None


def ensure_position_mode(_hedged: bool) -> bool:
    """DEPRECATED: no fuerces el modo. Mantengo por compatibilidad: no lanza, no toca."""

    cur = get_position_mode_cached()
    if cur is None:
        logger.info(
            "Modo de posiciones desconocido; no lo cambio (evito /positionSide/dual)."
        )
        return False
    return cur == bool(_hedged)


__all__ = [
    "get_ccxt",
    "reset_ccxt_client",
    "get_position_mode_cached",
    "ensure_position_mode",
]
