"""Shared CCXT client helpers and position mode utilities."""

import logging
import os
from typing import Any

import ccxt

logger = logging.getLogger(__name__)

_CCXT = None


def get_ccxt() -> ccxt.Exchange:
    """Crea un singleton CCXT para Binance USDM leyendo .env y admitiendo *_REAL/*_TEST."""
    global _CCXT
    if _CCXT is not None:
        return _CCXT

    # Cargar .env si existe (defensivo)
    try:
        from dotenv import load_dotenv  # no rompe si no está
        load_dotenv()
    except Exception:
        pass

    use_testnet = str(os.getenv("USE_TESTNET", "0")).lower() in ("1", "true")

    api_key = (
        os.getenv("BINANCE_API_KEY", "")
        or os.getenv("BINANCE_API_KEY_REAL", "")
        or (os.getenv("BINANCE_API_KEY_TEST", "") if use_testnet else "")
    ).strip()
    api_secret = (
        os.getenv("BINANCE_API_SECRET", "")
        or os.getenv("BINANCE_API_SECRET_REAL", "")
        or (os.getenv("BINANCE_API_SECRET_TEST", "") if use_testnet else "")
    ).strip()

    if not api_key or not api_secret:
        raise RuntimeError(
            "Faltan credenciales Binance USDM: definí BINANCE_API_KEY/_SECRET (o *_REAL / *_TEST)."
        )

    ex = ccxt.binanceusdm(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "fetchCurrencies": False,
            },
        }
    )

    if use_testnet:
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            pass

    _CCXT = ex
    return ex


def reset_ccxt_client() -> None:
    global _CCXT
    _CCXT = None


def ensure_position_mode(exchange: Any, hedge: bool = False) -> None:
    """No-op; CCXT no expone esto para USDM como API estable. Segura."""
    try:
        _ = exchange.id
    except Exception:
        pass


def normalize_symbol(symbol: str) -> str:
    """Normaliza un símbolo a formato "BASE/QUOTE:QUOTE" (ej. BTC/USDT:USDT)."""

    value = str(symbol or "").strip()
    if not value:
        return ""
    value = value.upper()
    if value.endswith(":USDT"):
        return value
    if value.endswith("/USDT"):
        return f"{value}:USDT"
    if value.endswith("USDT") and "/" not in value:
        base = value[:-4]
        return f"{base}/USDT:USDT"
    return value
