"""Shared CCXT client helpers and position mode utilities."""

import logging
import os
from typing import Any

import ccxt

logger = logging.getLogger(__name__)

_CCXT = None


def get_ccxt() -> ccxt.Exchange:
    global _CCXT
    if _CCXT is not None:
        return _CCXT
    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("binance_api_key")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("binance_api_secret")
    ex = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    ex.load_markets(reload=True)
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
