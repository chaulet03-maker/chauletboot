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

    use_testnet = (os.getenv("BINANCE_UMFUTURES_TESTNET", "false").lower() == "true")

    # Pares de variables admitidas (en orden de preferencia según testnet)
    candidates = [
        ("BINANCE_API_KEY_TEST", "BINANCE_API_SECRET_TEST"),
        ("BINANCE_API_KEY_REAL", "BINANCE_API_SECRET_REAL"),
        ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
        ("BINANCE_KEY", "BINANCE_SECRET"),
    ]
    ordered = (
        [candidates[0], candidates[1], candidates[2], candidates[3]] if use_testnet
        else [candidates[1], candidates[2], candidates[3], candidates[0]]
    )

    api_key = api_secret = None
    for k, s in ordered:
        ak = os.getenv(k) or os.getenv(k.lower())
        sk = os.getenv(s) or os.getenv(s.lower())
        if ak and sk:
            api_key, api_secret = ak, sk
            break

    if not api_key or not api_secret:
        raise RuntimeError("Faltan credenciales Binance USDM: definí BINANCE_API_KEY/_SECRET (o *_REAL / *_TEST).")

    params = {
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future", "adjustForTimeDifference": True},
        "recvWindow": 10000,
    }
    ex = ccxt.binanceusdm(params)
    try:
        if use_testnet and hasattr(ex, "set_sandbox_mode"):
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
