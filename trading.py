from __future__ import annotations

import logging
import os
from typing import Any, Optional

from bot.mode_manager import Mode, ModeResult, get_mode
from config import S
from brokers import ACTIVE_LIVE_CLIENT, ACTIVE_PAPER_STORE, build_broker
from binance_client import client_factory
from position_service import PositionService

logger = logging.getLogger(__name__)

BROKER: Any | None = None
POSITION_SERVICE: PositionService | None = None
PUBLIC_CCXT_CLIENT: Optional[Any] = None
ACTIVE_MODE: Mode = "simulado"


def _build_public_ccxt() -> Optional[Any]:
    try:
        import ccxt  # type: ignore
    except ImportError:
        logger.warning(
            "ccxt no está disponible. PositionService no podrá refrescar precios públicos."
        )
        return None

    try:
        return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    except Exception as exc:
        logger.warning("No se pudo inicializar cliente público ccxt: %s", exc)
        return None


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
    global BROKER, POSITION_SERVICE, PUBLIC_CCXT_CLIENT, ACTIVE_MODE
    ACTIVE_MODE = mode
    _sync_settings_mode(mode)
    PUBLIC_CCXT_CLIENT = _build_public_ccxt()
    BROKER = build_broker(S, client_factory)
    POSITION_SERVICE = PositionService(
        paper_store=ACTIVE_PAPER_STORE,
        live_client=ACTIVE_LIVE_CLIENT,
        ccxt_client=PUBLIC_CCXT_CLIENT,
        symbol="BTC/USDT",
    )
    logger.info("Trading stack reconstruido para modo %s", mode.upper())


# Inicialización al importar el módulo
rebuild(get_mode())


def position_status() -> dict[str, Any]:
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


def place_order_safe(side: str, qty: float, price: float, **kwargs):
    logger.info(
        "ORDER PATH: %s",
        "PAPER/SimBroker" if ACTIVE_MODE == "simulado" else "LIVE/Binance",
    )
    if BROKER is None:
        raise RuntimeError("Broker no inicializado")
    return BROKER.place_order(side, qty, price, **kwargs)
