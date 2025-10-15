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
_INITIALIZED: bool = False


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
        exchange = ccxt.binanceusdm(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )

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
    POSITION_SERVICE = PositionService(
        paper_store=ACTIVE_PAPER_STORE,
        live_client=ACTIVE_LIVE_CLIENT,
        ccxt_client=PUBLIC_CCXT_CLIENT,
        symbol="BTC/USDT",
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
    return BROKER.place_order(side, qty, price, **kwargs)


def close_now(symbol: str | None = None):
    """Cierra la posición actual de inmediato usando orden MARKET reduce-only."""

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
    return BROKER.place_order(
        close_side,
        qty,
        None,
        reduce_only=True,
        order_type="market",
        symbol=target_symbol,
    )
