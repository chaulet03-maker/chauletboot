from __future__ import annotations

import logging
from typing import Any, Optional

from config import S
from brokers import build_broker, ACTIVE_PAPER_STORE, ACTIVE_LIVE_CLIENT
from binance_client import client_factory
from position_service import PositionService

logger = logging.getLogger(__name__)


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


PUBLIC_CCXT_CLIENT = _build_public_ccxt()

BROKER = build_broker(S, client_factory)

POSITION_SERVICE = PositionService(
    paper_store=ACTIVE_PAPER_STORE,
    live_client=ACTIVE_LIVE_CLIENT,
    ccxt_client=PUBLIC_CCXT_CLIENT,
    symbol="BTC/USDT",
)


def place_order_safe(side: str, qty: float, price: float, **kwargs):
    logger.info("ORDER PATH: %s", "PAPER/SimBroker" if S.PAPER else "LIVE/Binance")
    return BROKER.place_order(side, qty, price, **kwargs)
