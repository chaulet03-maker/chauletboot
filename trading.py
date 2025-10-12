from __future__ import annotations

import logging

from config import S
from brokers import build_broker, ACTIVE_PAPER_STORE, ACTIVE_LIVE_CLIENT
from binance_client import client_factory
from position_service import build_position_service

logger = logging.getLogger(__name__)

BROKER = build_broker(S, client_factory)
POSITION_SERVICE = build_position_service(
    S,
    store=ACTIVE_PAPER_STORE,
    client=ACTIVE_LIVE_CLIENT,
    symbol="BTC/USDT",
)


def place_order_safe(side: str, qty: float, price: float, **kwargs):
    logger.info("ORDER PATH: %s", "PAPER/SimBroker" if S.PAPER else "LIVE/Binance")
    return BROKER.place_order(side, qty, price, **kwargs)
