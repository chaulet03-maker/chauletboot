from __future__ import annotations

import logging

from config import S
from brokers import build_broker
from binance_client import client_factory

logger = logging.getLogger(__name__)

BROKER = build_broker(S, client_factory)


def place_order_safe(side: str, qty: float, price: float, **kwargs):
    logger.info("ORDER PATH: %s", "PAPER/SimBroker" if S.PAPER else "LIVE/Binance")
    return BROKER.place_order(side, qty, price, **kwargs)
